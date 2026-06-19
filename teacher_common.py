import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine, rescale_affine
from nibabel.orientations import axcodes2ornt, inv_ornt_aff, ornt_transform
from scipy.ndimage import zoom


_PROGRESS_WIDTHS: Dict[str, int] = {}


def default_teacher_config() -> Dict[str, Any]:
    return {
        "file_suffix": ".nii.gz",
        "orientation": "RAS",
        "target_spacing": [1.5, 1.5, 1.5],
        "intensity": {
            "clip_min": -200.0,
            "clip_max": 250.0,
            "normalize_mode": "zscore_nonzero",
            "percentile_clip": {
                "enabled": False,
                "lower": 0.5,
                "upper": 99.5,
            },
        },
        "crop": {
            "enabled": True,
            "threshold": -150.0,
            "margin_voxels": [12, 24, 24],
        },
        "validation_fraction": 0.2,
    }


def load_teacher_config(config_path: Path | None) -> Dict[str, Any]:
    config = default_teacher_config()
    if config_path is None:
        return config
    with config_path.open("r", encoding="utf-8") as f:
        overrides = json.load(f)
    return merge_dict(config, overrides)


def merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def print_progress(prefix: str, completed: int, total: int, extra: str = "") -> None:
    total_safe = max(1, int(total))
    finished = min(max(0, int(completed)), total_safe)
    progress = float(finished) / float(total_safe)
    suffix = f" {extra}" if extra else ""
    message = f"{prefix} {progress:.3f}/1{suffix}"
    padded = message.ljust(max(_PROGRESS_WIDTHS.get(prefix, 0), len(message)))
    if finished >= total_safe:
        print(f"\r{padded}")
        _PROGRESS_WIDTHS.pop(prefix, None)
    else:
        print(f"\r{padded}", end="", flush=True)
        _PROGRESS_WIDTHS[prefix] = len(padded)


def reorient_to_target(img: nib.Nifti1Image, target_codes: str) -> nib.Nifti1Image:
    source_ornt = nib.orientations.io_orientation(img.affine)
    target_ornt = axcodes2ornt(tuple(target_codes))
    transform = ornt_transform(source_ornt, target_ornt)
    data = img.get_fdata(dtype=np.float32)
    reoriented = nib.orientations.apply_orientation(data, transform)
    new_affine = img.affine @ inv_ornt_aff(transform, img.shape)
    return nib.Nifti1Image(reoriented, new_affine)


def clip_intensity(volume: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    return np.clip(volume, clip_min, clip_max).astype(np.float32)


def percentile_clip(volume: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    lower = float(np.percentile(volume, lower_percentile))
    upper = float(np.percentile(volume, upper_percentile))
    return np.clip(volume, lower, upper).astype(np.float32)


def normalize_intensity(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minmax":
        vmin = float(volume.min())
        vmax = float(volume.max())
        if vmax <= vmin:
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - vmin) / (vmax - vmin)).astype(np.float32)

    if mode == "zscore":
        mean = float(volume.mean())
        std = float(volume.std())
        if std <= 1e-8:
            return np.zeros_like(volume, dtype=np.float32)
        return ((volume - mean) / std).astype(np.float32)

    if mode == "zscore_nonzero":
        mask = volume > np.min(volume)
        if not np.any(mask):
            return np.zeros_like(volume, dtype=np.float32)
        selected = volume[mask]
        mean = float(selected.mean())
        std = float(selected.std())
        if std <= 1e-8:
            normalized = volume - mean
        else:
            normalized = (volume - mean) / std
        return normalized.astype(np.float32)

    if mode == "percentile_minmax":
        lower = float(np.percentile(volume, 1.0))
        upper = float(np.percentile(volume, 99.0))
        if upper <= lower:
            return np.zeros_like(volume, dtype=np.float32)
        clipped = np.clip(volume, lower, upper)
        return ((clipped - lower) / (upper - lower)).astype(np.float32)

    raise ValueError(f"Unsupported normalize mode: {mode}")


def resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))
    resampled = zoom(volume, zoom=zoom_factors, order=order)
    return np.asarray(resampled, dtype=np.float32), target_spacing


def extract_body_bbox(volume: np.ndarray, threshold: float, margin: Tuple[int, int, int]) -> Tuple[slice, slice, slice] | None:
    mask = volume > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    starts = [max(0, int(mn - mg)) for mn, mg in zip(mins, margin)]
    stops = [min(int(sz), int(mx + mg)) for mx, mg, sz in zip(maxs, margin, volume.shape)]
    return tuple(slice(start, stop) for start, stop in zip(starts, stops))


def crop_volume_if_enabled(volume: np.ndarray, crop_config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not crop_config.get("enabled", False):
        return volume.astype(np.float32), {"enabled": False}
    threshold = float(crop_config.get("threshold", -150.0))
    margin = tuple(int(v) for v in crop_config.get("margin_voxels", [12, 24, 24]))
    bbox = extract_body_bbox(volume, threshold, margin)
    if bbox is None:
        return volume.astype(np.float32), {"enabled": True, "applied": False, "reason": "empty_body_mask"}
    cropped = volume[bbox].astype(np.float32)
    bbox_record = [[sl.start, sl.stop] for sl in bbox]
    return cropped, {"enabled": True, "applied": True, "bbox": bbox_record}


def apply_crop_to_affine(affine: np.ndarray, crop_info: Dict[str, Any]) -> np.ndarray:
    if not crop_info.get("enabled") or not crop_info.get("applied"):
        return affine.astype(np.float32)
    bbox = crop_info["bbox"]
    start = np.array([bbox[0][0], bbox[1][0], bbox[2][0]], dtype=np.float32)
    cropped_affine = affine.astype(np.float32).copy()
    cropped_affine[:3, 3] = apply_affine(affine, start)
    return cropped_affine


def build_output_affine(
    source_affine: np.ndarray,
    source_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    return rescale_affine(source_affine, source_shape, target_spacing, new_shape=target_shape).astype(np.float32)


def preprocess_image_and_label(
    image_path: Path,
    label_path: Path,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = nib.load(str(image_path))
    reoriented_image = reorient_to_target(image, str(config["orientation"]))
    image_volume = reoriented_image.get_fdata(dtype=np.float32)
    original_spacing = tuple(float(v) for v in reoriented_image.header.get_zooms()[:3])
    target_spacing = tuple(float(v) for v in config["target_spacing"])

    intensity_config = config["intensity"]
    clipped = clip_intensity(
        image_volume,
        float(intensity_config["clip_min"]),
        float(intensity_config["clip_max"]),
    )
    percentile_clip_config = intensity_config.get("percentile_clip")
    if percentile_clip_config and percentile_clip_config.get("enabled", False):
        clipped = percentile_clip(
            clipped,
            float(percentile_clip_config.get("lower", 0.5)),
            float(percentile_clip_config.get("upper", 99.5)),
        )
    normalized = normalize_intensity(clipped, str(intensity_config["normalize_mode"]))
    normalized, _ = resample_volume(normalized, original_spacing, target_spacing, order=1)
    normalized, crop_info = crop_volume_if_enabled(normalized, config.get("crop", {}))

    label = nib.load(str(label_path))
    reoriented_label = reorient_to_target(label, str(config["orientation"]))
    label_volume = reoriented_label.get_fdata(dtype=np.float32)
    label_volume, _ = resample_volume(label_volume, original_spacing, target_spacing, order=0)
    if crop_info.get("enabled") and crop_info.get("applied"):
        bbox = crop_info["bbox"]
        label_volume = label_volume[
            bbox[0][0] : bbox[0][1],
            bbox[1][0] : bbox[1][1],
            bbox[2][0] : bbox[2][1],
        ]
    processed_label = (np.rint(label_volume) > 0).astype(np.uint8)

    output_affine = build_output_affine(
        reoriented_image.affine,
        tuple(int(v) for v in image_volume.shape),
        tuple(int(v) for v in normalized.shape),
        target_spacing,
    )
    output_affine = apply_crop_to_affine(output_affine, crop_info)
    return normalized.astype(np.float32), processed_label.astype(np.uint8), np.asarray(output_affine, dtype=np.float32)


def save_preprocessed_case(output_path: Path, image: np.ndarray, label: np.ndarray, affine: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    np.savez_compressed(
        str(temp_path),
        image=image.astype(np.float32),
        label=label.astype(np.uint8),
        affine=affine.astype(np.float32),
    )
    os.replace(str(temp_path), str(output_path))


def load_preprocessed_case(input_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(input_path), allow_pickle=False) as data:
        return {
            "image": np.asarray(data["image"], dtype=np.float32),
            "label": np.asarray(data["label"], dtype=np.uint8),
            "affine": np.asarray(data["affine"], dtype=np.float32),
        }


def list_preprocessed_cases(preprocessed_dir: Path) -> List[Path]:
    return sorted(path for path in preprocessed_dir.glob("*.npz") if path.is_file())
