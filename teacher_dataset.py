import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from teacher_common import load_preprocessed_case, preprocess_image_and_label


def list_case_pairs(images_dir: Path, labels_dir: Path, file_suffix: str) -> List[Tuple[str, Path, Path]]:
    image_paths = sorted(path for path in images_dir.glob(f"*{file_suffix}") if path.is_file())
    pairs: List[Tuple[str, Path, Path]] = []
    for image_path in image_paths:
        case_name = image_path.name[: -len(file_suffix)]
        label_path = labels_dir / f"{case_name}{file_suffix}"
        if not label_path.exists():
            continue
        pairs.append((case_name, image_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {images_dir} and {labels_dir}")
    return pairs


def split_case_pairs(case_pairs: Sequence[Tuple[str, Path, Path]], val_fraction: float, seed: int) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    if len(case_pairs) < 2 or val_fraction <= 0.0:
        return list(case_pairs), []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(case_pairs), dtype=np.int32)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(case_pairs) * val_fraction)))
    val_count = min(val_count, len(case_pairs) - 1)
    val_index_set = set(indices[:val_count].tolist())
    train_cases = [pair for idx, pair in enumerate(case_pairs) if idx not in val_index_set]
    val_cases = [pair for idx, pair in enumerate(case_pairs) if idx in val_index_set]
    return train_cases, val_cases


def _bbox_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mins.astype(np.int32), maxs.astype(np.int32)


def build_teacher_cache(
    case_pairs: Sequence[Tuple[str, Path, Path]],
    config: Dict[str, Any],
    cache_dir: Path,
) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []
    case_cache_paths: List[Path] = []

    for case_name, image_path, label_path in case_pairs:
        cache_path = cache_dir / f"{case_name}.npz"
        case_cache_paths.append(cache_path)
        if cache_path.exists():
            with np.load(str(cache_path), allow_pickle=False) as cached:
                manifest_entries.append(
                    {
                        "case_name": case_name,
                        "cache_path": str(cache_path),
                        "shape": [int(v) for v in cached["image"].shape],
                        "cached": True,
                    }
                )
            continue

        image, label, affine = preprocess_image_and_label(image_path, label_path, config)
        bbox = _bbox_from_mask(label)
        if bbox is None:
            bbox_mins = np.zeros(3, dtype=np.int32)
            bbox_maxs = np.asarray(label.shape, dtype=np.int32)
        else:
            bbox_mins, bbox_maxs = bbox

        np.savez_compressed(
            str(cache_path),
            image=image.astype(np.float32),
            label=label.astype(np.uint8),
            affine=affine.astype(np.float32),
            bbox_mins=bbox_mins.astype(np.int32),
            bbox_maxs=bbox_maxs.astype(np.int32),
        )
        manifest_entries.append(
            {
                "case_name": case_name,
                "cache_path": str(cache_path),
                "shape": [int(v) for v in image.shape],
                "cached": False,
            }
        )

    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
    return case_cache_paths


def load_cached_case(cache_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(cache_path), allow_pickle=False) as data:
        return {
            "image": np.asarray(data["image"], dtype=np.float32),
            "label": np.asarray(data["label"], dtype=np.uint8),
            "affine": np.asarray(data["affine"], dtype=np.float32),
            "bbox_mins": np.asarray(data["bbox_mins"], dtype=np.int32),
            "bbox_maxs": np.asarray(data["bbox_maxs"], dtype=np.int32),
        }


def _bbox_from_loaded_case(case: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    label = case["label"]
    bbox = _bbox_from_mask(label)
    if bbox is None:
        bbox_mins = np.zeros(3, dtype=np.int32)
        bbox_maxs = np.asarray(label.shape, dtype=np.int32)
    else:
        bbox_mins, bbox_maxs = bbox
    return {
        "image": case["image"],
        "label": label,
        "affine": case["affine"],
        "bbox_mins": bbox_mins.astype(np.int32),
        "bbox_maxs": bbox_maxs.astype(np.int32),
    }


def load_preprocessed_training_case(case_path: Path) -> Dict[str, np.ndarray]:
    return _bbox_from_loaded_case(load_preprocessed_case(case_path))


def _choose_patch_start(
    rng: np.random.Generator,
    shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    center: np.ndarray | None,
) -> np.ndarray:
    starts = np.zeros(3, dtype=np.int32)
    for axis, (dim, patch) in enumerate(zip(shape, patch_size)):
        max_start = max(0, dim - patch)
        if center is None:
            starts[axis] = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            continue
        nominal = int(center[axis]) - patch // 2
        low = max(0, nominal - patch // 4)
        high = min(max_start, nominal + patch // 4)
        if high < low:
            low = max(0, min(max_start, nominal))
            high = low
        starts[axis] = int(rng.integers(low, high + 1)) if high > low else low
    return starts


def _extract_patch(volume: np.ndarray, starts: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    slices = []
    pads = []
    for start, patch, dim in zip(starts.tolist(), patch_size, volume.shape):
        stop = min(dim, start + patch)
        slices.append(slice(start, stop))
        pads.append((0, patch - (stop - start)))
    patch = volume[tuple(slices)]
    if any(pad_after > 0 for _, pad_after in pads):
        patch = np.pad(patch, pads, mode="edge")
    return patch


class TeacherPatchDataset(Dataset):
    def __init__(
        self,
        cache_paths: Sequence[Path],
        patch_size: Tuple[int, int, int],
        samples_per_epoch: int,
        foreground_fraction: float,
        seed: int,
    ):
        self.cache_paths = [Path(path) for path in cache_paths]
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_fraction = float(foreground_fraction)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        case_path = self.cache_paths[int(rng.integers(0, len(self.cache_paths)))]
        case = load_preprocessed_training_case(case_path)
        label = case["label"]
        bbox_mins = case["bbox_mins"]
        bbox_maxs = case["bbox_maxs"]
        center = None

        use_foreground = rng.random() < self.foreground_fraction and np.any(label > 0)
        if use_foreground:
            coords = np.argwhere(label > 0)
            center = coords[int(rng.integers(0, len(coords)))].astype(np.int32)
        elif np.any(label > 0) and rng.random() < 0.5:
            center = ((bbox_mins + bbox_maxs) // 2).astype(np.int32)

        starts = _choose_patch_start(rng, tuple(int(v) for v in label.shape), self.patch_size, center)
        image_patch = _extract_patch(case["image"], starts, self.patch_size)
        label_patch = _extract_patch(label, starts, self.patch_size)
        return {
            "image": torch.from_numpy(image_patch[None, ...].astype(np.float32)),
            "label": torch.from_numpy(label_patch[None, ...].astype(np.float32)),
            "case_name": case_path.stem,
        }


class TeacherVolumeDataset(Dataset):
    def __init__(self, cache_paths: Sequence[Path]):
        self.cache_paths = [Path(path) for path in cache_paths]

    def __len__(self) -> int:
        return len(self.cache_paths)

    def __getitem__(self, index: int):
        case_path = self.cache_paths[index]
        case = load_preprocessed_training_case(case_path)
        return {
            "image": torch.from_numpy(case["image"][None, ...].astype(np.float32)),
            "label": torch.from_numpy(case["label"][None, ...].astype(np.float32)),
            "case_name": case_path.stem,
            "affine": torch.from_numpy(case["affine"].astype(np.float32)),
        }
