import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from student_recon_common import build_features_for_centers, sample_grid_in_bbox, teacher_roi_bbox
from teacher_common import default_teacher_config, list_preprocessed_cases, load_preprocessed_case, print_progress, resolve_path


def boundary_mask_from_label(label_volume: np.ndarray, radius_z: int, radius_xy: int) -> np.ndarray:
    foreground = label_volume > 0
    if not np.any(foreground):
        return np.zeros_like(foreground, dtype=bool)
    expanded = foreground.copy()
    eroded = foreground.copy()
    for dz in range(-radius_z, radius_z + 1):
        for dy in range(-radius_xy, radius_xy + 1):
            for dx in range(-radius_xy, radius_xy + 1):
                shifted = np.roll(foreground, shift=(dz, dy, dx), axis=(0, 1, 2))
                if dz > 0:
                    shifted[:dz, :, :] = False
                elif dz < 0:
                    shifted[dz:, :, :] = False
                if dy > 0:
                    shifted[:, :dy, :] = False
                elif dy < 0:
                    shifted[:, dy:, :] = False
                if dx > 0:
                    shifted[:, :, :dx] = False
                elif dx < 0:
                    shifted[:, :, dx:] = False
                expanded |= shifted
                eroded &= shifted
    return (expanded & (~foreground)) | (foreground & (~eroded))


def select_training_samples(
    centers: np.ndarray,
    labels: np.ndarray,
    teacher_probs: np.ndarray,
    boundary_flags: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    total = centers.shape[0]
    if total <= max_samples:
        return np.arange(total, dtype=np.int32)
    rng = np.random.default_rng(seed)

    hard_flags = np.abs(teacher_probs - labels.astype(np.float32)) >= 0.25
    uncertain_flags = np.abs(teacher_probs - 0.5) <= 0.2
    priority_flags = boundary_flags | hard_flags | uncertain_flags
    priority_indices = np.flatnonzero(priority_flags)
    regular_indices = np.flatnonzero(~priority_flags)

    take_priority = min(len(priority_indices), max_samples * 3 // 4)
    chosen_priority = rng.choice(priority_indices, size=take_priority, replace=False) if take_priority > 0 else np.empty((0,), dtype=np.int32)
    remaining = max_samples - chosen_priority.size
    chosen_regular = rng.choice(regular_indices, size=min(remaining, len(regular_indices)), replace=False) if remaining > 0 and len(regular_indices) > 0 else np.empty((0,), dtype=np.int32)
    chosen = np.concatenate([chosen_priority, chosen_regular])

    if chosen.size < max_samples:
        leftover = np.setdiff1d(np.arange(total, dtype=np.int32), chosen, assume_unique=False)
        extra = rng.choice(leftover, size=min(max_samples - chosen.size, len(leftover)), replace=False) if len(leftover) > 0 else np.empty((0,), dtype=np.int32)
        chosen = np.concatenate([chosen, extra])
    rng.shuffle(chosen)
    return chosen.astype(np.int32)


def load_teacher_probability(prob_npz_path: Path) -> np.ndarray:
    with np.load(str(prob_npz_path), allow_pickle=False) as data:
        return np.asarray(data["probability"], dtype=np.float32)


def build_case_student_data(
    case_path: Path,
    teacher_prob_path: Path,
    settings: Dict[str, int],
    roi_threshold: float,
    roi_margin_z: int,
    roi_margin_xy: int,
    max_samples: int,
    seed: int,
) -> Dict[str, np.ndarray | Dict[str, int] | str]:
    case_name = case_path.stem
    case = load_preprocessed_case(case_path)
    teacher_prob = load_teacher_probability(teacher_prob_path)
    label_volume = case["label"].astype(np.uint8)
    roi_bbox = teacher_roi_bbox(teacher_prob, threshold=roi_threshold, margin_z=roi_margin_z, margin_xy=roi_margin_xy)
    if roi_bbox is None:
        roi_bbox = ((0, label_volume.shape[0]), (0, label_volume.shape[1]), (0, label_volume.shape[2]))

    centers = sample_grid_in_bbox(tuple(int(v) for v in label_volume.shape), int(settings["stride_xy"]), int(settings["stride_z"]), roi_bbox)
    if centers.size == 0:
        centers = np.asarray([[label_volume.shape[0] // 2, label_volume.shape[1] // 2, label_volume.shape[2] // 2]], dtype=np.int32)

    labels = (label_volume[centers[:, 0], centers[:, 1], centers[:, 2]] > 0).astype(np.int64)
    teacher_probs = teacher_prob[centers[:, 0], centers[:, 1], centers[:, 2]].astype(np.float32)
    boundary_mask = boundary_mask_from_label(label_volume, int(settings["boundary_width_z"]), int(settings["boundary_width_xy"]))
    boundary_flags = boundary_mask[centers[:, 0], centers[:, 1], centers[:, 2]]
    chosen = select_training_samples(centers, labels, teacher_probs, boundary_flags, max_samples=max_samples, seed=seed)

    selected_centers = centers[chosen]
    selected_labels = labels[chosen].astype(np.int64)
    selected_teacher_probs = teacher_probs[chosen].astype(np.float32)
    selected_boundary_flags = boundary_flags[chosen].astype(np.uint8)
    features = build_features_for_centers(case["image"], selected_centers, settings, roi_bbox)

    sample_weights = np.ones(selected_labels.shape[0], dtype=np.float32)
    sample_weights[selected_boundary_flags > 0] *= 1.5
    sample_weights[np.abs(selected_teacher_probs - selected_labels.astype(np.float32)) >= 0.25] *= 1.5
    sample_weights[np.abs(selected_teacher_probs - 0.5) <= 0.2] *= 1.25

    return {
        "case_name": case_name,
        "features": features.astype(np.float32),
        "labels": selected_labels,
        "teacher_probs": selected_teacher_probs,
        "boundary_flags": selected_boundary_flags,
        "sample_weights": sample_weights.astype(np.float32),
        "centers": selected_centers.astype(np.int32),
        "stats": {
            "samples": int(selected_labels.shape[0]),
            "positives": int(np.sum(selected_labels == 1)),
            "negatives": int(np.sum(selected_labels == 0)),
        },
    }


def total_samples_from_case_results(case_results: List[Dict[str, np.ndarray | Dict[str, int] | str]]) -> int:
    return int(sum(int(case["stats"]["samples"]) for case in case_results))


def save_split_arrays(output_dir: Path, split_name: str, case_results: List[Dict[str, np.ndarray | Dict[str, int] | str]]) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not case_results:
        raise ValueError(f"No case results to save for split={split_name}")

    total_samples = total_samples_from_case_results(case_results)
    feature_dim = int(case_results[0]["features"].shape[1])

    features_path = output_dir / f"{split_name}_features.npy"
    labels_path = output_dir / f"{split_name}_labels.npy"
    teacher_probs_path = output_dir / f"{split_name}_teacher_probs.npy"
    boundary_flags_path = output_dir / f"{split_name}_boundary_flags.npy"
    sample_weights_path = output_dir / f"{split_name}_sample_weights.npy"

    features = np.lib.format.open_memmap(str(features_path), mode="w+", dtype=np.float32, shape=(total_samples, feature_dim))
    labels = np.lib.format.open_memmap(str(labels_path), mode="w+", dtype=np.int64, shape=(total_samples,))
    teacher_probs = np.lib.format.open_memmap(str(teacher_probs_path), mode="w+", dtype=np.float32, shape=(total_samples,))
    boundary_flags = np.lib.format.open_memmap(str(boundary_flags_path), mode="w+", dtype=np.uint8, shape=(total_samples,))
    sample_weights = np.lib.format.open_memmap(str(sample_weights_path), mode="w+", dtype=np.float32, shape=(total_samples,))

    offset = 0
    total_positive = 0
    total_negative = 0
    teacher_prob_sum = 0.0
    boundary_sum = 0.0
    case_stats = []
    for case in case_results:
        case_features = np.asarray(case["features"], dtype=np.float32)
        case_labels = np.asarray(case["labels"], dtype=np.int64)
        case_teacher_probs = np.asarray(case["teacher_probs"], dtype=np.float32)
        case_boundary_flags = np.asarray(case["boundary_flags"], dtype=np.uint8)
        case_sample_weights = np.asarray(case["sample_weights"], dtype=np.float32)
        count = int(case_features.shape[0])
        end = offset + count

        features[offset:end] = case_features
        labels[offset:end] = case_labels
        teacher_probs[offset:end] = case_teacher_probs
        boundary_flags[offset:end] = case_boundary_flags
        sample_weights[offset:end] = case_sample_weights

        total_positive += int(case["stats"]["positives"])
        total_negative += int(case["stats"]["negatives"])
        teacher_prob_sum += float(np.sum(case_teacher_probs, dtype=np.float64))
        boundary_sum += float(np.sum(case_boundary_flags.astype(np.float64), dtype=np.float64))
        case_stats.append(
            {
                "case_name": str(case["case_name"]),
                "samples": int(case["stats"]["samples"]),
                "positives": int(case["stats"]["positives"]),
                "negatives": int(case["stats"]["negatives"]),
            }
        )
        offset = end

    features.flush()
    labels.flush()
    teacher_probs.flush()
    boundary_flags.flush()
    sample_weights.flush()

    split_summary = {
        "samples": int(total_samples),
        "positives": int(total_positive),
        "negatives": int(total_negative),
        "feature_dim": int(feature_dim),
        "teacher_prob_mean": float(teacher_prob_sum / max(1, total_samples)),
        "boundary_fraction": float(boundary_sum / max(1, total_samples)),
        "case_stats": case_stats,
    }
    with (output_dir / f"{split_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2, ensure_ascii=False)
    return split_summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_root = base_dir / "outputs2"

    parser = argparse.ArgumentParser(description="Prepare Gazelle/student training data from teacher predictions.")
    parser.add_argument("--preprocessed-dir", type=Path, default=outputs_root / "teacher_preprocessed" / "organ")
    parser.add_argument("--teacher-pred-dir", type=Path, default=outputs_root / "teacher_predictions" / "organ")
    parser.add_argument("--output-dir", type=Path, default=outputs_root / "student_ready" / "organ")
    parser.add_argument("--roi-threshold", type=float, default=0.25)
    parser.add_argument("--roi-margin-z", type=int, default=10)
    parser.add_argument("--roi-margin-xy", type=int, default=36)
    parser.add_argument("--max-samples-per-case", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocessed_dir = resolve_path(base_dir, str(args.preprocessed_dir))
    teacher_pred_dir = resolve_path(base_dir, str(args.teacher_pred_dir))
    output_dir = resolve_path(base_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "fine_patch_xy": 12,
        "fine_patch_z": 3,
        "local_patch_xy": 28,
        "local_patch_z": 7,
        "context_patch_xy": 40,
        "context_patch_z": 9,
        "stride_xy": 8,
        "stride_z": 2,
        "boundary_width_xy": 12,
        "boundary_width_z": 5,
    }

    overall_summary: Dict[str, object] = {
        "preprocessed_dir": str(preprocessed_dir),
        "teacher_pred_dir": str(teacher_pred_dir),
        "output_dir": str(output_dir),
        "settings": settings,
        "roi_threshold": float(args.roi_threshold),
        "splits": {},
    }

    available_prediction_splits = []
    for split_name in ("train", "val"):
        if (teacher_pred_dir / split_name).exists():
            available_prediction_splits.append(split_name)
    if not available_prediction_splits:
        raise FileNotFoundError(
            "No teacher prediction splits were found. Run infer_teacher_3d_segmentation.py first, ideally with --split both."
        )

    for split_name in ("train", "val"):
        split_case_paths = list_preprocessed_cases(preprocessed_dir / split_name)
        if not split_case_paths:
            continue
        if not (teacher_pred_dir / split_name).exists():
            raise FileNotFoundError(
                f"Missing teacher prediction split '{split_name}' in {teacher_pred_dir / split_name}. "
                f"Run infer_teacher_3d_segmentation.py --split {split_name} or --split both first."
            )
        split_results: List[Dict[str, np.ndarray | Dict[str, int] | str]] = []
        total_cases = len(split_case_paths)
        for index, case_path in enumerate(split_case_paths, start=1):
            case_name = case_path.stem
            prob_path = teacher_pred_dir / split_name / f"{case_name}_prob.npz"
            if not prob_path.exists():
                raise FileNotFoundError(f"Missing teacher probability file for {case_name}: {prob_path}")
            case_result = build_case_student_data(
                case_path,
                prob_path,
                settings=settings,
                roi_threshold=float(args.roi_threshold),
                roi_margin_z=int(args.roi_margin_z),
                roi_margin_xy=int(args.roi_margin_xy),
                max_samples=int(args.max_samples_per_case),
                seed=args.seed + index,
            )
            split_results.append(case_result)
            print_progress(
                f"student_{split_name}",
                index,
                total_cases,
                extra=f"case={case_name} samples={case_result['stats']['samples']} pos={case_result['stats']['positives']}",
            )
        split_summary = save_split_arrays(output_dir, split_name, split_results)
        overall_summary["splits"][split_name] = split_summary

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    print(f"Student preparation complete. output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
