import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, label as connected_components

from teacher_common import list_preprocessed_cases, load_preprocessed_case, print_progress, resolve_path
from teacher_metrics import binary_stats_from_probs
from teacher_models import build_teacher_model


def parse_patch_size(text: str) -> Tuple[int, int, int]:
    parts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("patch size must be z,y,x")
    return int(parts[0]), int(parts[1]), int(parts[2])


def sliding_window_predict(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    _, _, depth, height, width = volume.shape
    patch_z, patch_y, patch_x = patch_size
    stride_z = max(1, patch_z // 2)
    stride_y = max(1, patch_y // 2)
    stride_x = max(1, patch_x // 2)

    z_starts = list(range(0, max(1, depth - patch_z + 1), stride_z))
    y_starts = list(range(0, max(1, height - patch_y + 1), stride_y))
    x_starts = list(range(0, max(1, width - patch_x + 1), stride_x))
    if z_starts[-1] != max(0, depth - patch_z):
        z_starts.append(max(0, depth - patch_z))
    if y_starts[-1] != max(0, height - patch_y):
        y_starts.append(max(0, height - patch_y))
    if x_starts[-1] != max(0, width - patch_x):
        x_starts.append(max(0, width - patch_x))

    prob_sum = torch.zeros((1, 1, depth, height, width), dtype=torch.float32, device=device)
    prob_count = torch.zeros_like(prob_sum)

    model.eval()
    with torch.no_grad():
        for z0 in z_starts:
            z1 = min(depth, z0 + patch_z)
            for y0 in y_starts:
                y1 = min(height, y0 + patch_y)
                for x0 in x_starts:
                    x1 = min(width, x0 + patch_x)
                    patch = volume[:, :, z0:z1, y0:y1, x0:x1]
                    pad_z = patch_z - patch.shape[2]
                    pad_y = patch_y - patch.shape[3]
                    pad_x = patch_x - patch.shape[4]
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        patch = torch.nn.functional.pad(patch, [0, pad_x, 0, pad_y, 0, pad_z], mode="replicate")
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(patch)
                    probs = torch.sigmoid(logits[:, :, : z1 - z0, : y1 - y0, : x1 - x0])
                    prob_sum[:, :, z0:z1, y0:y1, x0:x1] += probs
                    prob_count[:, :, z0:z1, y0:y1, x0:x1] += 1.0
    return prob_sum / torch.clamp(prob_count, min=1.0)


def predict_with_tta(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    use_tta: bool,
) -> torch.Tensor:
    if not use_tta:
        return sliding_window_predict(model, volume, patch_size, device)

    flip_axes = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4)]
    probs_sum = torch.zeros_like(volume, dtype=torch.float32, device=device)
    for axes in flip_axes:
        if axes:
            flipped = torch.flip(volume, dims=axes)
        else:
            flipped = volume
        probs = sliding_window_predict(model, flipped, patch_size, device)
        if axes:
            probs = torch.flip(probs, dims=axes)
        probs_sum += probs
    return probs_sum / float(len(flip_axes))


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, num_components = connected_components(mask.astype(np.uint8))
    if num_components <= 1:
        return mask.astype(np.uint8)
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))
    return (labeled == largest_label).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, keep_largest: bool, fill_holes: bool) -> np.ndarray:
    processed = mask.astype(np.uint8)
    if keep_largest and np.any(processed > 0):
        processed = keep_largest_connected_component(processed)
    if fill_holes and np.any(processed > 0):
        processed = binary_fill_holes(processed.astype(bool)).astype(np.uint8)
    return processed.astype(np.uint8)


def save_case_outputs(
    output_dir: Path,
    case_name: str,
    prob_volume: np.ndarray,
    pred_mask: np.ndarray,
    label_volume: np.ndarray,
    affine: np.ndarray,
    save_prob_nifti: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_npz_path = output_dir / f"{case_name}_prob.npz"
    mask_nii_path = output_dir / f"{case_name}_mask.nii.gz"
    prob_nii_path = output_dir / f"{case_name}_prob.nii.gz"
    label_nii_path = output_dir / f"{case_name}_label.nii.gz"

    np.savez_compressed(str(prob_npz_path), probability=prob_volume.astype(np.float32))
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine.astype(np.float32)), str(mask_nii_path))
    nib.save(nib.Nifti1Image(label_volume.astype(np.uint8), affine.astype(np.float32)), str(label_nii_path))
    if save_prob_nifti:
        nib.save(nib.Nifti1Image(prob_volume.astype(np.float32), affine.astype(np.float32)), str(prob_nii_path))

    result = {
        "prob_npz": str(prob_npz_path),
        "mask_nii": str(mask_nii_path),
        "label_nii": str(label_nii_path),
    }
    if save_prob_nifti:
        result["prob_nii"] = str(prob_nii_path)
    return result


def case_metrics_from_volume(prob_volume: np.ndarray, label_volume: np.ndarray, threshold: float, keep_largest: bool, fill_holes: bool) -> Tuple[Dict[str, float], np.ndarray]:
    pred_mask = (prob_volume >= threshold).astype(np.uint8)
    pred_mask = postprocess_mask(pred_mask, keep_largest=keep_largest, fill_holes=fill_holes)
    probs_tensor = torch.from_numpy(pred_mask[None, None, ...].astype(np.float32))
    label_tensor = torch.from_numpy(label_volume[None, None, ...].astype(np.float32))
    stats = binary_stats_from_probs(probs_tensor, label_tensor, threshold=0.5)
    return stats, pred_mask


def choose_best_threshold(
    cached_cases: List[Dict[str, object]],
    thresholds: List[float],
    keep_largest: bool,
    fill_holes: bool,
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    best_threshold = float(thresholds[0])
    best_metrics = {"dice": -1.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "specificity": 0.0}
    search_history: List[Dict[str, float]] = []

    for threshold in thresholds:
        totals = {"dice": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "specificity": 0.0}
        for case in cached_cases:
            stats, _ = case_metrics_from_volume(
                case["prob_volume"],
                case["label_volume"],
                threshold=float(threshold),
                keep_largest=keep_largest,
                fill_holes=fill_holes,
            )
            for key, value in stats.items():
                totals[key] += float(value)
        count = float(len(cached_cases))
        metrics = {key: totals[key] / count for key in totals}
        metrics["threshold"] = float(threshold)
        search_history.append(metrics)
        if metrics["dice"] > best_metrics["dice"]:
            best_threshold = float(threshold)
            best_metrics = {key: float(value) for key, value in metrics.items() if key != "threshold"}
    return best_threshold, best_metrics, search_history


def run_split(
    split_name: str,
    case_paths: List[Path],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    threshold: float,
    device: torch.device,
    output_root: Path,
    save_prob_nifti: bool,
    use_tta: bool,
    keep_largest: bool,
    fill_holes: bool,
    threshold_search_values: List[float] | None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    results: List[Dict[str, object]] = []
    total_cases = len(case_paths)
    split_output_dir = output_root / split_name
    cached_cases: List[Dict[str, object]] = []

    for index, case_path in enumerate(case_paths, start=1):
        case_name = case_path.stem
        case = load_preprocessed_case(case_path)
        image = torch.from_numpy(case["image"][None, None, ...].astype(np.float32)).to(device)
        label = case["label"].astype(np.uint8)

        probs = predict_with_tta(model, image, patch_size, device, use_tta=use_tta)
        prob_volume = probs.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        cached_cases.append({"case_name": case_name, "prob_volume": prob_volume, "label_volume": label, "affine": case["affine"]})
        print_progress(f"infer_{split_name}", index, total_cases, extra=f"case={case_name}")

    applied_threshold = float(threshold)
    threshold_search: Dict[str, object] = {"enabled": False}
    if threshold_search_values:
        applied_threshold, best_metrics, search_history = choose_best_threshold(
            cached_cases,
            thresholds=threshold_search_values,
            keep_largest=keep_largest,
            fill_holes=fill_holes,
        )
        threshold_search = {
            "enabled": True,
            "candidate_thresholds": [float(v) for v in threshold_search_values],
            "best_threshold": float(applied_threshold),
            "best_metrics": best_metrics,
            "history": search_history,
        }

    for index, case in enumerate(cached_cases, start=1):
        stats, pred_mask = case_metrics_from_volume(
            case["prob_volume"],
            case["label_volume"],
            threshold=applied_threshold,
            keep_largest=keep_largest,
            fill_holes=fill_holes,
        )
        output_paths = save_case_outputs(
            split_output_dir,
            str(case["case_name"]),
            case["prob_volume"],
            pred_mask,
            case["label_volume"],
            case["affine"],
            save_prob_nifti=save_prob_nifti,
        )

        case_result: Dict[str, object] = {
            "case_name": str(case["case_name"]),
            "metrics": stats,
            "shape": [int(v) for v in case["prob_volume"].shape],
            "threshold": float(applied_threshold),
        }
        case_result.update(output_paths)
        results.append(case_result)
        print_progress(f"post_{split_name}", index, total_cases, extra=f"case={case['case_name']} dice={stats['dice']:.5f}")
    return results, {"threshold": float(applied_threshold), "threshold_search": threshold_search}


def aggregate_metrics(case_results: List[Dict[str, object]]) -> Dict[str, float]:
    if not case_results:
        return {"dice": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "specificity": 0.0}
    keys = ["dice", "precision", "recall", "iou", "specificity"]
    totals = {key: 0.0 for key in keys}
    for result in case_results:
        metrics = result["metrics"]
        for key in keys:
            totals[key] += float(metrics[key])
    count = float(len(case_results))
    return {key: totals[key] / count for key in keys}


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_root = base_dir / "outputs2"

    parser = argparse.ArgumentParser(description="Run inference with the trained teacher segmentation model.")
    parser.add_argument("--preprocessed-dir", type=Path, default=outputs_root / "teacher_preprocessed" / "organ")
    parser.add_argument("--checkpoint", type=Path, default=outputs_root / "teacher_models" / "organ" / "teacher_best.pt")
    parser.add_argument("--output-dir", type=Path, default=outputs_root / "teacher_predictions" / "organ")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--patch-size", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--save-prob-nifti", action="store_true")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True, help="Enable flip test-time augmentation.")
    parser.add_argument("--keep-largest", action=argparse.BooleanOptionalAction, default=True, help="Keep only the largest connected component in predicted masks.")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True, help="Fill binary holes in predicted masks.")
    parser.add_argument("--threshold-search", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55", help="Comma-separated validation thresholds, e.g. 0.4,0.45,0.5,0.55")
    args = parser.parse_args()

    preprocessed_dir = resolve_path(base_dir, str(args.preprocessed_dir))
    checkpoint_path = resolve_path(base_dir, str(args.checkpoint))
    output_dir = resolve_path(base_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    patch_size = parse_patch_size(args.patch_size) if args.patch_size else tuple(int(v) for v in checkpoint["patch_size"])
    model_name = str(checkpoint.get("model_name", "unet3d"))
    base_channels = int(checkpoint.get("base_channels", 24))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_teacher_model(model_name, in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    requested_splits = [args.split] if args.split in {"train", "val"} else ["train", "val"]
    summary: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "preprocessed_dir": str(preprocessed_dir),
        "output_dir": str(output_dir),
        "threshold": float(args.threshold),
        "patch_size": [int(v) for v in patch_size],
        "tta": bool(args.tta),
        "keep_largest": bool(args.keep_largest),
        "fill_holes": bool(args.fill_holes),
        "splits": {},
    }
    threshold_search_values = [float(item.strip()) for item in args.threshold_search.split(",") if item.strip()]

    for split_name in requested_splits:
        case_paths = list_preprocessed_cases(preprocessed_dir / split_name)
        if not case_paths:
            print(f"No preprocessed cases found for split={split_name}, skipping.", flush=True)
            continue
        print(f"Teacher inference split={split_name} cases={len(case_paths)}", flush=True)
        case_results, split_options = run_split(
            split_name,
            case_paths,
            model,
            patch_size,
            threshold=float(args.threshold),
            device=device,
            output_root=output_dir,
            save_prob_nifti=bool(args.save_prob_nifti),
            use_tta=bool(args.tta),
            keep_largest=bool(args.keep_largest),
            fill_holes=bool(args.fill_holes),
            threshold_search_values=threshold_search_values if split_name == "val" and threshold_search_values else None,
        )
        split_metrics = aggregate_metrics(case_results)
        summary["splits"][split_name] = {
            "options": split_options,
            "metrics": split_metrics,
            "cases": case_results,
        }
        print(
            f"split={split_name} threshold={split_options['threshold']:.3f} dice={split_metrics['dice']:.5f} | iou={split_metrics['iou']:.5f} | "
            f"precision={split_metrics['precision']:.5f} | recall={split_metrics['recall']:.5f} | "
            f"specificity={split_metrics['specificity']:.5f}",
            flush=True,
        )

    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Teacher inference complete. summary={output_dir / 'inference_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
