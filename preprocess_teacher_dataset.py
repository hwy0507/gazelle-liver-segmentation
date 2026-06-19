import argparse
import json
from pathlib import Path
from typing import Dict, List

from teacher_common import (
    load_preprocessed_case,
    load_teacher_config,
    preprocess_image_and_label,
    print_progress,
    resolve_path,
    save_preprocessed_case,
)
from teacher_dataset import list_case_pairs, split_case_pairs


def is_valid_preprocessed_case(path: Path) -> bool:
    try:
        case = load_preprocessed_case(path)
    except Exception:
        return False
    return all(key in case for key in ("image", "label", "affine"))


def preprocess_split(
    split_name: str,
    case_pairs: List[tuple[str, Path, Path]],
    output_dir: Path,
    config: Dict[str, object],
) -> List[Dict[str, object]]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []
    total_cases = len(case_pairs)
    for index, (case_name, image_path, label_path) in enumerate(case_pairs, start=1):
        output_path = split_dir / f"{case_name}.npz"
        regenerated = False
        if output_path.exists():
            if is_valid_preprocessed_case(output_path):
                manifest.append({"case_name": case_name, "path": str(output_path), "cached": True})
                print_progress(f"preprocess_{split_name}", index, total_cases, extra=f"cached case={case_name}")
                continue
            output_path.unlink(missing_ok=True)
            regenerated = True
        image, label, affine = preprocess_image_and_label(image_path, label_path, config)
        save_preprocessed_case(output_path, image, label, affine)
        manifest.append(
            {
                "case_name": case_name,
                "path": str(output_path),
                "cached": False,
                "regenerated": regenerated,
                "shape": [int(v) for v in image.shape],
            }
        )
        print_progress(f"preprocess_{split_name}", index, total_cases, extra=f"saved case={case_name}")
    return manifest


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_root = base_dir / "outputs2"

    parser = argparse.ArgumentParser(description="Preprocess raw Task03_Liver data for standalone teacher training.")
    parser.add_argument("--config", type=Path, default=None, help="Optional standalone teacher config override JSON.")
    parser.add_argument("--images-dir", type=Path, default=base_dir / "Task03_Liver" / "imagesTr")
    parser.add_argument("--labels-dir", type=Path, default=base_dir / "Task03_Liver" / "labelsTr")
    parser.add_argument("--output-dir", type=Path, default=outputs_root / "teacher_preprocessed" / "organ")
    parser.add_argument("--val-fraction", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_teacher_config(args.config)
    val_fraction = float(args.val_fraction if args.val_fraction >= 0.0 else config.get("validation_fraction", 0.2))
    file_suffix = str(config.get("file_suffix", ".nii.gz"))

    images_dir = resolve_path(base_dir, str(args.images_dir))
    labels_dir = resolve_path(base_dir, str(args.labels_dir))
    output_dir = resolve_path(base_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    case_pairs = list_case_pairs(images_dir, labels_dir, file_suffix)
    train_pairs, val_pairs = split_case_pairs(case_pairs, val_fraction=val_fraction, seed=args.seed)

    print(f"Teacher preprocess cases={len(case_pairs)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)
    train_manifest = preprocess_split("train", train_pairs, output_dir, config)
    val_manifest = preprocess_split("val", val_pairs, output_dir, config) if val_pairs else []

    summary = {
        "config": str(args.config) if args.config is not None else None,
        "teacher_config": config,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "val_fraction": val_fraction,
        "seed": int(args.seed),
        "train_cases": [item["case_name"] for item in train_manifest],
        "val_cases": [item["case_name"] for item in val_manifest],
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    }
    with (output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Teacher preprocessing complete. output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
