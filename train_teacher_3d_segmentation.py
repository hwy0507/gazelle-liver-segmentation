import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from teacher_common import list_preprocessed_cases, load_teacher_config, print_progress, resolve_path
from teacher_dataset import (
    TeacherPatchDataset,
    TeacherVolumeDataset,
)
from teacher_metrics import binary_stats_from_probs, dice_bce_loss
from teacher_models import build_teacher_model


def parse_patch_size(text: str) -> Tuple[int, int, int]:
    parts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("patch size must be z,y,x")
    return int(parts[0]), int(parts[1]), int(parts[2])


def print_epoch_header(epoch: int, total_epochs: int, stage: str) -> None:
    print(f"[Epoch {epoch:03d}/{total_epochs:03d}] {stage}", flush=True)


def print_epoch_metrics(epoch: int, total_epochs: int, train_loss: float, val_metrics: Dict[str, float], lr: float, elapsed: float) -> None:
    print(
        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
        f"loss={train_loss:.5f} | dice={val_metrics['dice']:.5f} | iou={val_metrics['iou']:.5f} | "
        f"precision={val_metrics['precision']:.5f} | recall={val_metrics['recall']:.5f} | "
        f"specificity={val_metrics['specificity']:.5f} | lr={lr:.6g} | time={elapsed:.1f}s",
        flush=True,
    )


def sliding_window_predict(
    model: nn.Module,
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
                    logits = model(patch)
                    probs = torch.sigmoid(logits[:, :, : z1 - z0, : y1 - y0, : x1 - x0])
                    prob_sum[:, :, z0:z1, y0:y1, x0:x1] += probs
                    prob_count[:, :, z0:z1, y0:y1, x0:x1] += 1.0
    return prob_sum / torch.clamp(prob_count, min=1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dice_weight: float,
    amp_enabled: bool,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_steps = len(loader)
    print_epoch_header(epoch, total_epochs, "train")
    for step_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = dice_bce_loss(logits, labels, dice_weight=dice_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1
        print_progress("  train", step_index, total_steps, extra=f"step={step_index}/{total_steps} loss={loss.item():.5f}")
    return total_loss / max(1, total_batches)


def evaluate(
    model: nn.Module,
    dataset: TeacherVolumeDataset,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    if len(dataset) == 0:
        return {"dice": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "specificity": 0.0}

    totals = {"dice": 0.0, "precision": 0.0, "recall": 0.0, "iou": 0.0, "specificity": 0.0}
    total_cases = len(dataset)
    print_epoch_header(epoch, total_epochs, "val")
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].unsqueeze(0).to(device)
        probs = sliding_window_predict(model, image, patch_size, device)
        stats = binary_stats_from_probs(probs, label)
        for key, value in stats.items():
            totals[key] += float(value)
        print_progress("  val", index + 1, total_cases, extra=f"case={sample['case_name']}")
    case_count = float(len(dataset))
    return {key: value / case_count for key, value in totals.items()}


def save_checkpoint(output_dir: Path, checkpoint_name: str, checkpoint: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_dir / checkpoint_name))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_root = base_dir / "outputs2"

    parser = argparse.ArgumentParser(description="Train stage-1 teacher organ segmentation model.")
    parser.add_argument("--config", type=Path, default=None, help="Optional standalone teacher config override JSON.")
    parser.add_argument("--preprocessed-dir", type=Path, default=outputs_root / "teacher_preprocessed" / "organ")
    parser.add_argument("--output-dir", type=Path, default=outputs_root / "teacher_models" / "organ")
    parser.add_argument("--model-name", type=str, default="unet3d")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=800)
    parser.add_argument("--patch-size", type=str, default="32,96,96")
    parser.add_argument("--foreground-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=-1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_teacher_config(args.config)
    patch_size = parse_patch_size(args.patch_size)

    preprocessed_dir = resolve_path(base_dir, str(args.preprocessed_dir))
    output_dir = resolve_path(base_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache_paths = list_preprocessed_cases(preprocessed_dir / "train")
    val_cache_paths = list_preprocessed_cases(preprocessed_dir / "val")
    if not train_cache_paths:
        raise FileNotFoundError(f"No preprocessed training cases found in {preprocessed_dir / 'train'}")
    print(f"Teacher training cases={len(train_cache_paths)} val_cases={len(val_cache_paths)}", flush=True)

    train_dataset = TeacherPatchDataset(
        train_cache_paths,
        patch_size=patch_size,
        samples_per_epoch=args.samples_per_epoch,
        foreground_fraction=args.foreground_fraction,
        seed=args.seed,
    )
    val_dataset = TeacherVolumeDataset(val_cache_paths)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_teacher_model(args.model_name, in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = device.type == "cuda"

    history: List[Dict[str, float]] = []
    best_dice = -math.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            dice_weight=args.dice_weight,
            amp_enabled=amp_enabled,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()
        val_metrics = evaluate(model, val_dataset, patch_size, device, epoch=epoch, total_epochs=args.epochs) if len(val_dataset) > 0 else {
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "specificity": 0.0,
        }
        elapsed = time.perf_counter() - started
        current_lr = float(optimizer.param_groups[0]["lr"])

        record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_dice": float(val_metrics["dice"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_iou": float(val_metrics["iou"]),
            "val_specificity": float(val_metrics["specificity"]),
            "lr": current_lr,
            "runtime_seconds": float(elapsed),
        }
        history.append(record)
        print_epoch_metrics(epoch, args.epochs, train_loss, val_metrics, current_lr, elapsed)

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config_path": str(args.config) if args.config is not None else None,
            "teacher_config": config,
            "patch_size": list(patch_size),
            "base_channels": int(args.base_channels),
            "metrics": record,
            "preprocessed_dir": str(preprocessed_dir),
            "train_cases": [path.stem for path in train_cache_paths],
            "val_cases": [path.stem for path in val_cache_paths],
        }
        save_checkpoint(output_dir, "teacher_last.pt", checkpoint)
        if val_metrics["dice"] > best_dice:
            best_dice = float(val_metrics["dice"])
            best_epoch = epoch
            save_checkpoint(output_dir, "teacher_best.pt", checkpoint)

    summary = {
        "config": str(args.config) if args.config is not None else None,
        "teacher_config": config,
        "preprocessed_dir": str(preprocessed_dir),
        "output_dir": str(output_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "samples_per_epoch": int(args.samples_per_epoch),
        "patch_size": list(patch_size),
        "model_name": args.model_name,
        "base_channels": int(args.base_channels),
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_dice if best_dice > -math.inf else 0.0),
        "train_cases": [path.stem for path in train_cache_paths],
        "val_cases": [path.stem for path in val_cache_paths],
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Teacher training complete. best_epoch={best_epoch} best_val_dice={summary['best_val_dice']:.5f}", flush=True)


if __name__ == "__main__":
    main()
