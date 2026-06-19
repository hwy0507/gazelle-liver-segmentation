import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LAST_PROGRESS_WIDTH = 0


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, low, high):
        return torch.clamp(torch.round(x), low, high)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0], None, None


def quantize_uint4(x: torch.Tensor) -> torch.Tensor:
    x_min = torch.amin(x, dim=-1, keepdim=True).detach()
    x_shifted = x - x_min
    x_max = torch.amax(x_shifted, dim=-1, keepdim=True).detach()
    scale = torch.clamp(x_max / 15.0, min=1e-8)
    q_x = STEQuantize.apply(x_shifted / scale, 0, 15)
    return q_x.to(dtype=torch.float32)


def quantize_int4_weight(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.amax(torch.abs(w), dim=0, keepdim=True).detach()
    scale = torch.clamp(max_abs / 7.0, min=1e-8)
    q = STEQuantize.apply(w / scale, -8, 7)
    return q, scale


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.quantize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize:
            return x @ self.weight
        q_x = quantize_uint4(x)
        q_w, w_scale = quantize_int4_weight(self.weight)
        return (q_x.float() @ q_w.float()) * w_scale


class QuantizedMLP2Layer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2):
        super().__init__()
        self.fc1 = QuantizedLinear(input_dim, hidden_dim)
        self.act1 = nn.PReLU()
        self.fc2 = QuantizedLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        return self.fc2(x)

    def set_quantize(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, QuantizedLinear):
                module.quantize = enabled


class StudentDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, teacher_probs: np.ndarray, sample_weights: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.teacher_probs = torch.from_numpy(teacher_probs.astype(np.float32))
        self.sample_weights = torch.from_numpy(sample_weights.astype(np.float32))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.teacher_probs[index], self.sample_weights[index]


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def print_progress(prefix: str, completed: int, total: int, extra: str = "") -> None:
    global LAST_PROGRESS_WIDTH
    total_safe = max(1, int(total))
    finished = min(max(0, int(completed)), total_safe)
    progress = float(finished) / float(total_safe)
    suffix = f" {extra}" if extra else ""
    message = f"{prefix} {progress:.3f}/1{suffix}"
    padded = message.ljust(max(LAST_PROGRESS_WIDTH, len(message)))
    if finished >= total_safe:
        print(f"\r{padded}")
        LAST_PROGRESS_WIDTH = 0
    else:
        print(f"\r{padded}", end="", flush=True)
        LAST_PROGRESS_WIDTH = len(padded)


def choose_hidden_dim(input_dim: int) -> int:
    if input_dim <= 384:
        return 128
    if input_dim <= 768:
        return 192
    if input_dim <= 1536:
        return 256
    return 384


def infer_positive_class_weight(labels: torch.Tensor) -> float:
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives <= 0:
        return 1.0
    raw_weight = float(negatives) / float(positives)
    return min(max(raw_weight, 1.0), 8.0)


def compute_binary_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    total = tp + fp + tn + fn
    precision = float(tp) / float(max(1, tp + fp))
    recall = float(tp) / float(max(1, tp + fn))
    accuracy = float(tp + tn) / float(max(1, total))
    f1 = 0.0 if precision + recall <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    dice = 0.0 if (2 * tp + fp + fn) <= 0 else float(2 * tp) / float(2 * tp + fp + fn)
    iou = 0.0 if (tp + fp + fn) <= 0 else float(tp) / float(tp + fp + fn)
    specificity = float(tn) / float(max(1, tn + fp))
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def compute_composite_score(metrics: Dict[str, float]) -> float:
    return (
        0.50 * float(metrics.get("dice", 0.0))
        + 0.20 * float(metrics.get("iou", 0.0))
        + 0.20 * float(metrics.get("recall", 0.0))
        + 0.10 * float(metrics.get("precision", 0.0))
    )


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets_float = targets.float()
    weighted_probs = probs * sample_weights
    weighted_targets = targets_float * sample_weights
    intersection = torch.sum(weighted_probs * targets_float)
    denominator = torch.sum(weighted_probs) + torch.sum(weighted_targets)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weights: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.cross_entropy(logits, labels, weight=class_weight, reduction="none")
    return torch.mean(raw * sample_weights)


def distillation_kl_loss(logits: torch.Tensor, teacher_probs: torch.Tensor, sample_weights: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = torch.log_softmax(logits / temperature, dim=1)
    teacher_two_class = torch.stack([1.0 - teacher_probs, teacher_probs], dim=1)
    teacher_soft = torch.clamp(teacher_two_class, min=1e-6, max=1.0)
    teacher_soft = teacher_soft / torch.sum(teacher_soft, dim=1, keepdim=True)
    kl = torch.sum(teacher_soft * (torch.log(teacher_soft) - student_log_probs), dim=1)
    return torch.mean(kl * sample_weights) * (temperature ** 2)


def load_split(split_dir: Path, split_name: str) -> StudentDataset:
    features = np.load(str(split_dir / f"{split_name}_features.npy")).astype(np.float32)
    labels = np.load(str(split_dir / f"{split_name}_labels.npy")).astype(np.int64)
    teacher_probs = np.load(str(split_dir / f"{split_name}_teacher_probs.npy")).astype(np.float32)
    sample_weights = np.load(str(split_dir / f"{split_name}_sample_weights.npy")).astype(np.float32)
    return StudentDataset(features, labels, teacher_probs, sample_weights)


def split_model_params(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    optical_params: Dict[str, np.ndarray] = {}
    nonlinear_params: Dict[str, np.ndarray] = {}
    for name, param in state_dict.items():
        tensor = param.detach().cpu().numpy()
        if name in {"fc1.weight", "fc2.weight"}:
            optical_params[name] = tensor
        else:
            nonlinear_params[name] = tensor
    return optical_params, nonlinear_params


def export_gazelle_linear_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    exported: Dict[str, np.ndarray] = {}
    for layer_name in ("fc1.weight", "fc2.weight"):
        weight = state_dict[layer_name].detach().cpu()
        q_weight, scale = quantize_int4_weight(weight)
        exported[layer_name] = q_weight.numpy().astype(np.int8)
        exported[f"{layer_name}.scale"] = scale.numpy().astype(np.float32)
        exported[f"{layer_name}.shape"] = np.asarray(weight.shape, dtype=np.int32)
    return exported


def save_outputs(output_dir: Path, state_dict: Dict[str, torch.Tensor], input_dim: int, hidden_dim: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optical_params, nonlinear_params = split_model_params(state_dict)
    gazelle_linear_params = export_gazelle_linear_params(state_dict)
    metadata = {
        "architecture": "fc1 -> PReLU -> fc2",
        "input_dim": np.asarray([input_dim], dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "output_dim": np.asarray([2], dtype=np.int32),
        "input_type": np.asarray(["uint4"]),
        "weight_type": np.asarray(["int4"]),
    }
    with open(output_dir / "optical_params.pkl", "wb") as f:
        pickle.dump(optical_params, f)
    with open(output_dir / "nonlinear_params.pkl", "wb") as f:
        pickle.dump(nonlinear_params, f)
    with open(output_dir / "gazelle_optical_params.pkl", "wb") as f:
        pickle.dump(gazelle_linear_params, f)
    with open(output_dir / "gazelle_model_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)


def save_runtime_best(
    output_dir: Path,
    state_dict: Dict[str, torch.Tensor],
    input_dim: int,
    hidden_dim: int,
    best_metrics: Dict[str, float],
    history: List[Dict[str, float]],
) -> None:
    save_outputs(output_dir, state_dict, input_dim=input_dim, hidden_dim=hidden_dim)
    torch.save(
        {
            "model_state": state_dict,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "best_metrics": best_metrics,
            "history": history,
        },
        str(output_dir / "student_best.pt"),
    )
    summary = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "best_metrics": best_metrics,
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weight: torch.Tensor,
    clip_norm: float,
    device: torch.device,
    dice_loss_weight: float,
    distill_weight: float,
    temperature: float,
) -> Tuple[float, float]:
    net.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_steps = len(loader)
    for step_index, (xb, yb, tb, wb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        tb = tb.to(device)
        wb = wb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(xb)
        ce_loss = weighted_ce_loss(logits, yb, wb, class_weight)
        dice_loss = soft_dice_loss_from_logits(logits, yb, wb)
        distill_loss = distillation_kl_loss(logits, tb, wb, temperature=temperature)
        loss = ce_loss + dice_loss_weight * dice_loss + distill_weight * distill_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == yb).sum().item()
        total_seen += xb.size(0)
        print_progress("train", step_index, total_steps, extra=f"loss={loss.item():.5f}")
    return total_loss / max(1, total_seen), total_correct / max(1, total_seen)


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, class_weight: torch.Tensor, device: torch.device, distill_temperature: float = 2.0) -> Tuple[float, Dict[str, float]]:
    net.eval()
    total_loss = 0.0
    total_seen = 0
    tp = fp = tn = fn = 0
    total_steps = len(loader)
    for step_index, (xb, yb, tb, wb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        tb = tb.to(device)
        wb = wb.to(device)
        logits = net(xb)
        ce_loss = weighted_ce_loss(logits, yb, wb, class_weight)
        dice_loss = soft_dice_loss_from_logits(logits, yb, wb)
        distill_loss = distillation_kl_loss(logits, tb, wb, temperature=distill_temperature)
        loss = ce_loss + 0.4 * dice_loss + 0.6 * distill_loss
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        total_seen += xb.size(0)
        tp += int(((preds == 1) & (yb == 1)).sum().item())
        fp += int(((preds == 1) & (yb == 0)).sum().item())
        tn += int(((preds == 0) & (yb == 0)).sum().item())
        fn += int(((preds == 0) & (yb == 1)).sum().item())
        print_progress("val", step_index, total_steps, extra=f"step={step_index}/{total_steps}")
    metrics = compute_binary_metrics_from_counts(tp, fp, tn, fn)
    metrics.update({"tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})
    return total_loss / max(1, total_seen), metrics


def train_student(
    train_dataset: StudentDataset,
    val_dataset: StudentDataset,
    output_dir: Path,
    hidden_dim: int,
    max_epochs: int,
    base_lr: float,
    batch_size: int,
    device: torch.device,
    min_epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    enable_early_stop_after_full_quant: bool,
    positive_class_weight: float,
    warmup_epochs: int,
    quant_tune_epochs: int,
    weight_decay: float,
    eta_min: float,
    clip_float: float,
    clip_quant_tune: float,
    clip_full_quant: float,
    full_quant_converge_window: int,
    full_quant_converge_delta: float,
    max_full_quant_rounds: int,
    dice_loss_weight: float,
    distill_weight: float,
    temperature: float,
) -> Tuple[float, Dict[str, torch.Tensor], Dict[str, float], List[Dict[str, float]], Dict[str, float], Dict[str, torch.Tensor] | None]:
    input_dim = int(train_dataset.features.shape[1])
    net = QuantizedMLP2Layer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
    class_weight = torch.tensor([1.0, positive_class_weight], dtype=torch.float32, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_full_quant_score = -1.0
    best_full_quant_state: Dict[str, torch.Tensor] | None = None
    best_full_quant_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    total_cycles = max(1, max_epochs)

    print(f"输入维度: {input_dim} | hidden: {hidden_dim} | 输出维度: 2", flush=True)
    print(f"网络结构: {input_dim} -> {hidden_dim} -> 2 (fc1 -> PReLU -> fc2)", flush=True)
    print(
        f"学习率={base_lr:.6f} | 正类权重={positive_class_weight:.4f} | dice_weight={dice_loss_weight:.4f} | "
        f"distill_weight={distill_weight:.4f} | temperature={temperature:.2f}",
        flush=True,
    )
    print(
        f"循环设置: cycles={total_cycles}, float_per_cycle=1, quant_per_cycle=1, full_quant_until_converge=on",
        flush=True,
    )
    print(
        f"训练控制: full_quant后early_stop={'on' if enable_early_stop_after_full_quant else 'off'} | max_full_quant_rounds={max_full_quant_rounds} | converge_window={full_quant_converge_window} | converge_delta={full_quant_converge_delta:.6f}",
        flush=True,
    )

    global_step = 0
    for cycle in range(1, total_cycles + 1):
        print(f"\n=== Cycle {cycle}/{total_cycles}: 浮点 -> 量化微调 -> 全量化收敛 ===", flush=True)

        for phase_name, quantize_enabled, clip_norm in (
            ("浮点热身", False, clip_float),
            ("量化微调", True, clip_quant_tune),
        ):
            started = time.perf_counter()
            net.set_quantize(quantize_enabled)
            train_loss, train_acc = train_one_epoch(
                net,
                train_loader,
                optimizer,
                class_weight,
                clip_norm,
                device,
                dice_loss_weight,
                distill_weight,
                temperature,
            )
            val_loss, val_metrics = evaluate(net, val_loader, class_weight, device, distill_temperature=temperature)
            scheduler.step()
            score = compute_composite_score(val_metrics)
            elapsed = time.perf_counter() - started
            global_step += 1

            record = {
                "cycle": float(cycle),
                "step": float(global_step),
                "phase": phase_name,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "score": float(score),
                "dice": float(val_metrics["dice"]),
                "iou": float(val_metrics["iou"]),
                "precision": float(val_metrics["precision"]),
                "recall": float(val_metrics["recall"]),
                "specificity": float(val_metrics["specificity"]),
                "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "runtime_seconds": float(elapsed),
            }
            history.append(record)
            print(
                f"cycle={cycle}/{total_cycles} phase={phase_name} train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
                f"val_loss={val_loss:.5f} dice={val_metrics['dice']:.5f} iou={val_metrics['iou']:.5f} precision={val_metrics['precision']:.5f} "
                f"recall={val_metrics['recall']:.5f} score={score:.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )

        net.set_quantize(True)
        tracked_metric_names = ["dice", "iou", "precision", "recall", "specificity"]
        full_quant_score_history: List[float] = []
        full_quant_metric_history: Dict[str, List[float]] = {name: [] for name in tracked_metric_names}
        full_quant_round = 0
        cycle_converged = False
        last_round_metrics: Dict[str, float] | None = None
        last_round_state: Dict[str, torch.Tensor] | None = None

        while full_quant_round < max_full_quant_rounds:
            full_quant_round += 1
            started = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                net,
                train_loader,
                optimizer,
                class_weight,
                clip_full_quant,
                device,
                dice_loss_weight,
                distill_weight,
                temperature,
            )
            val_loss, val_metrics = evaluate(net, val_loader, class_weight, device, distill_temperature=temperature)
            scheduler.step()
            score = compute_composite_score(val_metrics)
            elapsed = time.perf_counter() - started
            global_step += 1

            record = {
                "cycle": float(cycle),
                "step": float(global_step),
                "full_quant_round": float(full_quant_round),
                "phase": "全量化训练",
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "score": float(score),
                "dice": float(val_metrics["dice"]),
                "iou": float(val_metrics["iou"]),
                "precision": float(val_metrics["precision"]),
                "recall": float(val_metrics["recall"]),
                "specificity": float(val_metrics["specificity"]),
                "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "runtime_seconds": float(elapsed),
            }
            history.append(record)
            print(
                f"cycle={cycle}/{total_cycles} full_quant_round={full_quant_round} train_loss={train_loss:.5f} train_acc={train_acc:.5f} "
                f"val_loss={val_loss:.5f} dice={val_metrics['dice']:.5f} iou={val_metrics['iou']:.5f} precision={val_metrics['precision']:.5f} "
                f"recall={val_metrics['recall']:.5f} score={score:.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )

            last_round_metrics = {
                "cycle": float(cycle),
                "step": float(global_step),
                "full_quant_round": float(full_quant_round),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                **{key: float(value) for key, value in val_metrics.items()},
                "score": float(score),
            }
            last_round_state = {name: param.detach().cpu().clone() for name, param in net.state_dict().items()}

            full_quant_score_history.append(score)
            if len(full_quant_score_history) > full_quant_converge_window:
                full_quant_score_history.pop(0)
            for metric_name in tracked_metric_names:
                full_quant_metric_history[metric_name].append(float(val_metrics[metric_name]))
                if len(full_quant_metric_history[metric_name]) > full_quant_converge_window:
                    full_quant_metric_history[metric_name].pop(0)

            if len(full_quant_metric_history["dice"]) < full_quant_converge_window:
                continue

            score_span = max(full_quant_score_history) - min(full_quant_score_history)
            metric_spans = {
                metric_name: max(values) - min(values)
                for metric_name, values in full_quant_metric_history.items()
            }
            converged_count = sum(1 for span in metric_spans.values() if span <= full_quant_converge_delta)
            metrics_converged = converged_count >= 3
            if metrics_converged and full_quant_round >= min_epochs:
                print(
                    f"Cycle {cycle} 全量化阶段按多指标收敛提前结束: rounds={full_quant_round}, "
                    f"dice_span={metric_spans['dice']:.6f}, iou_span={metric_spans['iou']:.6f}, "
                    f"precision_span={metric_spans['precision']:.6f}, recall_span={metric_spans['recall']:.6f}, "
                    f"specificity_span={metric_spans['specificity']:.6f}, converged_count={converged_count}/5, best_full_quant_dice={best_full_quant_metrics.get('dice', 0.0):.5f}",
                    flush=True,
                )
                cycle_converged = True
                break

        if not cycle_converged:
            print(
                f"Cycle {cycle} 达到全量化轮数上限: rounds={full_quant_round}, 采用本cycle最终收敛候选结果",
                flush=True,
            )

        cycle_final_metrics = last_round_metrics
        cycle_final_state = last_round_state
        if cycle_final_metrics is None or cycle_final_state is None:
            raise RuntimeError(f"Cycle {cycle} 未产生可用的全量化结果")

        cycle_final_score = float(cycle_final_metrics["score"])
        if cycle_final_score > best_full_quant_score + early_stop_min_delta:
            best_full_quant_score = cycle_final_score
            best_full_quant_state = cycle_final_state
            best_full_quant_metrics = dict(cycle_final_metrics)
            save_runtime_best(
                output_dir=output_dir,
                state_dict=best_full_quant_state,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                best_metrics=best_full_quant_metrics,
                history=history,
            )
            print(
                f"即时保存最优收敛权重: cycle={cycle} full_quant_round={int(best_full_quant_metrics.get('full_quant_round', -1))} dice={best_full_quant_metrics.get('dice', 0.0):.5f} score={best_full_quant_metrics.get('score', 0.0):.5f}",
                flush=True,
            )

    if best_full_quant_state is None:
        raise RuntimeError("训练未产生任何全量化阶段最优权重，无法导出 Gazelle 参数")
    return best_full_quant_score, best_full_quant_state, best_full_quant_metrics, history, best_full_quant_metrics, best_full_quant_state


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs_root = base_dir / "outputs2"
    parser = argparse.ArgumentParser(description="Train full-featured Gazelle-oriented student from teacher-prepared data.")
    parser.add_argument("--data-dir", type=Path, default=outputs_root / "student_ready" / "organ")
    parser.add_argument("--output-dir", type=Path, default=outputs_root / "student_models" / "organ")
    parser.add_argument("--epochs", type=int, default=14, help="Number of outer cycles. Each cycle runs float -> quant_tune -> full_quant_to_convergence")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--min-epochs", type=int, default=6, help="Minimum full-quant rounds before convergence checks can finish a cycle")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--enable-early-stop-after-full-quant", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--quant-tune-epochs", type=int, default=15)
    parser.add_argument("--clip-float", type=float, default=0.8)
    parser.add_argument("--clip-quant", type=float, default=0.6)
    parser.add_argument("--clip-full", type=float, default=0.5)
    parser.add_argument("--full-quant-window", type=int, default=6)
    parser.add_argument("--full-quant-delta", type=float, default=0.015)
    parser.add_argument("--max-full-quant-rounds", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = resolve_path(base_dir, str(args.data_dir))
    output_dir = resolve_path(base_dir, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_split(data_dir, "train")
    val_dataset = load_split(data_dir, "val")
    input_dim = int(train_dataset.features.shape[1])
    hidden_dim = int(args.hidden_dim) if int(args.hidden_dim) > 0 else choose_hidden_dim(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_class_weight = infer_positive_class_weight(train_dataset.labels)
    print(
        f"Student data loaded | train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | feature_dim={input_dim} | hidden_dim={hidden_dim}",
        flush=True,
    )

    best_score, best_state, best_metrics, history, best_full_quant_metrics, best_full_quant_state = train_student(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        hidden_dim=hidden_dim,
        max_epochs=args.epochs,
        base_lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        min_epochs=args.min_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        enable_early_stop_after_full_quant=bool(args.enable_early_stop_after_full_quant),
        positive_class_weight=positive_class_weight,
        warmup_epochs=args.warmup_epochs,
        quant_tune_epochs=args.quant_tune_epochs,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        clip_float=args.clip_float,
        clip_quant_tune=args.clip_quant,
        clip_full_quant=args.clip_full,
        full_quant_converge_window=args.full_quant_window,
        full_quant_converge_delta=args.full_quant_delta,
        max_full_quant_rounds=args.max_full_quant_rounds,
        dice_loss_weight=args.dice_weight,
        distill_weight=args.distill_weight,
        temperature=args.temperature,
    )

    export_state = best_full_quant_state if best_full_quant_state is not None else best_state
    export_metrics = best_full_quant_metrics if best_full_quant_metrics else best_metrics
    save_runtime_best(
        output_dir=output_dir,
        state_dict=export_state,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        best_metrics=export_metrics,
        history=history,
    )

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "best_score": float(export_metrics.get("score", best_score)),
        "best_metrics": export_metrics,
        "best_full_quant_metrics": best_full_quant_metrics,
        "positive_class_weight": positive_class_weight,
        "enable_early_stop_after_full_quant": bool(args.enable_early_stop_after_full_quant),
        "full_quant_window": int(args.full_quant_window),
        "full_quant_delta": float(args.full_quant_delta),
        "max_full_quant_rounds": int(args.max_full_quant_rounds),
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"Student training complete. saved_full_quant_epoch={int(export_metrics.get('epoch', -1))} saved_full_quant_score={float(export_metrics.get('score', best_score)):.5f} saved_full_quant_dice={float(export_metrics.get('dice', 0.0)):.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
