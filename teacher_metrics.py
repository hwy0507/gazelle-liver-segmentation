from typing import Dict

import torch


def dice_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()
    intersection = torch.sum(preds * targets)
    denominator = torch.sum(preds) + torch.sum(targets)
    return float(((2.0 * intersection + eps) / (denominator + eps)).item())


def binary_stats_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Dict[str, float]:
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds * (1.0 - targets))
    fn = torch.sum((1.0 - preds) * targets)
    tn = torch.sum((1.0 - preds) * (1.0 - targets))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": dice_from_probs(probs, targets, threshold=threshold, eps=eps),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "iou": float(iou.item()),
        "specificity": float(specificity.item()),
    }


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets, dim=(1, 2, 3, 4))
    denominator = torch.sum(probs, dim=(1, 2, 3, 4)) + torch.sum(targets, dim=(1, 2, 3, 4))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6))
    return (1.0 - dice_weight) * bce + dice_weight * dice_loss.mean()
