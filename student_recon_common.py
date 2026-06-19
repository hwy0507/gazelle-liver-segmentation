from typing import Dict, Tuple

import numpy as np


def compute_patch_offsets(size: int) -> np.ndarray:
    half = size // 2
    return np.asarray(list(range(-half, size - half)), dtype=np.int32)


def extract_patches_batch(volume: np.ndarray, centers: np.ndarray, patch_z: int, patch_xy: int) -> np.ndarray:
    pad_xy = patch_xy // 2
    pad_z = patch_z // 2
    padded = np.pad(volume, ((pad_z, pad_z), (pad_xy, pad_xy), (pad_xy, pad_xy)), mode="edge")
    padded_centers = centers.astype(np.int32, copy=False) + np.asarray([pad_z, pad_xy, pad_xy], dtype=np.int32)
    z_offsets = compute_patch_offsets(patch_z)
    y_offsets = compute_patch_offsets(patch_xy)
    x_offsets = compute_patch_offsets(patch_xy)
    z_idx = padded_centers[:, 0][:, None, None, None] + z_offsets[None, :, None, None]
    y_idx = padded_centers[:, 1][:, None, None, None] + y_offsets[None, None, :, None]
    x_idx = padded_centers[:, 2][:, None, None, None] + x_offsets[None, None, None, :]
    return padded[z_idx, y_idx, x_idx].astype(np.float32)


def build_position_features(
    centers: np.ndarray,
    volume_shape: Tuple[int, int, int],
    roi_bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] | None,
) -> np.ndarray:
    shape = np.asarray(volume_shape, dtype=np.float32)
    coords = centers.astype(np.float32)
    global_pos = coords / np.maximum(shape.reshape(1, 3) - 1.0, 1.0)
    if roi_bbox is None:
        return global_pos.astype(np.float32)
    roi_mins = np.asarray([roi_bbox[0][0], roi_bbox[1][0], roi_bbox[2][0]], dtype=np.float32)
    roi_maxs = np.asarray([roi_bbox[0][1], roi_bbox[1][1], roi_bbox[2][1]], dtype=np.float32)
    roi_sizes = np.maximum(roi_maxs - roi_mins - 1.0, 1.0)
    roi_pos = (coords - roi_mins.reshape(1, 3)) / roi_sizes.reshape(1, 3)
    roi_center = (roi_mins + roi_maxs - 1.0) * 0.5
    roi_delta = (coords - roi_center.reshape(1, 3)) / np.maximum(roi_sizes.reshape(1, 3), 1.0)
    return np.concatenate([global_pos, roi_pos, roi_delta], axis=1).astype(np.float32)


def build_features_for_centers(
    volume: np.ndarray,
    centers: np.ndarray,
    settings: Dict[str, int],
    roi_bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] | None,
) -> np.ndarray:
    fine_patch = extract_patches_batch(volume, centers, int(settings["fine_patch_z"]), int(settings["fine_patch_xy"]))
    local_patch = extract_patches_batch(volume, centers, int(settings["local_patch_z"]), int(settings["local_patch_xy"]))
    context_patch = extract_patches_batch(volume, centers, int(settings["context_patch_z"]), int(settings["context_patch_xy"]))
    context_patch = context_patch[:, ::2, ::2, ::2]

    fine_flat = fine_patch.reshape(fine_patch.shape[0], -1)
    local_flat = local_patch.reshape(local_patch.shape[0], -1)
    context_flat = context_patch.reshape(context_patch.shape[0], -1)

    fine_mean = fine_flat.mean(axis=1, keepdims=True, dtype=np.float32)
    fine_std = fine_flat.std(axis=1, keepdims=True, dtype=np.float32)
    local_mean = local_flat.mean(axis=1, keepdims=True, dtype=np.float32)
    local_std = local_flat.std(axis=1, keepdims=True, dtype=np.float32)
    context_mean = context_flat.mean(axis=1, keepdims=True, dtype=np.float32)
    context_std = context_flat.std(axis=1, keepdims=True, dtype=np.float32)
    center_fine = fine_patch[:, fine_patch.shape[1] // 2, fine_patch.shape[2] // 2, fine_patch.shape[3] // 2].reshape(-1, 1)
    center_local = local_patch[:, local_patch.shape[1] // 2, local_patch.shape[2] // 2, local_patch.shape[3] // 2].reshape(-1, 1)
    fine_local_delta = center_fine - local_mean
    local_context_delta = local_mean - context_mean
    stats = np.concatenate(
        [
            fine_mean,
            fine_std,
            local_mean,
            local_std,
            context_mean,
            context_std,
            center_fine,
            center_local,
            fine_local_delta,
            local_context_delta,
            np.abs(fine_local_delta),
            np.abs(local_context_delta),
        ],
        axis=1,
    ).astype(np.float32)
    position = build_position_features(centers, tuple(int(v) for v in volume.shape), roi_bbox)
    return np.concatenate([fine_flat, local_flat, context_flat, stats, position], axis=1).astype(np.float32)


def teacher_roi_bbox(
    prob_volume: np.ndarray,
    threshold: float,
    margin_z: int,
    margin_xy: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] | None:
    coords = np.argwhere(prob_volume >= threshold)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    depth, height, width = prob_volume.shape
    starts = [max(0, int(mins[0] - margin_z)), max(0, int(mins[1] - margin_xy)), max(0, int(mins[2] - margin_xy))]
    stops = [min(depth, int(maxs[0] + margin_z)), min(height, int(maxs[1] + margin_xy)), min(width, int(maxs[2] + margin_xy))]
    return ((starts[0], stops[0]), (starts[1], stops[1]), (starts[2], stops[2]))


def sample_grid_in_bbox(
    shape: Tuple[int, int, int],
    stride_xy: int,
    stride_z: int,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
) -> np.ndarray:
    z_start, z_stop = bbox[0]
    y_start, y_stop = bbox[1]
    x_start, x_stop = bbox[2]
    z_indices = list(range(max(0, int(z_start)), min(int(shape[0]), int(z_stop)), max(1, stride_z)))
    y_indices = list(range(max(0, int(y_start)), min(int(shape[1]), int(y_stop)), max(1, stride_xy)))
    x_indices = list(range(max(0, int(x_start)), min(int(shape[2]), int(x_stop)), max(1, stride_xy)))
    if not z_indices or not y_indices or not x_indices:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray([(z, y, x) for z in z_indices for y in y_indices for x in x_indices], dtype=np.int32)
