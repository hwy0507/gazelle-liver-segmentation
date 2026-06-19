import argparse
import copy
import json
from itertools import product
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
import torch
from torch import nn

from student_recon_common import build_features_for_centers, sample_grid_in_bbox, teacher_roi_bbox
from teacher_common import load_preprocessed_case
from teacher_metrics import binary_stats_from_probs


DEFAULT_SETTINGS = {
    "fine_patch_xy": 12,
    "fine_patch_z": 3,
    "local_patch_xy": 28,
    "local_patch_z": 7,
    "context_patch_xy": 40,
    "context_patch_z": 9,
    "stride_xy": 8,
    "stride_z": 2,
}


class InferenceMLP2Layer(nn.Module):
    def __init__(self, fc1_weight: np.ndarray, fc2_weight: np.ndarray, act1_weight: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(fc1_weight.shape[0], fc1_weight.shape[1], bias=False)
        self.act1 = nn.PReLU(num_parameters=1)
        self.fc2 = nn.Linear(fc2_weight.shape[0], fc2_weight.shape[1], bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.from_numpy(fc1_weight.T.astype(np.float32)))
            self.fc2.weight.copy_(torch.from_numpy(fc2_weight.T.astype(np.float32)))
            self.act1.weight.copy_(torch.from_numpy(act1_weight.astype(np.float32).reshape(-1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act1(self.fc1(x)))


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def default_model_dir(base_dir: Path) -> Path:
    return base_dir / "outputs2" / "student_models" / "organ"


def default_preprocessed_dir(base_dir: Path) -> Path:
    return base_dir / "outputs2" / "teacher_preprocessed" / "organ"


def default_teacher_pred_dir(base_dir: Path) -> Path:
    return base_dir / "outputs2" / "teacher_predictions" / "organ"


def default_output_dir(base_dir: Path) -> Path:
    return base_dir / "outputs2" / "student_reconstruction"


def load_pickle(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(model_dir: Path) -> InferenceMLP2Layer:
    optical_params = load_pickle(model_dir / "optical_params.pkl")
    nonlinear_params = load_pickle(model_dir / "nonlinear_params.pkl")
    fc1_weight = np.asarray(optical_params["fc1.weight"], dtype=np.float32)
    fc2_weight = np.asarray(optical_params["fc2.weight"], dtype=np.float32)
    act1_weight = np.asarray(nonlinear_params["act1.weight"], dtype=np.float32)
    return InferenceMLP2Layer(fc1_weight, fc2_weight, act1_weight)


def load_training_summary(model_dir: Path) -> Dict[str, object]:
    summary_path = model_dir / "training_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_case_path(preprocessed_dir: Path, case_name: str | None) -> Tuple[str, Path, str]:
    split_dirs = [preprocessed_dir / "val", preprocessed_dir / "train"]
    if case_name:
        for split_name, split_dir in (("val", split_dirs[0]), ("train", split_dirs[1])):
            case_path = split_dir / f"{case_name}.npz"
            if case_path.exists():
                return case_name, case_path, split_name
        raise FileNotFoundError(f"Case not found in preprocessed data: {case_name}")
    for split_name, split_dir in (("val", split_dirs[0]), ("train", split_dirs[1])):
        candidates = sorted(split_dir.glob("*.npz"))
        if candidates:
            case_path = candidates[0]
            return case_path.stem, case_path, split_name
    raise FileNotFoundError(f"No preprocessed cases found in {preprocessed_dir}")


def list_split_case_paths(preprocessed_dir: Path, split_name: str) -> List[Path]:
    split_dir = preprocessed_dir / split_name
    return sorted(split_dir.glob("*.npz"))


def load_teacher_probability(prob_npz_path: Path) -> np.ndarray:
    with np.load(str(prob_npz_path), allow_pickle=False) as data:
        return np.asarray(data["probability"], dtype=np.float32)


def predict_probabilities(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probs: List[np.ndarray] = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            xb = torch.from_numpy(features[start:end]).to(device)
            logits = model(xb)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float32)
            probs.append(batch_probs)
    return np.concatenate(probs, axis=0)


def fill_point_mask(centers: np.ndarray, probs: np.ndarray, volume_shape: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.zeros(tuple(int(v) for v in volume_shape.tolist()), dtype=np.uint8)
    chosen = centers[probs >= threshold]
    if chosen.size > 0:
        mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = 1
    return mask


def point_mask_to_patch_mask(
    centers: np.ndarray,
    probs: np.ndarray,
    volume_shape: np.ndarray,
    threshold: float,
    patch_z: int,
    patch_xy: int,
) -> np.ndarray:
    mask = np.zeros(tuple(int(v) for v in volume_shape.tolist()), dtype=np.uint8)
    selected = centers[probs >= threshold]
    if selected.size == 0:
        return mask
    z_half = patch_z // 2
    xy_half = patch_xy // 2
    depth, height, width = mask.shape
    for z, y, x in selected:
        z0 = max(0, int(z) - z_half)
        z1 = min(depth, int(z) + z_half + 1)
        y0 = max(0, int(y) - xy_half)
        y1 = min(height, int(y) + xy_half + 1)
        x0 = max(0, int(x) - xy_half)
        x1 = min(width, int(x) + xy_half + 1)
        mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


def binary_dilate(mask: np.ndarray, radius_z: int, radius_xy: int) -> np.ndarray:
    if radius_z <= 0 and radius_xy <= 0:
        return mask.astype(np.uint8)
    dilated = np.zeros_like(mask, dtype=bool)
    source = mask.astype(bool)
    for dz in range(-radius_z, radius_z + 1):
        for dy in range(-radius_xy, radius_xy + 1):
            for dx in range(-radius_xy, radius_xy + 1):
                shifted = np.roll(source, shift=(dz, dy, dx), axis=(0, 1, 2))
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
                dilated |= shifted
    return dilated.astype(np.uint8)


def binary_erode(mask: np.ndarray, radius_z: int, radius_xy: int) -> np.ndarray:
    if radius_z <= 0 and radius_xy <= 0:
        return mask.astype(np.uint8)
    eroded = mask.astype(bool).copy()
    source = mask.astype(bool)
    for dz in range(-radius_z, radius_z + 1):
        for dy in range(-radius_xy, radius_xy + 1):
            for dx in range(-radius_xy, radius_xy + 1):
                shifted = np.roll(source, shift=(dz, dy, dx), axis=(0, 1, 2))
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
                eroded &= shifted
    return eroded.astype(np.uint8)


def binary_close(mask: np.ndarray, radius_z: int, radius_xy: int) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius_z, radius_xy), radius_z, radius_xy)


def smooth_binary_mask(
    mask: np.ndarray,
    close_radius_z: int,
    close_radius_xy: int,
    dilate_radius_z: int,
    dilate_radius_xy: int,
    erode_radius_z: int,
    erode_radius_xy: int,
) -> np.ndarray:
    smoothed = mask.astype(np.uint8)
    smoothed = binary_close(smoothed, close_radius_z, close_radius_xy)
    smoothed = binary_dilate(smoothed, dilate_radius_z, dilate_radius_xy)
    smoothed = binary_erode(smoothed, erode_radius_z, erode_radius_xy)
    return smoothed.astype(np.uint8)


def add_gaussian_probability_field(
    centers: np.ndarray,
    probs: np.ndarray,
    volume_shape: np.ndarray,
    sigma_z: float,
    sigma_xy: float,
    truncate: float,
) -> np.ndarray:
    depth, height, width = (int(v) for v in volume_shape.tolist())
    field = np.zeros((depth, height, width), dtype=np.float32)
    weight = np.zeros((depth, height, width), dtype=np.float32)
    radius_z = max(1, int(round(truncate * sigma_z)))
    radius_xy = max(1, int(round(truncate * sigma_xy)))
    z_offsets = np.arange(-radius_z, radius_z + 1, dtype=np.int32)
    y_offsets = np.arange(-radius_xy, radius_xy + 1, dtype=np.int32)
    x_offsets = np.arange(-radius_xy, radius_xy + 1, dtype=np.int32)
    zz, yy, xx = np.meshgrid(z_offsets, y_offsets, x_offsets, indexing="ij")
    kernel = np.exp(
        -0.5
        * (
            (zz.astype(np.float32) / max(sigma_z, 1e-6)) ** 2
            + (yy.astype(np.float32) / max(sigma_xy, 1e-6)) ** 2
            + (xx.astype(np.float32) / max(sigma_xy, 1e-6)) ** 2
        )
    ).astype(np.float32)
    kernel /= np.max(kernel)
    for (z, y, x), prob in zip(centers, probs):
        z0 = max(0, int(z) - radius_z)
        z1 = min(depth, int(z) + radius_z + 1)
        y0 = max(0, int(y) - radius_xy)
        y1 = min(height, int(y) + radius_xy + 1)
        x0 = max(0, int(x) - radius_xy)
        x1 = min(width, int(x) + radius_xy + 1)
        kz0 = z0 - (int(z) - radius_z)
        kz1 = kz0 + (z1 - z0)
        ky0 = y0 - (int(y) - radius_xy)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (int(x) - radius_xy)
        kx1 = kx0 + (x1 - x0)
        local_kernel = kernel[kz0:kz1, ky0:ky1, kx0:kx1]
        field[z0:z1, y0:y1, x0:x1] += local_kernel * float(prob)
        weight[z0:z1, y0:y1, x0:x1] += local_kernel
    valid = weight > 1e-8
    field[valid] /= weight[valid]
    return field.astype(np.float32)


def smooth_scalar_volume(volume: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return volume.astype(np.float32)
    smoothed = volume.astype(np.float32).copy()
    for _ in range(iterations):
        padded = np.pad(smoothed, ((1, 1), (1, 1), (1, 1)), mode="edge")
        neighbor_sum = np.zeros_like(smoothed, dtype=np.float32)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    neighbor_sum += padded[
                        dz : dz + smoothed.shape[0],
                        dy : dy + smoothed.shape[1],
                        dx : dx + smoothed.shape[2],
                    ]
        smoothed = neighbor_sum / 27.0
    return smoothed


def probability_field_to_mask(field: np.ndarray, smooth_iterations: int, threshold: float) -> np.ndarray:
    smoothed = smooth_scalar_volume(field, smooth_iterations)
    return (smoothed >= threshold).astype(np.uint8)


def build_teacher_support_mask(
    teacher_prob: np.ndarray,
    threshold: float,
    dilate_radius_z: int,
    dilate_radius_xy: int,
) -> np.ndarray:
    support = (teacher_prob >= threshold).astype(np.uint8)
    if np.any(support > 0):
        support = binary_dilate(support, dilate_radius_z, dilate_radius_xy)
    return support.astype(np.uint8)


def build_fused_probability_field(
    centers: np.ndarray,
    probs: np.ndarray,
    teacher_prob: np.ndarray,
    volume_shape: np.ndarray,
    sigma_z: float,
    sigma_xy: float,
    truncate: float,
    teacher_floor: float,
    teacher_weight: float,
) -> np.ndarray:
    student_field = add_gaussian_probability_field(
        centers,
        probs,
        volume_shape,
        sigma_z,
        sigma_xy,
        truncate,
    )
    teacher_field = np.clip(teacher_prob.astype(np.float32), 0.0, 1.0)
    teacher_norm = np.clip((teacher_field - float(teacher_floor)) / max(1.0 - float(teacher_floor), 1e-6), 0.0, 1.0)
    fused = float(teacher_weight) * teacher_norm + float(1.0 - teacher_weight) * student_field
    return np.clip(fused * teacher_norm, 0.0, 1.0).astype(np.float32)


def build_initial_mask(
    centers: np.ndarray,
    probs: np.ndarray,
    teacher_prob: np.ndarray,
    volume_shape: np.ndarray,
    mask_mode: str,
    threshold: float,
    patch_z: int,
    patch_xy: int,
    prob_sigma_z: float,
    prob_sigma_xy: float,
    prob_truncate: float,
    voxel_smooth_iterations: int,
    teacher_floor: float,
    teacher_weight: float,
) -> np.ndarray:
    if mask_mode == "point":
        return fill_point_mask(centers, probs, volume_shape, threshold)
    if mask_mode == "patch":
        return point_mask_to_patch_mask(centers, probs, volume_shape, threshold, patch_z, patch_xy)
    probability_field = build_fused_probability_field(
        centers,
        probs,
        teacher_prob,
        volume_shape,
        prob_sigma_z,
        prob_sigma_xy,
        prob_truncate,
        teacher_floor,
        teacher_weight,
    )
    return probability_field_to_mask(probability_field, voxel_smooth_iterations, threshold)


def voxel_field_smooth_mask(mask: np.ndarray, smooth_iterations: int, threshold: float) -> np.ndarray:
    if smooth_iterations <= 0:
        return mask.astype(np.uint8)
    volume = smooth_scalar_volume(mask.astype(np.float32), smooth_iterations)
    return (volume >= threshold).astype(np.uint8)


def save_nifti(mask: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine.astype(np.float32))
    nib.save(nii, str(output_path))


def gaussian_kernel1d(sigma: float, truncate: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)
    radius = max(1, int(round(truncate * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2).astype(np.float32)
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def convolve_along_axis(volume: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size == 1:
        return volume.astype(np.float32)
    radius = kernel.size // 2
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(volume.astype(np.float32), pad_width, mode="edge")
    output = np.zeros_like(volume, dtype=np.float32)
    slicer = [slice(None), slice(None), slice(None)]
    for idx, weight in enumerate(kernel):
        start = idx
        stop = start + volume.shape[axis]
        slicer[axis] = slice(start, stop)
        output += padded[tuple(slicer)] * float(weight)
    return output


def gaussian_smooth_volume(volume: np.ndarray, sigma_z: float, sigma_xy: float, truncate: float) -> np.ndarray:
    smoothed = volume.astype(np.float32)
    smoothed = convolve_along_axis(smoothed, gaussian_kernel1d(sigma_z, truncate), axis=0)
    kernel_xy = gaussian_kernel1d(sigma_xy, truncate)
    smoothed = convolve_along_axis(smoothed, kernel_xy, axis=1)
    smoothed = convolve_along_axis(smoothed, kernel_xy, axis=2)
    return smoothed.astype(np.float32)


def build_surface_field(mask: np.ndarray, sigma_z: float, sigma_xy: float, truncate: float) -> np.ndarray:
    base = mask.astype(np.float32)
    if sigma_z <= 0 and sigma_xy <= 0:
        return base
    return gaussian_smooth_volume(base, sigma_z, sigma_xy, truncate)


def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    depth, height, width = mask.shape
    components: List[List[Tuple[int, int, int]]] = []
    positive_coords = np.argwhere(mask > 0)
    for coord in positive_coords:
        z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
        if visited[z, y, x]:
            continue
        stack = [(z, y, x)]
        component: List[Tuple[int, int, int]] = []
        visited[z, y, x] = True
        while stack:
            cz, cy, cx = stack.pop()
            component.append((cz, cy, cx))
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if nz < 0 or ny < 0 or nx < 0 or nz >= depth or ny >= height or nx >= width:
                    continue
                if visited[nz, ny, nx] or mask[nz, ny, nx] == 0:
                    continue
                visited[nz, ny, nx] = True
                stack.append((nz, ny, nx))
        components.append(component)
    return components


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    components = connected_components(mask)
    if not components:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = max(components, key=len)
    output = np.zeros_like(mask, dtype=np.uint8)
    for z, y, x in largest:
        output[z, y, x] = 1
    return output


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(np.uint8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for component in connected_components(mask):
        if len(component) < min_size:
            continue
        for z, y, x in component:
            output[z, y, x] = 1
    return output


def fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    solid = mask.astype(bool)
    background_visited = np.zeros_like(solid, dtype=bool)
    background = ~solid
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    depth, height, width = solid.shape
    stack: List[Tuple[int, int, int]] = []

    def push_if_background(z: int, y: int, x: int) -> None:
        if background[z, y, x] and not background_visited[z, y, x]:
            background_visited[z, y, x] = True
            stack.append((z, y, x))

    for z in range(depth):
        for y in range(height):
            push_if_background(z, y, 0)
            push_if_background(z, y, width - 1)
    for z in range(depth):
        for x in range(width):
            push_if_background(z, 0, x)
            push_if_background(z, height - 1, x)
    for y in range(height):
        for x in range(width):
            push_if_background(0, y, x)
            push_if_background(depth - 1, y, x)

    while stack:
        cz, cy, cx = stack.pop()
        for dz, dy, dx in neighbors:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if nz < 0 or ny < 0 or nx < 0 or nz >= depth or ny >= height or nx >= width:
                continue
            if background_visited[nz, ny, nx] or not background[nz, ny, nx]:
                continue
            background_visited[nz, ny, nx] = True
            stack.append((nz, ny, nx))

    filled = solid | (~background_visited & background)
    return filled.astype(np.uint8)


def postprocess_liver_mask(
    mask: np.ndarray,
    close_radius_z: int,
    close_radius_xy: int,
    dilate_radius_z: int,
    dilate_radius_xy: int,
    erode_radius_z: int,
    erode_radius_xy: int,
    voxel_smooth_iterations: int,
    voxel_smooth_threshold: float,
    fill_holes: bool,
    keep_largest: bool,
    min_component_size: int,
) -> np.ndarray:
    processed = smooth_binary_mask(
        mask,
        close_radius_z,
        close_radius_xy,
        dilate_radius_z,
        dilate_radius_xy,
        erode_radius_z,
        erode_radius_xy,
    )
    if fill_holes and np.any(processed > 0):
        processed = fill_holes_3d(processed)
    processed = voxel_field_smooth_mask(processed, voxel_smooth_iterations, voxel_smooth_threshold)
    if fill_holes and np.any(processed > 0):
        processed = fill_holes_3d(processed)
    if keep_largest and np.any(processed > 0):
        processed = keep_largest_component(processed)
    processed = remove_small_components(processed, min_component_size)
    return processed.astype(np.uint8)


def apply_reconstruction_constraints(
    initial_mask: np.ndarray,
    teacher_support_mask: np.ndarray | None,
    close_radius_z: int,
    close_radius_xy: int,
    dilate_radius_z: int,
    dilate_radius_xy: int,
    erode_radius_z: int,
    erode_radius_xy: int,
    voxel_smooth_iterations: int,
    voxel_smooth_threshold: float,
    fill_holes: bool,
    keep_largest_component: bool,
    min_component_size: int,
) -> np.ndarray:
    constrained = initial_mask.astype(np.uint8)
    if teacher_support_mask is not None and np.any(teacher_support_mask > 0):
        constrained = (constrained.astype(bool) & teacher_support_mask.astype(bool)).astype(np.uint8)
    return postprocess_liver_mask(
        constrained,
        close_radius_z,
        close_radius_xy,
        dilate_radius_z,
        dilate_radius_xy,
        erode_radius_z,
        erode_radius_xy,
        voxel_smooth_iterations,
        voxel_smooth_threshold,
        fill_holes,
        keep_largest_component,
        min_component_size,
    )


def reconstruction_stats(pred_mask: np.ndarray, reference_mask: np.ndarray) -> Dict[str, float]:
    pred_tensor = torch.from_numpy(pred_mask[None, None, ...].astype(np.float32))
    ref_tensor = torch.from_numpy(reference_mask[None, None, ...].astype(np.float32))
    return binary_stats_from_probs(pred_tensor, ref_tensor, threshold=0.5)


def reconstruction_score(metrics: Dict[str, float]) -> float:
    return (
        0.55 * float(metrics.get("dice", 0.0))
        + 0.20 * float(metrics.get("iou", 0.0))
        + 0.15 * float(metrics.get("recall", 0.0))
        + 0.10 * float(metrics.get("precision", 0.0))
    )


def reconstruction_objective(metrics: Dict[str, float], pred_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    base = reconstruction_score(metrics)
    ref_voxels = int(reference_mask.sum())
    pred_voxels = int(pred_mask.sum())
    if ref_voxels <= 0:
        return base
    volume_error = abs(pred_voxels - ref_voxels) / float(max(1, ref_voxels))
    empty_penalty = 0.25 if pred_voxels <= 0 else 0.0
    return float(base - 0.12 * min(volume_error, 1.5) - empty_penalty)


def candidate_postprocess_presets() -> List[Dict[str, object]]:
    return [
        {
            "close_radius_z": 0,
            "close_radius_xy": 1,
            "dilate_radius_z": 0,
            "dilate_radius_xy": 0,
            "erode_radius_z": 1,
            "erode_radius_xy": 1,
            "post_voxel_smooth_iterations": 1,
            "post_voxel_smooth_threshold": 0.50,
            "fill_holes": True,
            "keep_largest_component": True,
            "min_component_size": 64,
        },
        {
            "close_radius_z": 0,
            "close_radius_xy": 1,
            "dilate_radius_z": 0,
            "dilate_radius_xy": 0,
            "erode_radius_z": 1,
            "erode_radius_xy": 1,
            "post_voxel_smooth_iterations": 2,
            "post_voxel_smooth_threshold": 0.45,
            "fill_holes": True,
            "keep_largest_component": True,
            "min_component_size": 128,
        },
        {
            "close_radius_z": 0,
            "close_radius_xy": 2,
            "dilate_radius_z": 0,
            "dilate_radius_xy": 0,
            "erode_radius_z": 1,
            "erode_radius_xy": 1,
            "post_voxel_smooth_iterations": 2,
            "post_voxel_smooth_threshold": 0.45,
            "fill_holes": True,
            "keep_largest_component": True,
            "min_component_size": 128,
        },
    ]


def build_search_candidates(enable_teacher_support: bool) -> List[Dict[str, object]]:
    threshold_candidates = [0.48, 0.56, 0.64]
    initial_smooth_candidates = [1, 2]
    sigma_candidates = [(1.0, 2.5), (1.5, 3.0), (2.0, 4.0)]
    teacher_blend_candidates = [(0.10, 0.55), (0.20, 0.70), (0.25, 0.80)]
    support_candidates: List[Tuple[float | None, int | None]] = [(None, None)]
    if enable_teacher_support:
        support_candidates.extend([(0.25, 4), (0.30, 6)])

    candidates: List[Dict[str, object]] = []
    postprocess_presets = candidate_postprocess_presets()
    for threshold, initial_smooth_iterations, sigmas, teacher_blend, support_config, postprocess in product(
        threshold_candidates,
        initial_smooth_candidates,
        sigma_candidates,
        teacher_blend_candidates,
        support_candidates,
        postprocess_presets,
    ):
        sigma_z, sigma_xy = sigmas
        teacher_floor, teacher_weight = teacher_blend
        support_threshold, support_dilate_xy = support_config
        candidate = {
            "threshold": float(threshold),
            "initial_voxel_smooth_iterations": int(initial_smooth_iterations),
            "prob_sigma_z": float(sigma_z),
            "prob_sigma_xy": float(sigma_xy),
            "teacher_floor": float(teacher_floor),
            "teacher_weight": float(teacher_weight),
            "teacher_support_enabled": bool(enable_teacher_support and support_threshold is not None and support_dilate_xy is not None),
            "teacher_support_threshold": None if support_threshold is None else float(support_threshold),
            "teacher_support_dilate_z": None if support_threshold is None else 2,
            "teacher_support_dilate_xy": None if support_dilate_xy is None else int(support_dilate_xy),
            **copy.deepcopy(postprocess),
        }
        candidates.append(candidate)
    return candidates


def build_refined_candidates(best_candidate: Dict[str, object], enable_teacher_support: bool) -> List[Dict[str, object]]:
    threshold_center = float(best_candidate["threshold"])
    teacher_floor_center = float(best_candidate["teacher_floor"])
    teacher_weight_center = float(best_candidate["teacher_weight"])
    base_support_threshold = best_candidate.get("teacher_support_threshold")
    support_threshold_center = None if base_support_threshold is None else float(base_support_threshold)
    min_component_center = int(best_candidate["min_component_size"])

    thresholds = sorted({round(max(0.35, min(0.75, threshold_center + delta)), 2) for delta in (-0.04, -0.02, 0.0, 0.02)})
    teacher_floors = sorted({round(max(0.10, min(0.45, teacher_floor_center + delta)), 2) for delta in (-0.05, 0.0, 0.05)})
    teacher_weights = sorted({round(max(0.45, min(0.90, teacher_weight_center + delta)), 2) for delta in (-0.10, -0.05, 0.0, 0.05)})
    min_component_sizes = sorted({max(32, int(v)) for v in (min_component_center, min_component_center * 2)})

    support_candidates: List[Tuple[float | None, int | None]] = [(None, None)]
    if enable_teacher_support:
        if support_threshold_center is None:
            support_candidates.extend([(0.25, 4), (0.30, 6)])
        else:
            dilate_center = int(best_candidate.get("teacher_support_dilate_xy") or 4)
            local_thresholds = sorted({round(max(0.15, min(0.45, support_threshold_center + delta)), 2) for delta in (-0.05, 0.0, 0.05)})
            local_dilates = sorted({max(2, int(v)) for v in (dilate_center, dilate_center + 2)})
            support_candidates = [(thr, dilate) for thr in local_thresholds for dilate in local_dilates]

    refined: List[Dict[str, object]] = []
    for threshold, teacher_floor, teacher_weight, min_component_size, support_config in product(
        thresholds,
        teacher_floors,
        teacher_weights,
        min_component_sizes,
        support_candidates,
    ):
        support_threshold, support_dilate_xy = support_config
        candidate = dict(best_candidate)
        candidate["threshold"] = float(threshold)
        candidate["teacher_floor"] = float(teacher_floor)
        candidate["teacher_weight"] = float(teacher_weight)
        candidate["min_component_size"] = int(min_component_size)
        candidate["teacher_support_enabled"] = bool(enable_teacher_support and support_threshold is not None and support_dilate_xy is not None)
        candidate["teacher_support_threshold"] = None if support_threshold is None else float(support_threshold)
        candidate["teacher_support_dilate_z"] = None if support_threshold is None else 2
        candidate["teacher_support_dilate_xy"] = None if support_dilate_xy is None else int(support_dilate_xy)
        refined.append(candidate)
    return refined


def evaluate_reconstruction_candidate(
    case_data: Dict[str, object],
    mask_mode: str,
    patch_z: int,
    patch_xy: int,
    prob_truncate: float,
    candidate: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, float], float]:
    teacher_support_mask = None
    if bool(candidate.get("teacher_support_enabled", False)):
        teacher_support_mask = build_teacher_support_mask(
            np.asarray(case_data["teacher_prob"], dtype=np.float32),
            float(candidate["teacher_support_threshold"]),
            int(candidate["teacher_support_dilate_z"]),
            int(candidate["teacher_support_dilate_xy"]),
        )
    initial_mask = build_initial_mask(
        np.asarray(case_data["centers"], dtype=np.int32),
        np.asarray(case_data["probs"], dtype=np.float32),
        np.asarray(case_data["teacher_prob"], dtype=np.float32),
        np.asarray(case_data["volume_shape"], dtype=np.int32),
        mask_mode,
        float(candidate["threshold"]),
        patch_z,
        patch_xy,
        float(candidate["prob_sigma_z"]),
        float(candidate["prob_sigma_xy"]),
        prob_truncate,
        int(candidate["initial_voxel_smooth_iterations"]),
        float(candidate["teacher_floor"]),
        float(candidate["teacher_weight"]),
    )
    mask = apply_reconstruction_constraints(
        initial_mask,
        teacher_support_mask,
        int(candidate["close_radius_z"]),
        int(candidate["close_radius_xy"]),
        int(candidate["dilate_radius_z"]),
        int(candidate["dilate_radius_xy"]),
        int(candidate["erode_radius_z"]),
        int(candidate["erode_radius_xy"]),
        int(candidate["post_voxel_smooth_iterations"]),
        float(candidate["post_voxel_smooth_threshold"]),
        bool(candidate["fill_holes"]),
        bool(candidate["keep_largest_component"]),
        int(candidate["min_component_size"]),
    )
    reference_mask = np.asarray(case_data["reference_mask"], dtype=np.uint8)
    metrics = reconstruction_stats(mask, reference_mask)
    score = reconstruction_objective(metrics, mask, reference_mask)
    return mask.astype(np.uint8), None if teacher_support_mask is None else teacher_support_mask.astype(np.uint8), metrics, float(score)


def search_candidates_across_cases(
    cases: List[Dict[str, object]],
    mask_mode: str,
    patch_z: int,
    patch_xy: int,
    prob_truncate: float,
    candidates: List[Dict[str, object]],
    progress_prefix: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    best_result: Dict[str, object] | None = None
    history: List[Dict[str, object]] = []
    total_candidates = len(candidates)
    for candidate_index, candidate in enumerate(candidates, start=1):
        case_scores: List[float] = []
        metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0}
        foreground_voxels = 0
        for case_data in cases:
            mask, _, metrics, score = evaluate_reconstruction_candidate(case_data, mask_mode, patch_z, patch_xy, prob_truncate, candidate)
            case_scores.append(float(score))
            foreground_voxels += int(mask.sum())
            for key in metric_sums:
                metric_sums[key] += float(metrics.get(key, 0.0))
        mean_metrics = {key: value / float(max(1, len(cases))) for key, value in metric_sums.items()}
        result = dict(candidate)
        result["score"] = float(np.mean(case_scores))
        result["score_std"] = float(np.std(case_scores))
        result["foreground_voxels_mean"] = float(foreground_voxels) / float(max(1, len(cases)))
        result["case_count"] = int(len(cases))
        result["metrics"] = mean_metrics
        history.append(result)
        accuracy_percent = 100.0 * float(mean_metrics.get("dice", 0.0))
        support_display = "off"
        if bool(result.get("teacher_support_enabled", False)):
            support_display = f"thr={result['teacher_support_threshold']:.2f},dxy={int(result['teacher_support_dilate_xy'])}"
        print(
            f"[{progress_prefix}] candidate {candidate_index}/{total_candidates} | "
            f"acc={accuracy_percent:.2f}% | "
            f"thr={float(result['threshold']):.2f} | "
            f"sigma=({float(result['prob_sigma_z']):.1f},{float(result['prob_sigma_xy']):.1f}) | "
            f"tfloor={float(result['teacher_floor']):.2f} | "
            f"tweight={float(result['teacher_weight']):.2f} | "
            f"support={support_display} | "
            f"post=({int(result['post_voxel_smooth_iterations'])},{float(result['post_voxel_smooth_threshold']):.2f}) | "
            f"min_comp={int(result['min_component_size'])}",
            flush=True,
        )
        if best_result is None or float(result["score"]) > float(best_result["score"]):
            best_result = result
        if candidate_index == 1 or candidate_index == total_candidates or candidate_index % 20 == 0:
            print(
                f"[{progress_prefix}] scanned {candidate_index}/{total_candidates} candidates | "
                f"best_acc={100.0 * float(best_result['metrics']['dice']):.2f}% | best_score={float(best_result['score']):.5f}",
                flush=True,
            )
    if best_result is None:
        raise RuntimeError("Candidate search did not evaluate any reconstruction setting.")
    history.sort(key=lambda item: float(item["score"]), reverse=True)
    return best_result, history


def search_reconstruction_parameters(
    centers: np.ndarray,
    probs: np.ndarray,
    volume_shape: np.ndarray,
    mask_mode: str,
    patch_z: int,
    patch_xy: int,
    prob_sigma_z: float,
    prob_sigma_xy: float,
    prob_truncate: float,
    reference_mask: np.ndarray,
    teacher_prob: np.ndarray,
    enable_teacher_support: bool,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray | None, List[Dict[str, object]]]:
    case_data = {
        "case_name": "single_case",
        "centers": centers.astype(np.int32),
        "probs": probs.astype(np.float32),
        "teacher_prob": teacher_prob.astype(np.float32),
        "volume_shape": volume_shape.astype(np.int32),
        "reference_mask": reference_mask.astype(np.uint8),
    }
    coarse_candidates = build_search_candidates(enable_teacher_support=enable_teacher_support)
    coarse_best, coarse_history = search_candidates_across_cases(
        [case_data],
        mask_mode,
        patch_z,
        patch_xy,
        prob_truncate,
        coarse_candidates,
        progress_prefix="reconstruct-search-coarse",
    )
    refined_candidates = build_refined_candidates(coarse_best, enable_teacher_support=enable_teacher_support)
    refined_best, refined_history = search_candidates_across_cases(
        [case_data],
        mask_mode,
        patch_z,
        patch_xy,
        prob_truncate,
        refined_candidates,
        progress_prefix="reconstruct-search-refine",
    )
    best_mask, best_support, metrics, score = evaluate_reconstruction_candidate(
        case_data,
        mask_mode,
        patch_z,
        patch_xy,
        prob_truncate,
        refined_best,
    )
    best_result = dict(refined_best)
    best_result["foreground_voxels"] = int(best_mask.sum())
    best_result["metrics"] = {key: float(value) for key, value in metrics.items()}
    best_result["score"] = float(score)
    history = coarse_history[:12] + refined_history[:24]

    if best_mask is None:
        raise RuntimeError("Automatic reconstruction parameter search failed to produce a candidate.")
    return best_result, best_mask, best_support, history


def build_mesh(volume: np.ndarray, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray] | None:
    if float(volume.max()) < level:
        return None
    if min(volume.shape) < 2:
        return None
    vertices, faces, _, _ = marching_cubes(volume.astype(np.float32), level=level)
    return vertices.astype(np.float32), faces.astype(np.int32)


def save_obj(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def vertices_to_world(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine, vertices).astype(np.float32)


def save_mesh_obj(
    mask: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    surface_smooth_sigma_z: float,
    surface_smooth_sigma_xy: float,
    surface_smooth_truncate: float,
) -> bool:
    surface_field = build_surface_field(mask, surface_smooth_sigma_z, surface_smooth_sigma_xy, surface_smooth_truncate)
    mesh = build_mesh(surface_field, level=0.5)
    if mesh is None:
        return False
    vertices, faces = mesh
    vertices = vertices_to_world(vertices, affine)
    save_obj(vertices, faces, output_path)
    return True


def save_single_mesh_preview(
    mask: np.ndarray,
    preview_path: Path,
    title: str,
    edge_color: Tuple[float, float, float, float],
    surface_smooth_sigma_z: float,
    surface_smooth_sigma_xy: float,
    surface_smooth_truncate: float,
) -> bool:
    surface_field = build_surface_field(mask, surface_smooth_sigma_z, surface_smooth_sigma_xy, surface_smooth_truncate)
    mesh = build_mesh(surface_field)
    if mesh is None:
        return False

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    vertices, faces = mesh
    poly = Poly3DCollection(vertices[faces], alpha=0.08)
    poly.set_facecolor((0.95, 0.95, 0.95, 0.08))
    poly.set_edgecolor(edge_color)
    poly.set_linewidth(0.35)
    ax.add_collection3d(poly)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1.0)
    ax.set_xlim(center[2] - radius, center[2] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[0] - radius, center[0] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=20, azim=35)
    ax.grid(False)
    ax.set_facecolor("white")
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(str(preview_path), dpi=200)
    plt.close(fig)
    return True


def save_mesh_preview_compare(
    pred_mask: np.ndarray,
    reference_mask: np.ndarray,
    preview_path: Path,
    surface_smooth_sigma_z: float,
    surface_smooth_sigma_xy: float,
    surface_smooth_truncate: float,
) -> bool:
    pred_surface = build_surface_field(pred_mask, surface_smooth_sigma_z, surface_smooth_sigma_xy, surface_smooth_truncate)
    ref_surface = build_surface_field(reference_mask, surface_smooth_sigma_z, surface_smooth_sigma_xy, surface_smooth_truncate)
    pred_mesh = build_mesh(pred_surface)
    ref_mesh = build_mesh(ref_surface)
    if pred_mesh is None and ref_mesh is None:
        return False

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    all_vertices: List[np.ndarray] = []

    if pred_mesh is not None:
        pred_vertices, pred_faces = pred_mesh
        pred_poly = Poly3DCollection(pred_vertices[pred_faces], alpha=0.08)
        pred_poly.set_facecolor((0.95, 0.95, 0.95, 0.08))
        pred_poly.set_edgecolor((0.85, 0.20, 0.10, 0.95))
        pred_poly.set_linewidth(0.35)
        ax.add_collection3d(pred_poly)
        all_vertices.append(pred_vertices)

    if ref_mesh is not None:
        ref_vertices, ref_faces = ref_mesh
        ref_poly = Poly3DCollection(ref_vertices[ref_faces], alpha=0.04)
        ref_poly.set_facecolor((0.95, 0.95, 0.95, 0.04))
        ref_poly.set_edgecolor((0.10, 0.35, 0.95, 0.80))
        ref_poly.set_linewidth(0.28)
        ax.add_collection3d(ref_poly)
        all_vertices.append(ref_vertices)

    stacked = np.concatenate(all_vertices, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1.0)
    ax.set_xlim(center[2] - radius, center[2] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[0] - radius, center[0] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Predicted vs Reference Liver Contours")
    ax.view_init(elev=20, azim=35)
    ax.grid(False)
    ax.set_facecolor("white")
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(str(preview_path), dpi=200)
    plt.close(fig)
    return True


def save_summary(
    output_path: Path,
    case_name: str,
    case_path: Path,
    teacher_prob_path: Path,
    model_dir: Path,
    threshold: float,
    mask_mode: str,
    roi_bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    features: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray,
    reference_mask: np.ndarray | None,
    teacher_support_mask: np.ndarray | None,
    auto_search_result: Dict[str, object] | None,
    auto_search_history: List[Dict[str, object]] | None,
) -> None:
    summary = {
        "case_name": case_name,
        "case_path": str(case_path),
        "teacher_prob_path": str(teacher_prob_path),
        "model_dir": str(model_dir),
        "threshold": float(threshold),
        "mask_mode": mask_mode,
        "roi_bbox": [[int(v) for v in axis] for axis in roi_bbox],
        "feature_shape": [int(v) for v in features.shape],
        "sample_count": int(probs.shape[0]),
        "positive_samples": int(np.sum(probs >= threshold)),
        "foreground_voxels": int(mask.sum()),
        "reference_foreground_voxels": int(reference_mask.sum()) if reference_mask is not None else None,
        "teacher_support_voxels": int(teacher_support_mask.sum()) if teacher_support_mask is not None else None,
        "has_reference_label": reference_mask is not None,
        "probability_min": float(np.min(probs)),
        "probability_max": float(np.max(probs)),
        "probability_mean": float(np.mean(probs)),
        "auto_search": auto_search_result,
        "auto_search_history": auto_search_history,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def load_case_for_reconstruction(
    case_path: Path,
    split_name: str,
    teacher_pred_dir: Path,
    expected_input_dim: int,
) -> Dict[str, object]:
    case_name = case_path.stem
    case = load_preprocessed_case(case_path)
    teacher_prob_path = teacher_pred_dir / split_name / f"{case_name}_prob.npz"
    if not teacher_prob_path.exists():
        raise FileNotFoundError(f"Missing teacher probability file: {teacher_prob_path}")
    teacher_prob = load_teacher_probability(teacher_prob_path)
    roi_bbox = teacher_roi_bbox(
        teacher_prob,
        threshold=0.35,
        margin_z=6,
        margin_xy=20,
    )
    if roi_bbox is None:
        shape = case["image"].shape
        roi_bbox = ((0, shape[0]), (0, shape[1]), (0, shape[2]))
    centers = sample_grid_in_bbox(
        tuple(int(v) for v in case["image"].shape),
        int(DEFAULT_SETTINGS["stride_xy"]),
        int(DEFAULT_SETTINGS["stride_z"]),
        roi_bbox,
    )
    if centers.size == 0:
        shape = case["image"].shape
        centers = np.asarray([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.int32)
    features = build_features_for_centers(case["image"], centers, DEFAULT_SETTINGS, roi_bbox)
    if expected_input_dim > 0 and features.shape[1] != expected_input_dim:
        raise ValueError(
            f"Feature dimension mismatch for {case_name}: generated={features.shape[1]} expected={expected_input_dim}."
        )
    return {
        "case_name": case_name,
        "case": case,
        "teacher_prob": teacher_prob.astype(np.float32),
        "teacher_prob_path": teacher_prob_path,
        "roi_bbox": roi_bbox,
        "centers": centers.astype(np.int32),
        "features": features.astype(np.float32),
        "reference_mask": case["label"].astype(np.uint8),
        "volume_shape": np.asarray(case["image"].shape, dtype=np.int32),
    }


def attach_case_probabilities(cases: List[Dict[str, object]], model: nn.Module, batch_size: int, device: torch.device) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for case_index, case_data in enumerate(cases, start=1):
        probs = predict_probabilities(model, np.asarray(case_data["features"], dtype=np.float32), batch_size, device)
        updated = dict(case_data)
        updated["probs"] = probs.astype(np.float32)
        enriched.append(updated)
        print(
            f"[reconstruct-scan] prepared case {case_index}/{len(cases)}: {case_data['case_name']} | samples={int(probs.shape[0])}",
            flush=True,
        )
    return enriched


def scan_validation_reconstruction_parameters(
    preprocessed_dir: Path,
    teacher_pred_dir: Path,
    model: nn.Module,
    expected_input_dim: int,
    batch_size: int,
    device: torch.device,
    mask_mode: str,
    patch_z: int,
    patch_xy: int,
    prob_truncate: float,
    enable_teacher_support: bool,
    limit_cases: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    case_paths = list_split_case_paths(preprocessed_dir, "val")
    if limit_cases > 0:
        case_paths = case_paths[:limit_cases]
    if not case_paths:
        raise FileNotFoundError(f"No validation cases found in {preprocessed_dir / 'val'}")

    prepared_cases = [
        load_case_for_reconstruction(case_path, "val", teacher_pred_dir, expected_input_dim)
        for case_path in case_paths
    ]
    cases_with_probs = attach_case_probabilities(prepared_cases, model, batch_size, device)

    coarse_candidates = build_search_candidates(enable_teacher_support=enable_teacher_support)
    coarse_best, coarse_history = search_candidates_across_cases(
        cases_with_probs,
        mask_mode,
        patch_z,
        patch_xy,
        prob_truncate,
        coarse_candidates,
        progress_prefix="reconstruct-val-coarse",
    )
    refined_candidates = build_refined_candidates(coarse_best, enable_teacher_support=enable_teacher_support)
    refined_best, refined_history = search_candidates_across_cases(
        cases_with_probs,
        mask_mode,
        patch_z,
        patch_xy,
        prob_truncate,
        refined_candidates,
        progress_prefix="reconstruct-val-refine",
    )
    history = coarse_history[:20] + refined_history[:40]
    return refined_best, history, cases_with_probs


def save_scan_summary(
    output_dir: Path,
    best_params: Dict[str, object],
    history: List[Dict[str, object]],
    case_count: int,
    mask_mode: str,
    patch_z: int,
    patch_xy: int,
    prob_truncate: float,
    teacher_support_enabled: bool,
) -> Path:
    scan_output_path = output_dir / "val_reconstruction_scan_summary.json"
    with scan_output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_count": case_count,
                "best_params": best_params,
                "history": history,
                "mask_mode": mask_mode,
                "patch_z": int(patch_z),
                "patch_xy": int(patch_xy),
                "prob_truncate": float(prob_truncate),
                "teacher_support_enabled": bool(teacher_support_enabled),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return scan_output_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reconstruct liver mask from outputs2 student parameters.")
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir(base_dir)))
    parser.add_argument("--preprocessed-dir", type=str, default=str(default_preprocessed_dir(base_dir)))
    parser.add_argument("--teacher-pred-dir", type=str, default=str(default_teacher_pred_dir(base_dir)))
    parser.add_argument("--case-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir(base_dir)))
    parser.add_argument("--roi-threshold", type=float, default=0.25)
    parser.add_argument("--roi-margin-z", type=int, default=12)
    parser.add_argument("--roi-margin-xy", type=int, default=36)
    parser.add_argument("--threshold", type=float, default=0.56)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--mask-mode", choices=["point", "patch", "probability"], default="probability")
    parser.add_argument("--patch-z", type=int, default=5)
    parser.add_argument("--patch-xy", type=int, default=11)
    parser.add_argument("--prob-sigma-z", type=float, default=1.5)
    parser.add_argument("--prob-sigma-xy", type=float, default=3.0)
    parser.add_argument("--prob-truncate", type=float, default=2.5)
    parser.add_argument("--voxel-smooth-iterations", type=int, default=2)
    parser.add_argument("--teacher-floor", type=float, default=0.20)
    parser.add_argument("--teacher-weight", type=float, default=0.75)
    parser.add_argument("--teacher-support-threshold", type=float, default=0.30)
    parser.add_argument("--teacher-support-dilate-z", type=int, default=0)
    parser.add_argument("--teacher-support-dilate-xy", type=int, default=4)
    parser.add_argument("--disable-teacher-support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-radius-z", type=int, default=0)
    parser.add_argument("--close-radius-xy", type=int, default=2)
    parser.add_argument("--dilate-radius-z", type=int, default=0)
    parser.add_argument("--dilate-radius-xy", type=int, default=0)
    parser.add_argument("--erode-radius-z", type=int, default=1)
    parser.add_argument("--erode-radius-xy", type=int, default=1)
    parser.add_argument("--post-voxel-smooth-iterations", type=int, default=2)
    parser.add_argument("--post-voxel-smooth-threshold", type=float, default=0.40)
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-component-size", type=int, default=128)
    parser.add_argument("--surface-smooth-sigma-z", type=float, default=1.2)
    parser.add_argument("--surface-smooth-sigma-xy", type=float, default=1.2)
    parser.add_argument("--surface-smooth-truncate", type=float, default=2.5)
    parser.add_argument("--auto-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scan-val-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-limit-cases", type=int, default=0)
    args = parser.parse_args()

    model_dir = resolve_path(base_dir, args.model_dir)
    preprocessed_dir = resolve_path(base_dir, args.preprocessed_dir)
    teacher_pred_dir = resolve_path(base_dir, args.teacher_pred_dir)
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "optical_params.pkl").exists():
        raise FileNotFoundError(f"Missing model file: {model_dir / 'optical_params.pkl'}")
    if not (model_dir / "nonlinear_params.pkl").exists():
        raise FileNotFoundError(f"Missing model file: {model_dir / 'nonlinear_params.pkl'}")

    training_summary = load_training_summary(model_dir)
    expected_input_dim = int(training_summary.get("input_dim", 0))
    case_name, case_path, split_name = choose_case_path(preprocessed_dir, args.case_name)
    teacher_prob_path = teacher_pred_dir / split_name / f"{case_name}_prob.npz"
    if not teacher_prob_path.exists():
        raise FileNotFoundError(f"Missing teacher probability file: {teacher_prob_path}")

    case = load_preprocessed_case(case_path)
    teacher_prob = load_teacher_probability(teacher_prob_path)
    roi_bbox = teacher_roi_bbox(
        teacher_prob,
        threshold=float(args.roi_threshold),
        margin_z=int(args.roi_margin_z),
        margin_xy=int(args.roi_margin_xy),
    )
    if roi_bbox is None:
        shape = case["image"].shape
        roi_bbox = ((0, shape[0]), (0, shape[1]), (0, shape[2]))

    centers = sample_grid_in_bbox(
        tuple(int(v) for v in case["image"].shape),
        int(DEFAULT_SETTINGS["stride_xy"]),
        int(DEFAULT_SETTINGS["stride_z"]),
        roi_bbox,
    )
    if centers.size == 0:
        shape = case["image"].shape
        centers = np.asarray([[shape[0] // 2, shape[1] // 2, shape[2] // 2]], dtype=np.int32)

    features = build_features_for_centers(case["image"], centers, DEFAULT_SETTINGS, roi_bbox)
    if expected_input_dim > 0 and features.shape[1] != expected_input_dim:
        raise ValueError(
            f"Feature dimension mismatch: generated={features.shape[1]} expected={expected_input_dim}. "
            f"Check that reconstruction settings match the training pipeline."
        )

    model = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.scan_val_search:
        best_params, scan_history, scan_cases = scan_validation_reconstruction_parameters(
            preprocessed_dir=preprocessed_dir,
            teacher_pred_dir=teacher_pred_dir,
            model=model,
            expected_input_dim=expected_input_dim,
            batch_size=int(args.batch_size),
            device=device,
            mask_mode=args.mask_mode,
            patch_z=int(args.patch_z),
            patch_xy=int(args.patch_xy),
            prob_truncate=float(args.prob_truncate),
            enable_teacher_support=not args.disable_teacher_support,
            limit_cases=int(args.scan_limit_cases),
        )
        print(
            f"[reconstruct-scan] best_acc={100.0 * float(best_params['metrics']['dice']):.2f}% | "
            f"best_params={json.dumps(best_params, ensure_ascii=False)}",
            flush=True,
        )
        scan_output_path = save_scan_summary(
            output_dir=output_dir,
            best_params=best_params,
            history=scan_history,
            case_count=len(scan_cases),
            mask_mode=args.mask_mode,
            patch_z=int(args.patch_z),
            patch_xy=int(args.patch_xy),
            prob_truncate=float(args.prob_truncate),
            teacher_support_enabled=not args.disable_teacher_support,
        )
        print(f"[reconstruct-scan] summary saved: {scan_output_path}", flush=True)
        return

    probs = predict_probabilities(model, features, args.batch_size, device)
    volume_shape = np.asarray(case["image"].shape, dtype=np.int32)
    reference_mask = case["label"].astype(np.uint8)
    auto_search_result = None
    auto_search_history = None

    if args.auto_search and np.any(reference_mask > 0):
        auto_search_result, mask, teacher_support_mask, auto_search_history = search_reconstruction_parameters(
            centers,
            probs,
            volume_shape,
            args.mask_mode,
            args.patch_z,
            args.patch_xy,
            args.prob_sigma_z,
            args.prob_sigma_xy,
            args.prob_truncate,
            reference_mask,
            teacher_prob,
            enable_teacher_support=not args.disable_teacher_support,
        )
    else:
        teacher_support_mask = None
        if not args.disable_teacher_support:
            teacher_support_mask = build_teacher_support_mask(
                teacher_prob,
                float(args.teacher_support_threshold),
                int(args.teacher_support_dilate_z),
                int(args.teacher_support_dilate_xy),
            )
        initial_mask = build_initial_mask(
            centers,
            probs,
            teacher_prob,
            volume_shape,
            args.mask_mode,
            args.threshold,
            args.patch_z,
            args.patch_xy,
            args.prob_sigma_z,
            args.prob_sigma_xy,
            args.prob_truncate,
            args.voxel_smooth_iterations,
            args.teacher_floor,
            args.teacher_weight,
        )
        mask = apply_reconstruction_constraints(
            initial_mask,
            teacher_support_mask,
            args.close_radius_z,
            args.close_radius_xy,
            args.dilate_radius_z,
            args.dilate_radius_xy,
            args.erode_radius_z,
            args.erode_radius_xy,
            args.post_voxel_smooth_iterations,
            args.post_voxel_smooth_threshold,
            args.fill_holes,
            args.keep_largest_component,
            args.min_component_size,
        )

    mask_path = output_dir / f"{case_name}_liver_mask.nii.gz"
    prob_path = output_dir / f"{case_name}_liver_prob.npy"
    mesh_obj_path = output_dir / f"{case_name}_liver.obj"
    preview_pred_path = output_dir / f"{case_name}_liver_preview_pred.png"
    preview_ref_path = output_dir / f"{case_name}_liver_preview_ref.png"
    preview_compare_path = output_dir / f"{case_name}_liver_preview_compare.png"
    summary_path = output_dir / f"{case_name}_reconstruction_summary.json"
    save_nifti(mask, case["affine"], mask_path)
    np.save(str(prob_path), probs.astype(np.float32))
    mesh_saved = save_mesh_obj(
        mask,
        case["affine"],
        mesh_obj_path,
        args.surface_smooth_sigma_z,
        args.surface_smooth_sigma_xy,
        args.surface_smooth_truncate,
    )
    preview_pred_saved = save_single_mesh_preview(
        mask,
        preview_pred_path,
        "Predicted Liver Reconstruction",
        (0.85, 0.20, 0.10, 0.95),
        args.surface_smooth_sigma_z,
        args.surface_smooth_sigma_xy,
        args.surface_smooth_truncate,
    )
    preview_ref_saved = save_single_mesh_preview(
        reference_mask,
        preview_ref_path,
        "Reference Liver Contour",
        (0.10, 0.35, 0.95, 0.80),
        args.surface_smooth_sigma_z,
        args.surface_smooth_sigma_xy,
        args.surface_smooth_truncate,
    )
    preview_compare_saved = save_mesh_preview_compare(
        mask,
        reference_mask,
        preview_compare_path,
        args.surface_smooth_sigma_z,
        args.surface_smooth_sigma_xy,
        args.surface_smooth_truncate,
    )
    save_summary(
        summary_path,
        case_name,
        case_path,
        teacher_prob_path,
        model_dir,
        args.threshold,
        args.mask_mode,
        roi_bbox,
        features,
        probs,
        mask,
        reference_mask,
        teacher_support_mask,
        auto_search_result,
        auto_search_history,
    )

    accuracy_percent = 0.0
    if np.any(reference_mask > 0):
        metrics = reconstruction_stats(mask, reference_mask)
        accuracy_percent = 100.0 * float(metrics.get("dice", 0.0))
    print(f"{accuracy_percent:.2f}%", flush=True)


if __name__ == "__main__":
    main()
