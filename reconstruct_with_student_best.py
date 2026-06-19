import argparse
import json
import subprocess
import sys
from pathlib import Path


def append_bool_flag(command: list[str], flag_name: str, enabled: bool) -> None:
    command.append(flag_name if enabled else f"--no-{flag_name[2:]}")


def append_optional_arg(command: list[str], flag_name: str, value: object) -> None:
    if value is None:
        return
    command.extend([flag_name, str(value)])


def append_common_args(command: list[str], args: argparse.Namespace) -> None:
    append_optional_arg(command, "--case-name", args.case_name)
    append_optional_arg(command, "--model-dir", args.model_dir)
    append_optional_arg(command, "--preprocessed-dir", args.preprocessed_dir)
    append_optional_arg(command, "--teacher-pred-dir", args.teacher_pred_dir)
    append_optional_arg(command, "--output-dir", args.output_dir)
    append_optional_arg(command, "--batch-size", args.batch_size)
    append_optional_arg(command, "--mask-mode", args.mask_mode)
    append_optional_arg(command, "--patch-z", args.patch_z)
    append_optional_arg(command, "--patch-xy", args.patch_xy)


def append_best_params(command: list[str], best_params: dict[str, object]) -> None:
    command.extend(["--threshold", str(float(best_params["threshold"]))])
    command.extend(["--voxel-smooth-iterations", str(int(best_params["initial_voxel_smooth_iterations"]))])
    if "prob_sigma_z" in best_params:
        command.extend(["--prob-sigma-z", str(float(best_params["prob_sigma_z"]))])
    if "prob_sigma_xy" in best_params:
        command.extend(["--prob-sigma-xy", str(float(best_params["prob_sigma_xy"]))])
    command.extend(["--teacher-floor", str(float(best_params["teacher_floor"]))])
    command.extend(["--teacher-weight", str(float(best_params["teacher_weight"]))])
    command.extend(["--close-radius-z", str(int(best_params["close_radius_z"]))])
    command.extend(["--close-radius-xy", str(int(best_params["close_radius_xy"]))])
    command.extend(["--dilate-radius-z", str(int(best_params["dilate_radius_z"]))])
    command.extend(["--dilate-radius-xy", str(int(best_params["dilate_radius_xy"]))])
    command.extend(["--erode-radius-z", str(int(best_params["erode_radius_z"]))])
    command.extend(["--erode-radius-xy", str(int(best_params["erode_radius_xy"]))])
    command.extend(["--post-voxel-smooth-iterations", str(int(best_params["post_voxel_smooth_iterations"]))])
    command.extend(["--post-voxel-smooth-threshold", str(float(best_params["post_voxel_smooth_threshold"]))])
    command.extend(["--min-component-size", str(int(best_params["min_component_size"]))])
    append_bool_flag(command, "--fill-holes", bool(best_params.get("fill_holes", True)))
    append_bool_flag(command, "--keep-largest-component", bool(best_params.get("keep_largest_component", True)))

    support_enabled = bool(best_params.get("teacher_support_enabled", False))
    append_bool_flag(command, "--disable-teacher-support", not support_enabled)
    if support_enabled:
        append_optional_arg(command, "--teacher-support-threshold", best_params.get("teacher_support_threshold"))
        append_optional_arg(command, "--teacher-support-dilate-z", best_params.get("teacher_support_dilate_z"))
        append_optional_arg(command, "--teacher-support-dilate-xy", best_params.get("teacher_support_dilate_xy"))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    default_output_dir = base_dir / "outputs2" / "student_reconstruction"

    parser = argparse.ArgumentParser(
        description="Reconstruct with previously scanned best student reconstruction parameters."
    )
    parser.add_argument("--case-name", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--teacher-pred-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-mode", type=str, default=None)
    parser.add_argument("--patch-z", type=int, default=None)
    parser.add_argument("--patch-xy", type=int, default=None)
    args = parser.parse_args()

    summary_path = Path(args.output_dir) / "val_reconstruction_scan_summary.json"
    reconstruct_script = base_dir / "reconstruct_with_student.py"
    if not summary_path.exists():
        scan_command = [
            sys.executable,
            str(reconstruct_script),
            "--scan-val-search",
            "--output-dir",
            str(args.output_dir),
        ]
        append_common_args(scan_command, args)
        subprocess.run(scan_command, check=True)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing best-parameter summary after scan: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    best_params = summary.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError(f"Invalid best_params in {summary_path}")

    command = [
        sys.executable,
        str(reconstruct_script),
        "--no-scan-val-search",
    ]
    append_common_args(command, args)
    append_best_params(command, best_params)

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
