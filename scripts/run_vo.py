from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vo.dataset import KittiOdometryDataset
from vo.metrics import ate_rmse, trajectory_xyz, write_kitti_poses
from vo.pipeline import MonocularVisualOdometry


DEFAULT_DATA = PROJECT_ROOT / "data" / "kitti_mini"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monocular VO on a mini KITTI subset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--feature", choices=["orb", "sift", "both"], default="both")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-evo", action="store_true", help="Run evo_ape if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KittiOdometryDataset(root=args.data_root, sequence=args.sequence)
    image_paths = dataset.image_paths(max_frames=args.max_frames)
    images = [dataset.load_gray(path) for path in image_paths]
    gt_poses = dataset.load_poses(max_frames=len(images))
    camera_matrix = dataset.load_camera_matrix()

    features = ["orb", "sift"] if args.feature == "both" else [args.feature]
    summary = {
        "data_root": str(args.data_root.resolve()),
        "sequence": args.sequence,
        "frames": len(images),
        "features": {},
    }

    gt_path = args.output_dir / "gt_kitti.txt"
    write_kitti_poses(gt_poses[: len(images)], gt_path)

    rows = []
    for feature in features:
        print(f"Running {feature.upper()} on {len(images)} frames")
        vo = MonocularVisualOdometry(camera_matrix=camera_matrix, feature=feature)
        result = vo.run(images, gt_poses)
        pred_path = args.output_dir / f"pred_{feature}_kitti.txt"
        write_kitti_poses(result.poses, pred_path)

        metrics = ate_rmse(result.poses, gt_poses, align=True, with_scale=True)
        feature_summary = {
            **metrics,
            "avg_runtime_ms": result.avg_runtime_ms,
            "avg_good_matches": result.avg_good_matches,
            "avg_inlier_ratio": result.avg_inlier_ratio,
            "pred_kitti": str(pred_path),
        }

        if args.run_evo:
            feature_summary["evo"] = _run_evo(gt_path, pred_path)

        summary["features"][feature] = feature_summary
        rows.append(
            {
                "feature": feature,
                "frames": metrics["frames"],
                "ate_rmse_m": metrics["ate_rmse_m"],
                "ate_mean_m": metrics["ate_mean_m"],
                "ate_max_m": metrics["ate_max_m"],
                "alignment_scale": metrics["alignment_scale"],
                "avg_runtime_ms": result.avg_runtime_ms,
                "avg_good_matches": result.avg_good_matches,
                "avg_inlier_ratio": result.avg_inlier_ratio,
            }
        )
        _plot_trajectory(gt_poses, result.poses, args.output_dir / f"trajectory_{feature}.png", feature)

    _write_csv(rows, args.output_dir / "comparison.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    for row in rows:
        print(
            f"{row['feature'].upper():4s} | ATE RMSE: {row['ate_rmse_m']:.3f} m | "
            f"runtime: {row['avg_runtime_ms']:.1f} ms/frame | "
            f"matches: {row['avg_good_matches']:.1f} | "
            f"inlier ratio: {row['avg_inlier_ratio']:.2f}"
        )
    print(f"\nWrote results to {args.output_dir.resolve()}")


def _plot_trajectory(gt_poses, pred_poses, output_path: Path, feature: str) -> None:
    gt_xyz = trajectory_xyz(gt_poses)
    pred_xyz = trajectory_xyz(pred_poses)
    count = min(len(gt_xyz), len(pred_xyz))
    gt_xyz = gt_xyz[:count]
    pred_xyz = pred_xyz[:count]

    plt.figure(figsize=(8, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground truth", linewidth=2)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], label=f"{feature.upper()} VO", linewidth=2)
    plt.scatter(gt_xyz[0, 0], gt_xyz[0, 2], marker="o", label="Start")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(f"KITTI 00 Monocular VO - {feature.upper()}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_evo(gt_path: Path, pred_path: Path) -> dict:
    local_evo = Path(sys.executable).with_name("evo_ape.exe")
    commands = [
        [str(local_evo), "kitti", str(gt_path), str(pred_path), "--align", "--correct_scale", "-v"],
        ["evo_ape", "kitti", str(gt_path), str(pred_path), "--align", "--correct_scale", "-v"],
        [
            sys.executable,
            "-m",
            "evo.main_ape",
            "kitti",
            str(gt_path),
            str(pred_path),
            "--align",
            "--correct_scale",
            "-v",
        ],
    ]
    for command in commands:
        if command[0].endswith("evo_ape.exe") and not Path(command[0]).exists():
            continue
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return {
                "ok": True,
                "command": command,
                "output": (completed.stdout + completed.stderr).strip(),
            }
        last_error = completed.stderr
    return {"ok": False, "error": locals().get("last_error", "evo command not found")}


if __name__ == "__main__":
    main()
