from __future__ import annotations

from pathlib import Path

import numpy as np


def trajectory_xyz(poses: list[np.ndarray]) -> np.ndarray:
    return np.array([pose[:3, 3] for pose in poses], dtype=np.float64)


def umeyama_align(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Align source points to target points with Umeyama similarity alignment."""
    if source_xyz.shape != target_xyz.shape:
        raise ValueError("source and target trajectories must have the same shape")
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("trajectories must be shaped as N x 3 arrays")

    mu_source = source_xyz.mean(axis=0)
    mu_target = target_xyz.mean(axis=0)
    source_centered = source_xyz - mu_source
    target_centered = target_xyz - mu_target

    covariance = source_centered.T @ target_centered / source_xyz.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)

    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1

    rotation = vt.T @ np.diag(sign) @ u.T
    scale = 1.0
    if with_scale:
        variance = np.mean(np.sum(source_centered**2, axis=1))
        if variance > 0:
            scale = float(np.sum(singular_values * sign) / variance)

    translation = mu_target - scale * rotation @ mu_source
    aligned = (scale * (rotation @ source_xyz.T)).T + translation
    return aligned, scale, rotation


def ate_rmse(
    predicted_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    align: bool = True,
    with_scale: bool = True,
) -> dict[str, float]:
    pred_xyz = trajectory_xyz(predicted_poses)
    gt_xyz = trajectory_xyz(gt_poses)
    count = min(len(pred_xyz), len(gt_xyz))
    pred_xyz = pred_xyz[:count]
    gt_xyz = gt_xyz[:count]

    if align:
        pred_xyz, scale, _ = umeyama_align(pred_xyz, gt_xyz, with_scale=with_scale)
    else:
        scale = 1.0

    errors = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "alignment_scale": float(scale),
        "frames": int(count),
    }


def write_kitti_poses(poses: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pose in poses:
        flat = pose[:3, :4].reshape(-1)
        lines.append(" ".join(f"{value:.9e}" for value in flat))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
