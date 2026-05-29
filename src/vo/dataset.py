from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class KittiOdometryDataset:
    root: Path
    sequence: str = "00"

    @property
    def sequence_dir(self) -> Path:
        return self.root / "sequences" / self.sequence

    @property
    def image_dir(self) -> Path:
        return self.sequence_dir / "image_0"

    @property
    def pose_file(self) -> Path:
        return self.root / "poses" / f"{self.sequence}.txt"

    @property
    def calib_file(self) -> Path:
        return self.sequence_dir / "calib.txt"

    def image_paths(self, max_frames: int | None = None) -> list[Path]:
        paths = sorted(self.image_dir.glob("*.png"))
        if max_frames is not None:
            paths = paths[:max_frames]
        if len(paths) < 2:
            raise FileNotFoundError(
                f"Need at least two PNG frames in {self.image_dir}. "
                "Run scripts/download_kitti_mini.py first."
            )
        return paths

    def load_gray(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def load_camera_matrix(self) -> np.ndarray:
        projections = _load_calib(self.calib_file)
        if "P0" in projections:
            p = projections["P0"]
        elif "P2" in projections:
            p = projections["P2"]
        else:
            raise KeyError(f"Expected P0 or P2 in calibration file: {self.calib_file}")

        return np.array(
            [
                [p[0, 0], 0.0, p[0, 2]],
                [0.0, p[1, 1], p[1, 2]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def load_poses(self, max_frames: int | None = None) -> list[np.ndarray]:
        if not self.pose_file.exists():
            raise FileNotFoundError(f"Missing KITTI pose file: {self.pose_file}")

        poses: list[np.ndarray] = []
        for line in self.pose_file.read_text(encoding="utf-8").splitlines():
            values = [float(value) for value in line.split()]
            if len(values) != 12:
                continue
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = np.array(values, dtype=np.float64).reshape(3, 4)
            poses.append(pose)
            if max_frames is not None and len(poses) >= max_frames:
                break

        if len(poses) < 2:
            raise ValueError(f"Need at least two ground-truth poses in {self.pose_file}")
        return poses


def _load_calib(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration file: {path}")

    projections: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_values = line.split(":", maxsplit=1)
        values = [float(value) for value in raw_values.split()]
        if len(values) == 12:
            projections[key] = np.array(values, dtype=np.float64).reshape(3, 4)
    return projections
