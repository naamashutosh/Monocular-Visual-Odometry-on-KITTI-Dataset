from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass
class FrameStats:
    frame_index: int
    keypoints_prev: int
    keypoints_curr: int
    raw_matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    runtime_ms: float
    scale_m: float


@dataclass
class VoResult:
    feature: str
    poses: list[np.ndarray]
    stats: list[FrameStats]

    @property
    def avg_runtime_ms(self) -> float:
        return _mean([item.runtime_ms for item in self.stats])

    @property
    def avg_good_matches(self) -> float:
        return _mean([item.good_matches for item in self.stats])

    @property
    def avg_inlier_ratio(self) -> float:
        return _mean([item.inlier_ratio for item in self.stats])


class MonocularVisualOdometry:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        feature: str = "orb",
        min_matches: int = 12,
        ratio_test: float = 0.75,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
    ) -> None:
        self.camera_matrix = camera_matrix
        self.feature = feature.lower()
        self.min_matches = min_matches
        self.ratio_test = ratio_test
        self.ransac_prob = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.detector, self.norm_type = _make_detector(self.feature)

    def run(self, images: list[np.ndarray], gt_poses: list[np.ndarray]) -> VoResult:
        if len(images) != len(gt_poses):
            count = min(len(images), len(gt_poses))
            images = images[:count]
            gt_poses = gt_poses[:count]

        predicted_poses = [np.eye(4, dtype=np.float64)]
        stats: list[FrameStats] = []

        for idx in range(1, len(images)):
            start = perf_counter()
            prev_image = images[idx - 1]
            curr_image = images[idx]

            kp_prev, desc_prev = self.detector.detectAndCompute(prev_image, None)
            kp_curr, desc_curr = self.detector.detectAndCompute(curr_image, None)
            matches = self._match(desc_prev, desc_curr)

            relative_pose = np.eye(4, dtype=np.float64)
            inliers = 0
            inlier_ratio = 0.0
            scale = _relative_gt_scale(gt_poses[idx - 1], gt_poses[idx])

            if len(matches) >= self.min_matches:
                pts_prev = np.float32([kp_prev[match.queryIdx].pt for match in matches])
                pts_curr = np.float32([kp_curr[match.trainIdx].pt for match in matches])

                # Passing current points first gives an increment that composes
                # naturally with KITTI camera-to-world poses in this small VO loop.
                essential, mask = cv2.findEssentialMat(
                    pts_curr,
                    pts_prev,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=self.ransac_prob,
                    threshold=self.ransac_threshold,
                )

                if essential is not None and mask is not None:
                    _, rotation, translation, pose_mask = cv2.recoverPose(
                        essential,
                        pts_curr,
                        pts_prev,
                        self.camera_matrix,
                        mask=mask,
                    )
                    inliers = int(np.count_nonzero(pose_mask))
                    inlier_ratio = inliers / max(len(matches), 1)
                    relative_pose[:3, :3] = rotation
                    relative_pose[:3, 3] = (translation[:, 0] * scale)

            predicted_poses.append(predicted_poses[-1] @ relative_pose)
            runtime_ms = (perf_counter() - start) * 1000.0
            stats.append(
                FrameStats(
                    frame_index=idx,
                    keypoints_prev=len(kp_prev),
                    keypoints_curr=len(kp_curr),
                    raw_matches=getattr(self, "_last_raw_matches", 0),
                    good_matches=len(matches),
                    inliers=inliers,
                    inlier_ratio=inlier_ratio,
                    runtime_ms=runtime_ms,
                    scale_m=scale,
                )
            )

        return VoResult(feature=self.feature, poses=predicted_poses, stats=stats)

    def _match(
        self,
        desc_prev: np.ndarray | None,
        desc_curr: np.ndarray | None,
    ) -> list[cv2.DMatch]:
        self._last_raw_matches = 0
        if desc_prev is None or desc_curr is None:
            return []
        if len(desc_prev) < 2 or len(desc_curr) < 2:
            return []

        matcher = cv2.BFMatcher(self.norm_type)
        knn_matches = matcher.knnMatch(desc_prev, desc_curr, k=2)
        self._last_raw_matches = len(knn_matches)

        good_matches = []
        for pair in knn_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < self.ratio_test * second.distance:
                good_matches.append(first)
        return sorted(good_matches, key=lambda match: match.distance)


def _make_detector(feature: str):
    if feature == "orb":
        return (
            cv2.ORB_create(
                nfeatures=2500,
                scaleFactor=1.2,
                nlevels=8,
                fastThreshold=12,
            ),
            cv2.NORM_HAMMING,
        )
    if feature == "sift":
        return (
            cv2.SIFT_create(
                nfeatures=2500,
                contrastThreshold=0.03,
                edgeThreshold=10,
            ),
            cv2.NORM_L2,
        )
    raise ValueError(f"Unsupported feature '{feature}'. Use orb, sift, or both.")


def _relative_gt_scale(prev_pose: np.ndarray, curr_pose: np.ndarray) -> float:
    return float(np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3]))


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0
