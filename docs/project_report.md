# Project Report: Monocular Visual Odometry on KITTI Sequence 00

## Abstract

This project implements a classical monocular visual odometry pipeline for KITTI Sequence 00. The system uses ORB and SIFT keypoint pipelines, brute-force descriptor matching with a ratio test, RANSAC-based essential matrix estimation, and relative pose recovery. The estimated trajectory is benchmarked against KITTI ground truth using Absolute Trajectory Error (ATE), with exports compatible with the evo toolkit.

## Objective

The goal is to understand the geometry and engineering trade-offs behind real-time monocular odometry, especially for resource-constrained UAV deployment. The project intentionally uses classical computer vision instead of deep learning so that every major failure mode can be traced to feature quality, calibration, epipolar geometry, outlier rejection, scale, or drift.

## Dataset

The full KITTI odometry grayscale archive is large, so the local project uses a mini Sequence 00 subset. Images are fetched from the Hugging Face KITTI dataset viewer API, while ground-truth poses are fetched from the official KITTI odometry pose archive. The downloader saves the data in the standard KITTI odometry structure.

Sequence 00 is useful because it is a real urban driving sequence with forward motion, turns, texture changes, and loop-like behavior. Ground truth is available for KITTI sequences 00-10, making it suitable for local benchmarking.

## Pipeline

1. Load grayscale monocular frames from `image_0`.
2. Read camera intrinsics from `calib.txt`.
3. Detect and describe features with ORB or SIFT.
4. Match descriptors using a brute-force matcher.
5. Filter matches with Lowe's ratio test.
6. Estimate the essential matrix with RANSAC.
7. Recover relative rotation and translation direction.
8. Apply frame-to-frame ground-truth scale for metric evaluation.
9. Compose relative poses into a global trajectory.
10. Compare predicted and ground-truth trajectories with ATE.

## Mathematical Core

For a calibrated camera, matching points between two frames satisfy the epipolar constraint:

```text
x2.T E x1 = 0
```

where `x1` and `x2` are normalized image coordinates and `E` is the essential matrix. The essential matrix can be decomposed into relative rotation `R` and translation direction `t`:

```text
E = [t]_x R
```

Because the camera is monocular, `t` is known only up to scale. This implementation uses ground-truth relative displacement to recover metric scale for evaluation. A production monocular system would need another source of scale, such as stereo, IMU, known object sizes, wheel odometry, learned depth, or map constraints.

## ORB vs. SIFT Trade-Off

ORB is fast and uses binary descriptors, making it suitable for real-time CPU pipelines and UAV-style compute budgets. It is less robust than SIFT under stronger scale and illumination variation, but usually offers a better speed/energy trade-off.

SIFT uses floating-point descriptors and is generally more distinctive and stable across scale and illumination changes. The cost is higher runtime and heavier matching. In a real-time UAV pipeline, SIFT may be useful for lower-rate keyframes, initialization, or offline mapping, while ORB is usually better for high-rate tracking.

## Evaluation

The project reports:

- ATE RMSE in meters.
- Mean, median, and max trajectory error.
- Average runtime per frame pair.
- Average good matches after ratio filtering.
- Average RANSAC/recoverPose inlier ratio.

ATE is computed after similarity alignment by default. This is common for monocular trajectory evaluation because monocular estimates have scale ambiguity.

## Limitations

- No bundle adjustment.
- No loop closure.
- No keyframe map.
- No relocalization.
- Monocular scale is supplied from ground truth for evaluation.
- Short mini subsets are useful for fast local validation but not for final benchmark claims.

## Improvements

- Add keyframe selection and local bundle adjustment.
- Fuse IMU to estimate scale and improve rotation.
- Use optical flow tracking between feature detection steps.
- Add loop closure with bag-of-visual-words.
- Compare against stereo VO using KITTI `image_0` and `image_1`.
- Evaluate longer KITTI subsequences and multiple sequences.
