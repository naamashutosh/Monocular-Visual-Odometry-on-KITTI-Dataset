# Monocular Visual Odometry Theory Q&A

## Core Concepts

### What is visual odometry?

Visual odometry estimates camera motion from image sequences. It tracks visual changes between frames and integrates relative motion over time to form a trajectory.

### How is visual odometry different from SLAM?

Visual odometry estimates motion incrementally. SLAM estimates motion and builds or updates a map while also using map constraints to reduce drift. A VO pipeline can be a front end inside a SLAM system, but SLAM usually adds mapping, loop closure, relocalization, and global optimization.

### Why is monocular VO harder than stereo VO?

A monocular camera observes only bearing, not depth. It can estimate rotation and translation direction between two frames, but not absolute metric translation scale. Stereo VO can triangulate depth from left-right disparity, so scale is directly observable.

### What is the scale ambiguity problem?

If all scene depths and camera translations are multiplied by the same constant, the monocular image projections remain valid. This means monocular two-view geometry cannot know whether the camera moved 0.1 m or 10 m without extra information.

### How does this project handle scale?

The pipeline uses KITTI ground-truth frame-to-frame displacement to scale each recovered translation vector during evaluation. This makes trajectory plots and ATE meaningful while keeping the focus on classical feature and pose estimation.

## Feature Detection

### Why use ORB?

ORB is fast, rotation-aware, and uses binary descriptors. It is practical for real-time systems because Hamming-distance matching is cheap.

### Why use SIFT?

SIFT is more distinctive and robust to scale and illumination changes. It often gives higher-quality matches but costs more CPU time because descriptors are floating point and more expensive to match.

### What is repeatability?

Repeatability measures whether the same physical scene points are detected again under viewpoint, lighting, blur, or scale changes. Good repeatability helps pose estimation because the feature correspondences remain stable across frames.

### Why compare ORB and SIFT?

The comparison exposes a real engineering trade-off: ORB is usually faster, while SIFT is often more robust. For UAV deployment, runtime and power matter as much as raw matching quality.

## Matching

### What is descriptor matching?

Descriptor matching pairs feature descriptors from two frames. For ORB, the project uses Hamming distance. For SIFT, it uses Euclidean distance.

### What is Lowe's ratio test?

For each descriptor, the matcher finds the two nearest candidates. The best match is accepted only if it is much better than the second-best match. This removes ambiguous matches in repetitive textures.

### Why are outliers dangerous?

A small number of incorrect matches can produce a wrong essential matrix, causing a bad rotation or translation direction. Outlier rejection is essential before pose recovery.

## Epipolar Geometry

### What is the essential matrix?

The essential matrix describes the epipolar geometry between two calibrated camera views. It encodes the relative rotation and translation direction between the two camera poses.

### What is the epipolar constraint?

For corresponding normalized image points `x1` and `x2`, the constraint is:

```text
x2.T E x1 = 0
```

Correct matches should satisfy this relation approximately, allowing for image noise and imperfect calibration.

### Why use the essential matrix instead of the fundamental matrix?

The essential matrix assumes known camera intrinsics and works in normalized camera coordinates. KITTI provides calibration, so the essential matrix is the right model for calibrated VO.

### What is RANSAC doing?

RANSAC repeatedly samples small match sets, estimates a model, and counts how many matches agree with it. It keeps the model with the best consensus and rejects outliers.

### What does `recoverPose` return?

It decomposes the essential matrix into relative rotation and translation direction. It also applies a cheirality check, choosing the solution where triangulated points lie in front of the camera.

## Trajectory And Error

### How are relative poses accumulated?

Each recovered relative transform is multiplied into the previous global pose. Small frame-to-frame errors accumulate, which creates drift over time.

### What is drift?

Drift is the gradual deviation between estimated and true trajectory. It is caused by noisy matches, wrong scale, rolling shutter, dynamic objects, calibration errors, and numerical approximation.

### What is ATE?

Absolute Trajectory Error measures the distance between predicted and ground-truth camera positions after alignment. RMSE ATE summarizes the overall trajectory accuracy.

### Why align trajectories before ATE?

Alignment compensates for global coordinate differences. For monocular VO, similarity alignment is especially useful because scale can be ambiguous.

### What is evo?

evo is a common trajectory evaluation toolkit for visual odometry and SLAM. It supports KITTI, TUM, EuRoC-style workflows, trajectory alignment, ATE/RPE metrics, and plotting.

## KITTI Details

### Why use KITTI Sequence 00?

KITTI Sequence 00 is a standard odometry benchmark sequence with real road motion and available ground truth. It is widely used for testing VO and SLAM pipelines.

### Which KITTI camera is used?

This project uses the left grayscale camera, commonly stored as `image_0` in the KITTI odometry dataset.

### What is in `calib.txt`?

The calibration file contains projection matrices. The VO pipeline extracts focal lengths and principal point to build the camera intrinsic matrix.

### What is in `poses/00.txt`?

Each line is a 3 x 4 camera pose matrix for one frame. The first three columns are rotation and the last column is translation.

## UAV Deployment Discussion

### Why is ORB attractive for UAVs?

ORB is lightweight and fast on CPU. UAVs often have limited compute, battery, and thermal headroom, so an efficient feature pipeline matters.

### When might SIFT still be useful?

SIFT can help in challenging texture, scale, or illumination conditions. It may be useful for lower-frequency keyframes, offline reconstruction, or map initialization.

### What extra sensors help monocular VO on UAVs?

An IMU is the most common addition. It improves rotation estimation, helps with scale observability, and supports visual-inertial odometry during fast motion or motion blur.

### What are the biggest failure cases?

Low texture, repetitive patterns, motion blur, dynamic objects, strong exposure changes, pure rotation, very small baseline, and incorrect calibration can all degrade pose estimation.

### Why is pure rotation difficult?

Pure rotation gives little or no translation baseline. Without parallax, translation and depth become poorly constrained.

## Interview-Style Answers

### Explain the project in one minute.

I built a classical monocular visual odometry pipeline on a mini KITTI Sequence 00 subset. It detects ORB or SIFT features, matches descriptors between frames, filters matches with a ratio test, estimates the essential matrix using RANSAC, recovers camera motion, and accumulates poses into a trajectory. Since monocular VO has scale ambiguity, I used KITTI ground-truth displacement for metric scaling during evaluation. I compared ORB and SIFT on runtime, match quality, inlier ratio, and ATE.

### What did you learn from ORB vs. SIFT?

ORB is faster and more suitable for real-time UAV constraints, while SIFT usually provides more distinctive features and can improve robustness. The trade-off is not just accuracy; it is accuracy per millisecond and per watt.

### Why not use deep learning?

The goal was to understand the geometry first. Classical VO makes the role of calibration, matching, RANSAC, pose recovery, scale, and drift very explicit. Deep methods can be powerful, but they often hide these geometric failure modes.

### What would you improve next?

I would add keyframes, local bundle adjustment, IMU fusion for scale, and loop closure. Those changes would move the project from a VO front end toward a more complete SLAM system.
