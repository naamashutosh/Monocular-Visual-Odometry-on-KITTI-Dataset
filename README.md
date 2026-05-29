# Monocular Visual Odometry on KITTI Sequence 00

Classical monocular visual odometry project using KITTI Sequence 00 frames. The pipeline detects ORB or SIFT features, matches frame pairs, estimates the essential matrix with RANSAC, recovers relative camera pose, scales monocular translation for evaluation, and benchmarks trajectory error with Absolute Trajectory Error (ATE). It also exports KITTI-format trajectories that can be evaluated with the `evo` toolkit.

Project period: Dec 2025 - Feb 2026

## What This Builds

- Monocular VO pipeline on a mini KITTI Sequence 00 subset.
- ORB vs. SIFT comparison for runtime, match count, and inlier ratio.
- ATE evaluation against KITTI ground truth poses.
- Trajectory plots and machine-readable result summaries.
- Theory-heavy Q&A notes for interviews, viva, reports, and documentation.

## Project Layout

```text
.
├── data/                         # downloaded mini dataset, git-ignored
├── docs/
│   ├── project_report.md
│   └── theory_qna.md
├── results/                      # generated plots/metrics, git-ignored
├── scripts/
│   ├── download_kitti_mini.py
│   └── run_vo.py
├── src/
│   └── vo/
│       ├── dataset.py
│       ├── metrics.py
│       └── pipeline.py
└── requirements.txt
```

## Setup

Use Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If system Python is not registered on Windows, use the Python executable provided by your IDE/runtime to create the virtual environment.

## Download A Mini KITTI Subset

The downloader pulls a small Sequence 00 image slice from the Hugging Face dataset viewer API and the official KITTI odometry pose archive. By default it downloads 30 frames, enough for a fast local sanity run.

```powershell
python scripts/download_kitti_mini.py --frames 30
```

Generated structure:

```text
data/kitti_mini/
├── poses/00.txt
└── sequences/00/
    ├── calib.txt
    ├── image_0/000000.png ...
    └── times.txt
```

## Run The VO Pipeline

Run ORB and SIFT back to back:

```powershell
python scripts/run_vo.py --feature both --max-frames 30
```

Run only ORB:

```powershell
python scripts/run_vo.py --feature orb --max-frames 30
```

Outputs are written to `results/`:

- `comparison.csv`
- `summary.json`
- `trajectory_orb.png`
- `trajectory_sift.png`
- `pred_orb_kitti.txt`
- `pred_sift_kitti.txt`
- `gt_kitti.txt`

## evo Evaluation

The runner computes ATE internally. It also writes KITTI-format files so you can run evo directly:

```powershell
evo_ape kitti results/gt_kitti.txt results/pred_orb_kitti.txt --align --correct_scale --plot
```

If the `evo_ape` console script is unavailable but `evo` is installed in the environment:

```powershell
python -m evo.main_ape kitti results/gt_kitti.txt results/pred_orb_kitti.txt --align --correct_scale --plot
```

## Important Monocular VO Note

A monocular camera can recover rotation and translation direction from two views, but not metric translation scale. This project uses KITTI ground-truth frame-to-frame displacement to scale the recovered translation during evaluation. That keeps the focus on classical geometry, feature quality, and trajectory drift while making the estimated path comparable to ground truth.

## References

- KITTI Vision Benchmark Suite: https://www.cvlibs.net/datasets/kitti/
- KITTI odometry poses archive: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
- Hugging Face KITTI subset used by the mini downloader: https://huggingface.co/datasets/UniflexAI/kitti
- evo toolkit: https://github.com/MichaelGrupp/evo
