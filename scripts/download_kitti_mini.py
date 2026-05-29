from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "kitti_mini"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
POSES_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a tiny KITTI Sequence 00 slice.")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to save.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Dataset output root.")
    parser.add_argument("--chunk-size", type=int, default=10, help="Rows requested per API call.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing mini dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least 2")

    output = args.output.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)

    image_dir = output / "sequences" / "00" / "image_0"
    image_dir.mkdir(parents=True, exist_ok=True)
    (output / "poses").mkdir(parents=True, exist_ok=True)

    rows = _download_rows(args.frames, args.chunk_size)
    if len(rows) < args.frames:
        raise RuntimeError(f"Only received {len(rows)} rows; expected {args.frames}")

    _write_images(rows, image_dir)
    _write_calibration(rows[0], output / "sequences" / "00" / "calib.txt")
    _write_times(args.frames, output / "sequences" / "00" / "times.txt")
    _write_poses(args.frames, output / "poses" / "00.txt")

    print(f"Downloaded {args.frames} frames to {output}")
    print(f"Images: {image_dir}")


def _download_rows(frames: int, chunk_size: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < frames:
        length = min(chunk_size, frames - len(rows))
        url = (
            f"{HF_ROWS_URL}?dataset=UniflexAI%2Fkitti&config=default"
            f"&split=train&offset={offset}&length={length}"
        )
        print(f"Fetching rows {offset}..{offset + length - 1}")
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        for item in payload.get("rows", []):
            row = item["row"]
            if row.get("sequence_id") == "00":
                rows.append(row)
        offset += length
    return rows[:frames]


def _write_images(rows: list[dict], image_dir: Path) -> None:
    for index, row in enumerate(rows):
        image_bytes = base64.b64decode(row["left_image"])
        filename = f"{index:06d}.png"
        (image_dir / filename).write_bytes(image_bytes)


def _write_calibration(row: dict, path: Path) -> None:
    fx = float(row["fx"])
    fy = float(row["fy"])
    cx = float(row["cx"])
    cy = float(row["cy"])
    baseline = float(row["baseline"])
    p0 = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    p1 = [fx, 0.0, cx, -fx * baseline, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    body = "P0: " + " ".join(f"{value:.12e}" for value in p0) + "\n"
    body += "P1: " + " ".join(f"{value:.12e}" for value in p1) + "\n"
    path.write_text(body, encoding="utf-8")


def _write_times(frames: int, path: Path) -> None:
    lines = [f"{index * 0.1:.6f}" for index in range(frames)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_poses(frames: int, path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "data_odometry_poses.zip"
        print("Fetching KITTI ground-truth poses")
        urllib.request.urlretrieve(POSES_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            member = "dataset/poses/00.txt"
            lines = archive.read(member).decode("utf-8").splitlines()[:frames]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise
