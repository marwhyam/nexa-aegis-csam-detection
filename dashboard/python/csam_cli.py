#!/usr/bin/env python3
"""
csam_cli.py
Simple CLI wrapper to run your csam_inference pipeline on a list of image paths.
Usage:
    python csam_cli.py --out results.json /path/to/img1.jpg /path/to/img2.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure python path includes current folder so we can import csam_inference
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import your functions from csam_inference.py
# Ensure csam_inference exposes: load_nsfw_model, load_face_model, load_age_model, process_image, process_video
try:
    from csam_inference import (
        load_nsfw_model,
        load_face_model,
        load_age_model,
        process_image,
        process_video,
        VIDEO_FRAME_STRIDE,
        VIDEO_MAX_FRAMES,
    )
except Exception as e:
    print(f"[ERROR] Could not import csam_inference: {e}", file=sys.stderr)
    raise

# Default model paths (relative to this folder)
DEFAULT_NSFW = str(HERE / "models" / "nsfw.pt")
DEFAULT_FACE = str(HERE / "models" / "yolov8l-face.pt")
DEFAULT_AGE  = str(HERE / "models" / "phase2_best_model.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Run CSAM pipeline on images/videos (CLI shim for Electron).")
    p.add_argument("inputs", nargs="+", help="Paths to image or video files")
    p.add_argument("--out", "-o", default=str(HERE / "results.json"), help="Output JSON file")
    p.add_argument("--nsfw", default=DEFAULT_NSFW, help="NSFW model path")
    p.add_argument("--face", default=DEFAULT_FACE, help="Face model path")
    p.add_argument("--age", default=DEFAULT_AGE, help="Age model path")
    p.add_argument("--video-frame-stride", type=int, default=VIDEO_FRAME_STRIDE, help="Process every Nth frame")
    p.add_argument("--video-max-frames", type=int, default=VIDEO_MAX_FRAMES if VIDEO_MAX_FRAMES is not None else -1,
                   help="Max frames per video (-1 for all)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load models (prints to stdout/stderr so Electron can show logs)
    nsfw_model = None
    try:
        nsfw_model = load_nsfw_model(args.nsfw)
    except Exception as e:
        print(f"[WARN] NSFW model could not be loaded: {e}")

    face_model = None
    try:
        face_model = load_face_model(args.face)
    except Exception as e:
        print(f"[WARN] Face model could not be loaded: {e}")

    try:
        age_model = load_age_model(args.age)
    except Exception as e:
        print(f"[ERROR] Age model could not be loaded: {e}")
        age_model = None

    # Normalize inputs by extension
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    all_results = []

    max_frames = None if args.video_max_frames == -1 else args.video_max_frames

    for pth in args.inputs:
        ext = Path(pth).suffix.lower()
        try:
            print(f"[INFO] Processing: {pth}")
            if ext in video_exts:
                res = process_video(pth, nsfw_model, face_model, age_model,
                                    frame_stride=args.video_frame_stride,
                                    max_frames=max_frames)
            else:
                res = process_image(pth, nsfw_model, face_model, age_model)
            if res is not None:
                all_results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed processing {pth}: {e}", file=sys.stderr)

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    print(f"[OK] Results saved: {out_path}")
    # Print the path to stdout for the Electron app to capture
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
