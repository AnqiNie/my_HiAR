#!/usr/bin/env python3
"""Merge N sets of videos side-by-side with labels, output H.264 mp4."""

import os
import glob
import argparse
import subprocess
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def add_label(frame, text, font_scale=1.0, thickness=2):
    """Add a centered label at the top of the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = th + 16
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + baseline + 8),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    return frame


def merge_one(video_paths, labels, output_path, overwrite=False):
    """Merge N videos side-by-side with labels."""
    if os.path.exists(output_path) and not overwrite:
        return True, os.path.basename(output_path), "skipped"

    try:
        caps = [cv2.VideoCapture(p) for p in video_paths]

        fps = caps[0].get(cv2.CAP_PROP_FPS)
        w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = len(caps)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w * n}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        while True:
            frames = []
            all_ok = True
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)
            if not all_ok:
                break

            for i, (frame, label) in enumerate(zip(frames, labels)):
                frames[i] = add_label(frame, label)

            merged = np.concatenate(frames, axis=1)
            proc.stdin.write(merged.tobytes())

        proc.stdin.close()
        proc.wait(timeout=120)
        for cap in caps:
            cap.release()

        if proc.returncode != 0:
            err = proc.stderr.read().decode()[-300:]
            return False, os.path.basename(output_path), err
        return True, os.path.basename(output_path), None

    except Exception as e:
        return False, os.path.basename(output_path), str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs="+", required=True,
                        help="Video directories in display order (left to right)")
    parser.add_argument("--labels", type=str, nargs="+", required=True,
                        help="Labels for each directory (same order)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    assert len(args.dirs) == len(args.labels), "Number of dirs must match number of labels"
    assert len(args.dirs) >= 2, "Need at least 2 directories to merge"

    os.makedirs(args.output_dir, exist_ok=True)

    # Use first directory as reference for filenames
    ref_videos = sorted(glob.glob(os.path.join(args.dirs[0], "*.mp4")))
    print(f"Found {len(ref_videos)} videos in {args.dirs[0]}")

    groups = []
    for ref_path in ref_videos:
        name = os.path.basename(ref_path)
        paths = [ref_path]
        all_exist = True
        for d in args.dirs[1:]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"  Warning: no match for {name} in {d}")
                all_exist = False
                break
        if all_exist:
            output_path = os.path.join(args.output_dir, name)
            groups.append((paths, output_path))

    print(f"Merging {len(groups)} video groups ({len(args.dirs)}-way)...")

    success, skip, fail = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(merge_one, paths, args.labels, out, args.overwrite): out
            for paths, out in groups
        }
        for future in as_completed(futures):
            ok, name, msg = future.result()
            if ok:
                if msg == "skipped":
                    skip += 1
                else:
                    success += 1
                    print(f"  Merged: {name}")
            else:
                fail += 1
                print(f"  FAILED: {name}: {msg}")

    print(f"\nDone: {success} merged, {skip} skipped, {fail} failed")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
