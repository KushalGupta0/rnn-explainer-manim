#!/usr/bin/env python3
"""
Cross-Platform Video Rendering Pipeline for RNN Explainer Video.
Supports Windows, macOS, and Linux across all quality tiers.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

QUALITY_MAP = {
    "low": {"flag": "-ql", "res": "480p15", "desc": "Preview (480p, 15 fps)"},
    "480p": {"flag": "-ql", "res": "480p15", "desc": "Preview (480p, 15 fps)"},
    "medium": {"flag": "-qm", "res": "720p30", "desc": "Standard (720p, 30 fps)"},
    "720p": {"flag": "-qm", "res": "720p30", "desc": "Standard (720p, 30 fps)"},
    "high": {"flag": "-qh", "res": "1080p60", "desc": "Full HD (1080p, 60 fps)"},
    "1080p": {"flag": "-qh", "res": "1080p60", "desc": "Full HD (1080p, 60 fps)"},
    "4k": {"flag": "-qk", "res": "2160p60", "desc": "Ultra HD 4K (2160p, 60 fps)"},
    "2160p": {"flag": "-qk", "res": "2160p60", "desc": "Ultra HD 4K (2160p, 60 fps)"},
}


def find_python() -> str:
    """Finds the virtual environment Python if available, else current sys.executable."""
    root = Path(__file__).resolve().parent
    venv_win = root / ".venv" / "Scripts" / "python.exe"
    venv_nix = root / ".venv" / "bin" / "python"

    if venv_win.exists():
        return str(venv_win)
    if venv_nix.exists():
        return str(venv_nix)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the full unified RNN Explainer video or specific scenes."
    )
    parser.add_argument(
        "--quality",
        "-q",
        choices=["low", "medium", "high", "4k", "480p", "720p", "1080p", "2160p"],
        default="low",
        help="Target video quality tier (default: low)",
    )
    parser.add_argument(
        "--scene",
        "-s",
        default="RNNExplainer",
        help="Scene class to render from rnn_explainer.py (default: RNNExplainer for full video)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Destination directory for completed video (default: output)",
    )
    args = parser.parse_args()

    q_info = QUALITY_MAP[args.quality.lower()]
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    py_exe = find_python()
    script_path = root_dir / "rnn_explainer.py"

    if not script_path.exists():
        print(f"[Error] Script not found: {script_path}")
        sys.exit(1)

    print("=" * 70)
    print("RNN EXPLAINER VIDEO — RENDERING PIPELINE")
    print("=" * 70)
    print(f"Scene:       {args.scene}")
    print(f"Quality:     {q_info['desc']} [{q_info['flag']}]")
    print(f"Python:      {py_exe}")
    print(f"Source:      {script_path.name}")
    print(f"Output Dir:  {output_dir}")
    print("=" * 70)

    # Set layout check environment variable
    env = os.environ.copy()
    env["MANIM_LAYOUT_CHECK"] = "1"

    cmd = [
        py_exe,
        "-m",
        "manim",
        q_info["flag"],
        str(script_path),
        args.scene,
    ]

    print(f"\n[Running] {' '.join(cmd)}\n")
    ret = subprocess.call(cmd, cwd=str(root_dir), env=env)

    if ret != 0:
        print(f"\n[Error] Manim render exited with code {ret}")
        sys.exit(ret)

    # Locate generated video in Manim's media structure
    media_pattern = root_dir / "media" / "videos" / "rnn_explainer" / q_info["res"] / f"{args.scene}.mp4"

    if media_pattern.exists():
        final_name = f"{args.scene.lower()}_{args.quality.lower()}.mp4"
        dest_file = output_dir / final_name
        shutil.copy2(media_pattern, dest_file)
        file_size_mb = dest_file.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print("RENDER COMPLETE!")
        print("=" * 70)
        print(f"Output Video: {dest_file}")
        print(f"File Size:    {file_size_mb:.2f} MB")
        print("=" * 70)
    else:
        print(f"\n[Warning] Could not find rendered file at expected path: {media_pattern}")


if __name__ == "__main__":
    main()
