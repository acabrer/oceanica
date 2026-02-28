#!/usr/bin/env python3
"""
================================================================================
ECHOGRAM VIDEO MONTAGE
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya, Departament de Cultura

Combines an existing echogram PNG and audio WAV into an MP4 video
with a moving white playhead line. Pure ffmpeg montage — no data processing.

Requires: ffmpeg (brew install ffmpeg)

Usage:
    python src/visualization/create_echogram_video.py [YYYYMMDD]

Author: Oceànica (Alex Cabrer & Joan Sala)
================================================================================
"""

import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DATA, OUTPUT_VIZ


def get_wav_duration(wav_path):
    """Get WAV duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(wav_path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def detect_plot_bounds(png_path):
    """
    Detect the plot area within the echogram PNG by scanning for
    non-white pixel columns. Returns (left, right, top, bottom) in pixels.
    """
    from PIL import Image
    import numpy as np

    img = np.array(Image.open(png_path).convert('RGB'))
    h, w, _ = img.shape

    # A pixel is "white-ish" if all channels > 245
    is_white = np.all(img > 245, axis=2)

    # Find leftmost column that has a substantial non-white region
    col_nonwhite = np.sum(~is_white, axis=0)  # non-white pixel count per column
    threshold = h * 0.3  # at least 30% of column height must be non-white

    cols_with_data = np.where(col_nonwhite > threshold)[0]
    left = int(cols_with_data[0])

    # Find the first large gap (>20px) to separate echogram from colorbar
    gaps = np.diff(cols_with_data)
    big_gaps = np.where(gaps > 20)[0]
    if len(big_gaps) > 0:
        right = int(cols_with_data[big_gaps[0]])  # stop before the gap
    else:
        right = int(cols_with_data[-1])

    # Find top/bottom rows with substantial non-white content
    row_nonwhite = np.sum(~is_white, axis=1)
    row_threshold = w * 0.3
    rows_with_data = np.where(row_nonwhite > row_threshold)[0]
    top = int(rows_with_data[0])
    bottom = int(rows_with_data[-1])

    return left, right, top, bottom, w, h


def create_video(date_str):
    """Create MP4 montage: echogram PNG + playhead + audio WAV."""
    png_path = OUTPUT_VIZ / f"echogram_24h_{date_str}.png"
    wav_path = OUTPUT_DATA / f"echogram_audio_{date_str}.wav"
    mp4_path = OUTPUT_VIZ / f"echogram_sonogram_{date_str}.mp4"

    # Validate inputs
    if not png_path.exists():
        print(f"ERROR: Echogram not found: {png_path}")
        print(f"  Run: python src/visualization/create_echogram_24h_validation.py {date_str}")
        return
    if not wav_path.exists():
        print(f"ERROR: Audio not found: {wav_path}")
        print(f"  Run: python src/extraction/echogram_to_audio.py {date_str}")
        return
    if not shutil.which('ffmpeg'):
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        return

    print("=" * 70)
    print("ECHOGRAM VIDEO MONTAGE")
    print("=" * 70)
    print(f"Date:  {date_str}")
    print(f"Image: {png_path.name}")
    print(f"Audio: {wav_path.name}")

    # Get audio duration
    duration = get_wav_duration(wav_path)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Detect plot area in the PNG
    print("\nDetecting plot area...")
    left, right, top, bottom, img_w, img_h = detect_plot_bounds(png_path)
    plot_width = right - left
    plot_height = bottom - top
    print(f"  Image: {img_w}x{img_h} px")
    print(f"  Plot area: x={left}–{right}, y={top}–{bottom} ({plot_width}x{plot_height} px)")

    # Ensure even dimensions (required by libx264)
    out_w = img_w if img_w % 2 == 0 else img_w - 1
    out_h = img_h if img_h % 2 == 0 else img_h - 1

    # Load base image with PIL
    print("\nLoading base image...")
    base_img = Image.open(png_path).convert('RGB')
    if (out_w, out_h) != (base_img.width, base_img.height):
        base_img = base_img.resize((out_w, out_h), Image.LANCZOS)

    # Pre-build playhead line image (6px wide, white)
    line_img = Image.new('RGB', (6, plot_height), (255, 255, 255))

    # Frame generation: PIL draws each frame, pipes raw RGB to ffmpeg
    fps = 30
    total_frames = int(fps * duration)
    print(f"  Frames: {total_frames} ({fps} fps x {duration:.1f}s)")

    # Start ffmpeg subprocess reading raw frames from stdin
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-i', str(wav_path),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(mp4_path)
    ]

    print(f"\nRendering video...")
    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    try:
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            playhead_x = int(left + progress * plot_width)

            frame = base_img.copy()

            # Dark overlay on unplayed region (right of playhead)
            unplayed_width = right - playhead_x
            if unplayed_width > 0:
                region = frame.crop((playhead_x, top, right, bottom))
                dark = Image.new('RGB', region.size, (0, 0, 0))
                region = Image.blend(region, dark, 0.4)
                frame.paste(region, (playhead_x, top))

            # White playhead line (6px)
            if playhead_x + 6 <= out_w:
                frame.paste(line_img, (playhead_x, top))

            proc.stdin.write(frame.tobytes())

            # Progress every 10%
            if frame_num % (total_frames // 10) == 0:
                print(f"  {100 * frame_num // total_frames}%", end="", flush=True)
        print(f"  100%")
    except BrokenPipeError:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"\n  ffmpeg error: {stderr[-500:]}")
        return
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"  ffmpeg error: {stderr[-500:]}")
        return

    size_mb = mp4_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {mp4_path} ({size_mb:.1f} MB)")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Output: {mp4_path}")
    print(f"  Open with: open {mp4_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Create MP4 video: echogram + playhead + audio."
    )
    parser.add_argument(
        'date', nargs='?', default='20110126',
        help='Date in YYYYMMDD format (default: 20110126)'
    )
    args = parser.parse_args()

    create_video(args.date)
