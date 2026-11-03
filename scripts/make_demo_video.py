"""Make a slowed-down, labeled video from eval episodes, side by side when given several.

    python scripts/make_demo_video.py out.mp4 "label one=path/one.mp4" "label two=path/two.mp4"

Eval videos are written at 80 fps, so a 72-step episode lasts under a second. Each clip holds
its last frame until the longest one ends. Footage is not otherwise changed.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 20
SIZE = 512
BANNER = 44


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def banner(text, path):
    image = Image.new("RGB", (SIZE, BANNER), "#111111")
    draw = ImageDraw.Draw(image)
    size = 22
    while draw.textlength(text, font=font(size)) > SIZE - 16:
        size -= 1
    draw.text((SIZE / 2, BANNER / 2), text, font=font(size), fill="white", anchor="mm")
    image.save(path)


output = sys.argv[1]
clips = [argument.split("=", 1) for argument in sys.argv[2:]]
with tempfile.TemporaryDirectory() as folder:
    inputs, filters, labels = [], [], []
    for i, (label, path) in enumerate(clips):
        picture = Path(folder) / f"{i}.png"
        banner(label, picture)
        inputs += ["-r", str(FPS), "-i", path, "-i", str(picture)]
        filters.append(
            f"[{2 * i}:v]scale={SIZE}:{SIZE}:flags=lanczos,tpad=stop=-1:stop_mode=clone[v{i}];"
            f"[{2 * i + 1}:v][v{i}]vstack[s{i}]"
        )
        labels.append(f"[s{i}]")
    if len(clips) > 1:
        filters.append("".join(labels) + f"hstack=inputs={len(clips)}:shortest=0[out]")
    else:
        filters.append("[s0]null[out]")

    longest = max(
        float(
            subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                 "stream=nb_read_frames", "-of", "csv=p=0", path],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        for _, path in clips
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
         "-t", str(longest / FPS + 1), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output],
        check=True,
    )
print("wrote", output)
