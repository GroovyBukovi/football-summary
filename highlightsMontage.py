import json
import os
import subprocess
from pathlib import Path

# === CONFIG ===
VIDEO_PATH = "FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.mp4"
HIGHLIGHTS_JSON = "highlights-PORT-SPAIN.json"
CLIPS_DIR = Path("highlight_clips")
FINAL_OUTPUT = "montage_final.mp4"
CLIP_LIST_FILE = CLIPS_DIR / "clip_list.txt"
PADDING = 10  # seconds to add before and after each segment

# === Load and pad segments ===
with open(HIGHLIGHTS_JSON, "r") as f:
    highlights = json.load(f)

# Apply padding
padded_segments = []
for seg in highlights:
    start = max(0, float(seg["startTime"]) - PADDING)
    end = float(seg["endTime"]) + PADDING
    padded_segments.append((start, end))

# Sort and merge overlapping
padded_segments.sort()
merged_segments = []

for start, end in padded_segments:
    if not merged_segments:
        merged_segments.append([start, end])
    else:
        prev_start, prev_end = merged_segments[-1]
        if start <= prev_end:  # Overlap
            merged_segments[-1][1] = max(prev_end, end)
        else:
            merged_segments.append([start, end])

# === Create clips directory ===
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# === Extract merged segments ===
clip_paths = []

for idx, (start, end) in enumerate(merged_segments):
    duration = end - start
    clip_filename = f"clip_{idx:03d}.mp4"
    clip_output = CLIPS_DIR / clip_filename
    clip_paths.append(clip_filename)

    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        str(clip_output)
    ]
    subprocess.run(cmd, check=True)
    print(f"🎞️  Saved: {clip_output}")

# === Write list file for FFmpeg concat ===
with open(CLIP_LIST_FILE, "w") as f:
    for filename in clip_paths:
        f.write(f"file '{filename}'\n")

# === Concatenate all clips ===
concat_cmd = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", CLIP_LIST_FILE.name,
    "-c", "copy",
    f"../{FINAL_OUTPUT}"
]

subprocess.run(concat_cmd, check=True, cwd=CLIPS_DIR)
print(f"✅ Final highlight montage saved to: {FINAL_OUTPUT}")