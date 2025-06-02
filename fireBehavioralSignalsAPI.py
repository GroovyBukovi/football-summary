import os
import json
import subprocess
from math import ceil
from pathlib import Path
from pydub import AudioSegment
from moviepy import VideoFileClip

# === CONFIG ===
video_path = "YUVE-ATLETI-highlights.mp4"
output_wav = "YUVE-ATLETI.wav"
segment_duration_ms = 5 * 60 * 1000  # 30 minutes in milliseconds
segments_dir = "segments-YUVE-ATLETI"
final_output_json = "final_features-YUVE-ATLETI.json"
api_script = "send_data_to_api.py"  # from the Behavioral Signals repo

# === STEP 1: Extract audio from MP4 ===
def extract_audio(video_path, wav_path):
    print("🎵 Extracting audio from MP4...")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(wav_path, codec="pcm_s16le")
    print(f"✅ Audio saved to {wav_path}")

# === STEP 2: Split audio into segments ===
def split_audio(wav_path, output_dir):
    print("✂️ Splitting audio...")
    audio = AudioSegment.from_wav(wav_path)
    total_length = len(audio)
    os.makedirs(output_dir, exist_ok=True)

    num_parts = ceil(total_length / segment_duration_ms)
    for i in range(num_parts):
        start = i * segment_duration_ms
        end = min(start + segment_duration_ms, total_length)
        segment = audio[start:end]
        segment_path = os.path.join(output_dir, f"part_{i}.wav")
        segment.export(segment_path, format="wav")
        print(f"  → Saved: {segment_path}")
    return num_parts

# === STEP 3: Send to API & collect responses ===
def process_segments_with_api(num_parts):
    print("📡 Sending segments to API...")
    for i in range(num_parts):
        segment_path = os.path.join(segments_dir, f"part_{i}.wav")
        subprocess.run(["python3", api_script, "-i", segment_path], check=True)

# === STEP 4: Merge all *_features.json ===
"""def merge_jsons(num_parts):
    print("🧩 Merging responses...")
    merged = []
    offset_seconds = 0.0

    for i in range(num_parts):
        segment_file = os.path.join(segments_dir, f"part_{i}.wav")
        json_file = f"{segment_file[:-4]}_features.json"

        if not os.path.exists(json_file):
            print(f"⚠️ Warning: Missing {json_file}, skipping.")
            continue

        with open(json_file) as f:
            data = json.load(f)

        # Adjust timestamps
        for entry in data:
            entry["startTime"] = str(float(entry["startTime"]) + offset_seconds)
            entry["endTime"] = str(float(entry["endTime"]) + offset_seconds)
            merged.append(entry)

        # Update offset
        audio = AudioSegment.from_wav(segment_file)
        offset_seconds += len(audio) / 1000.0

    with open(final_output_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✅ Final merged JSON saved to {final_output_json}")"""

global_id = 0  # unique ID counter

def merge_jsons(num_parts):
    print("🧩 Merging responses...")
    merged = []
    offset_seconds = 0.0
    global_id = 0

    for i in range(num_parts):
        segment_file = os.path.join(segments_dir, f"part_{i}.wav")
        json_file = f"{segment_file[:-4]}_features.json"

        if not os.path.exists(json_file):
            print(f"⚠️ Warning: Missing {json_file}, skipping.")
            continue

        with open(json_file) as f:
            data = json.load(f)

        for entry in data:
            try:
                entry["startTime"] = str(float(entry["startTime"]) + offset_seconds)
                entry["endTime"] = str(float(entry["endTime"]) + offset_seconds)
                entry["id"] = str(global_id)
                merged.append(entry)
                global_id += 1
            except (KeyError, ValueError, TypeError) as e:
                print(f"⚠️ Skipping malformed entry: {entry} — Error: {e}")

        audio = AudioSegment.from_wav(segment_file)
        offset_seconds += len(audio) / 1000.0

    with open(final_output_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"✅ Final merged JSON saved to {final_output_json}")


# Step 1: Extract audio from video
extract_audio(video_path, output_wav)

# Step 2: Split extracted audio into segments
num_parts = split_audio(output_wav, segments_dir)

# Step 3: Send each segment to the API and save results
process_segments_with_api(num_parts)

from pathlib import Path

segments_dir = Path(segments_dir)
valid_indices = [
    int(f.stem.split("_")[1])
    for f in segments_dir.glob("part_*_features.json")
    if (segments_dir / f"part_{f.stem.split('_')[1]}.wav").exists()
]

num_parts = max(valid_indices) + 1  # assuming files are sequential
merge_jsons(num_parts)

