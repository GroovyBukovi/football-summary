import os
import json
import numpy as np
import pandas as pd
import librosa
from moviepy import VideoFileClip

# === CONFIGURATION ===
VIDEO_PATH = "PORTUGAL 3-3 SPAIN.mp4"
TEMP_AUDIO_PATH = "temp_audio.wav"
HIGHLIGHTS_JSON = "highlights-PORT-SPAIN.json"
OUTPUT_CSV = "unsupervised_mfcc_frames.csv"

FRAME_DURATION = 1.0  # seconds per frame
N_MFCC = 13
PADDING_SECONDS = 10  # seconds before and after each highlight

# === STEP 1: Extract audio from video ===
print("🎞️ Extracting audio from video...")
video = VideoFileClip(VIDEO_PATH)
audio = video.audio
audio.write_audiofile(TEMP_AUDIO_PATH, codec='pcm_s16le', logger=None)
print("✅ Audio saved to:", TEMP_AUDIO_PATH)

# === STEP 2: Load audio ===
y_full, sr = librosa.load(TEMP_AUDIO_PATH, sr=None)

# === STEP 3: Load highlight segments ===
with open(HIGHLIGHTS_JSON, "r") as f:
    highlights = json.load(f)

# === STEP 4: Extract frame-level MFCCs ===
rows = []

for segment_id, seg in enumerate(highlights):
    seg_start = max(0, float(seg["startTime"]) - PADDING_SECONDS)
    seg_end = float(seg["endTime"]) + PADDING_SECONDS
    happy_sum = float(seg.get("happy_sum", 0.0))
    angry_sum = float(seg.get("angry_sum", 0.0))

    segment_audio = y_full[int(seg_start * sr):int(seg_end * sr)]
    segment_duration = seg_end - seg_start
    num_frames = int(np.ceil(segment_duration / FRAME_DURATION))

    for i in range(num_frames):
        frame_start = seg_start + i * FRAME_DURATION
        frame_end = min(frame_start + FRAME_DURATION, seg_end)

        start_sample = int((frame_start - seg_start) * sr)
        end_sample = int((frame_end - seg_start) * sr)
        frame = segment_audio[start_sample:end_sample]

        if len(frame) == 0:
            continue

        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = mfcc.mean(axis=1)

        row = {
            "segment_id": segment_id,
            "frame_start": round(frame_start, 2),
            "frame_end": round(frame_end, 2),
            "happy_sum": happy_sum,
            "angry_sum": angry_sum,
            "Class": ""
        }
        row.update({f"mfcc_{j+1}": mfcc_mean[j] for j in range(N_MFCC)})
        rows.append(row)

# === STEP 5: Save to CSV ===
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved MFCC dataset to: {OUTPUT_CSV}")

# === STEP 6: Cleanup temporary audio ===
os.remove(TEMP_AUDIO_PATH)
