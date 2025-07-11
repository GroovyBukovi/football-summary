# -*- coding: utf-8 -*-
"""LLM football.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HsepBbbIIUtVEwxfiR_tT9q6VhZ0e1OT
"""

from google.colab import drive
drive.mount('/content/drive')

video_path = "/content/drive/My Drive/matches/FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.mp4"

"""!pip install -U openai-whisper
!pip install -U transformers huggingface_hub"""
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
video_path = "/content/drive/My Drive/matches/FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.mp4"
transcript_path = "/content/drive/My Drive/matches/FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.txt"

# Install Whisper
!pip install -U openai-whisper
import whisper

# Load Whisper model (use "base", "small", "medium", or "large")
model = whisper.load_model("base")  # or "small", "medium", "large"

# Transcribe the audio from the MP4 file
result = model.transcribe(video_path, verbose=True)

# Save transcript to Google Drive
with open(transcript_path, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
        f.write(f"[{start:.2f} - {end:.2f}] {text}\n")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set transcript and output paths in Drive
transcript_path = "/content/drive/My Drive/matches/FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.txt"
highlights_path = "/content/drive/My Drive/matches/match_highlights.txt"

# Install and import required libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login

# Hugging Face API login (keep secret in production)
login("asjcecneccnicneinenccke")

# Load model & tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load transcript from Google Drive
with open(transcript_path, "r", encoding="utf-8") as f:
    transcript = f.read()

print(len(transcript))

# Split transcript into chunks
def chunk_text(text, max_tokens=800):
    lines = text.splitlines()
    chunks, chunk = [], ""
    for line in lines:
        if len(tokenizer(chunk + line).input_ids) < max_tokens:
            chunk += line + "\n"
        else:
            chunks.append(chunk.strip())
            chunk = line + "\n"
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = chunk_text(transcript)

# Prompt template
def make_prompt(chunk):
    return f"""### Instruction:
You are a professional football match analyst. Your task is to extract only the most important match events from the transcript below.

Only include:
- Goals
- Red cards
- Penalty kicks or missed chances for goal

For each event:
- Include the timestamp exactly as shown (e.g., `[12.35 - 13.02]`)
- Use a short, clear description (1–2 sentences)
- Ignore irrelevant or minor commentary
- Understand the context between consecutive texts because a highlight is something that builds up, it does not just occur instantly

Only list real match highlights — do not hallucinate or guess events.
Out of all the chunks of the match you work on, you should extract as highlights only the top 6% most important. This means, that on a regular 90 minute match, around 5,5 minutes should reflect the highlights.
### Transcript:
{chunk}

### Highlights:"""

# Generate highlights
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

all_highlights = []
for i, chunk in enumerate(chunks):
    prompt = make_prompt(chunk)
    num_tokens = len(tokenizer(prompt).input_ids)
    print(f"Prompt tokens (chunk {i+1}): {num_tokens}")

    print(f"Processing chunk {i+1}/{len(chunks)}...")
    out = pipe(prompt)[0]['generated_text']
    print(out)
    highlights = out.split("### Highlights:")[-1].strip()
    all_highlights.append(highlights)

# Save highlights to Google Drive
with open(highlights_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_highlights))

import re
import json

# Path to the input and output files
input_txt_path = "/content/drive/My Drive/matches/match_highlights.txt"
output_json_path = "/content/drive/My Drive/matches/highlights-PORTUGAL-SPAIN.json"

# Helper to convert seconds to hh:mm:ss
def seconds_to_hhmmss(seconds):
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:05.2f}"

# Load and parse the text file
highlights = []

pattern = re.compile(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]:?\s*(.*)")

with open(input_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            start_sec, end_sec, desc = match.groups()
            highlights.append({
                "startTime": seconds_to_hhmmss(start_sec),
                "endTime": seconds_to_hhmmss(end_sec),
                "description": desc.strip()
            })

# Save to JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(highlights, f, indent=2)

print(f"✅ Converted {len(highlights)} highlights to JSON:")
print(f"📁 Saved to: {output_json_path}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# === CONFIG ===
input_json_path = "/content/drive/My Drive/matches/highlights-PORTUGAL-SPAIN.json"
output_json_path = "/content/drive/My Drive/matches/highlights-PORTUGAL-SPAIN-final.json"

# Install and import dependencies
!pip install -q transformers huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch
import json
import re

# Hugging Face login
login("shsjsdhjhcjehjhjehjhjs")

# Load model & tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Load highlights from JSON
with open(input_json_path, "r", encoding="utf-8") as f:
    highlights = json.load(f)

print(f"📥 Loaded {len(highlights)} highlights.")

# Format each highlight for the prompt
def format_highlight(h):
    return f"[{h['startTime']} - {h['endTime']}]: {h['description']}"

# Prompt template
def make_prompt(text_block):
    return f"""### Instruction:
You are a professional football video analyst. Your task is to extract only the top 6% most important moments from this match highlight list.

You may only select:
- Match-defining goals (equalizers, winners, hat-tricks)
- Critical penalty kicks (scored or missed)
- Important red cards
- Game-changing moments (saves, fouls, build-ups)

⚠️ You are allowed to keep highlights that together add up to ~5–6 minutes of match time in a 90-minute match. Choose wisely and do not guess.

### Full list of highlights:
{text_block}

### Final Top 6% Most Important Highlights (keep original format):"""

# Extract timestamps using regex
pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*-\s*(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]:\s*(.+)")

# Drop milliseconds from timestamp
def to_hhmmss(time_str):
    return time_str.split('.')[0] if '.' in time_str else time_str

# Smart split based on token length
def split_highlights_into_chunks(highlights, max_tokens=1500):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for h in highlights:
        text = format_highlight(h)
        tokens = len(tokenizer(text).input_ids)

        if current_token_count + tokens < max_tokens:
            current_chunk.append(h)
            current_token_count += tokens
        else:
            chunks.append(current_chunk)
            current_chunk = [h]
            current_token_count = tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Split into token-friendly chunks
highlight_chunks = split_highlights_into_chunks(highlights, max_tokens=1500)
print(f"🔧 Split into {len(highlight_chunks)} prompt-friendly chunks.")

# Generate pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)

final_highlights = []

# Process each chunk with the LLM
for i, chunk in enumerate(highlight_chunks, start=1):
    print(f"\n🚀 Processing chunk {i}/{len(highlight_chunks)}...")
    chunk_text = "\n".join(format_highlight(h) for h in chunk)
    prompt = make_prompt(chunk_text)

    out = pipe(prompt)[0]['generated_text']
    extracted = out.split("### Final Top 6% Most Important Highlights (keep original format):")[-1].strip()

    for line in extracted.splitlines():
        match = pattern.match(line.strip())
        if match:
            start, end, desc = match.groups()
            final_highlights.append({
                "startTime": to_hhmmss(start),
                "endTime": to_hhmmss(end),
                "description": desc.strip()
            })

# Save final highlights to Google Drive
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(final_highlights, f, indent=2)

print(f"\n✅ Final top 6% highlights saved to:\n📁 {output_json_path}")

import json
import os
import subprocess
from pathlib import Path

# === CONFIG ===
VIDEO_PATH = "/content/drive/My Drive/matches/FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.mp4"
HIGHLIGHTS_JSON = "/content/drive/My Drive/matches/highlights-PORTUGAL-SPAIN-final.json"
CLIPS_DIR = Path("/content/highlight_clips")
FINAL_OUTPUT = "/content/drive/My Drive/matches/llm_final.mp4"
CLIP_LIST_FILE = CLIPS_DIR / "clip_list.txt"
PADDING = 10  # seconds to add before and after each segment

# === Utility: convert hh:mm:ss to seconds
def hhmmss_to_seconds(hhmmss):
    h, m, s = map(int, hhmmss.split(":"))
    return h * 3600 + m * 60 + s

# === Load and pad segments ===
with open(HIGHLIGHTS_JSON, "r") as f:
    highlights = json.load(f)

padded_segments = []
for seg in highlights:
    start = max(0, hhmmss_to_seconds(seg["startTime"]) - PADDING)
    end = hhmmss_to_seconds(seg["endTime"]) + PADDING
    padded_segments.append((start, end))

# Sort and merge overlapping segments
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

# === Extract merged segments using ffmpeg ===
clip_paths = []

for idx, (start, end) in enumerate(merged_segments):
    duration = end - start
    clip_filename = f"clip_{idx:03d}.mp4"
    clip_output = CLIPS_DIR / clip_filename
    clip_paths.append(clip_output)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        str(clip_output)
    ]
    subprocess.run(cmd, check=True)
    print(f"🎞️  Saved: {clip_output}")

# === Write list of clips to concat file ===
with open(CLIP_LIST_FILE, "w") as f:
    for clip in clip_paths:
        f.write(f"file '{clip.name}'\n")

# === Concatenate all clips into final video ===
concat_cmd = [
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", str(CLIP_LIST_FILE),
    "-c", "copy",
    str(FINAL_OUTPUT)
]

subprocess.run(concat_cmd, check=True, cwd=CLIPS_DIR)
print(f"\n✅ Final highlight montage saved to: {FINAL_OUTPUT}")