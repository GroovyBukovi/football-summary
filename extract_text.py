import whisper

# Load Whisper model (use "base", "small", "medium", or "large")
model = whisper.load_model("base")  # or "small", "medium", "large"

# Transcribe the audio from the MP4 file
result = model.transcribe("FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.mp4", verbose=True)

# Print or process each segment
with open("FULL MATCH_ Portugal v Spain _ 2018 FIFA World Cup.txt", "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
        f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
