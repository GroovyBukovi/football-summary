import json
from datetime import timedelta
import numpy as np

# === CONFIG ===
INPUT_JSON = "final_features-Liv-MAC.json"
OUTPUT_JSON = "output_segments_with_hhmmss-Liv-MAC.json"

# === Utility: Convert seconds to hh:mm:ss ===
def seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=float(seconds))).split(".")[0]

# === Load data ===
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

sum_all_list = []
strong_list = []

# === Process each segment ===
for segment in data:
    # Convert time
    segment["startTime_hhmmss"] = seconds_to_hhmmss(segment["startTime"])
    segment["endTime_hhmmss"] = seconds_to_hhmmss(segment["endTime"])

    # Extract posteriors
    emotions = segment.get("emotion_posteriors", {})
    positivity = segment.get("positivity_posteriors", {})
    strength = segment.get("strength_posteriors", {})

    happy = float(emotions.get("happy", 0))
    positive = float(positivity.get("positive", 0))
    angry = float(emotions.get("angry", 0))
    negative = float(positivity.get("negative", 0))
    strong = float(strength.get("strong", 0))

    happy_sum = (happy + positive) * strong
    angry_sum = (angry + negative) * strong
    sum_all = happy_sum + angry_sum

    if strong >= 0.7:
        sum_all *= 1.5

    segment["happy_sum"] = happy_sum
    segment["angry_sum"] = angry_sum
    segment["sum_all"] = sum_all

    # Accumulate for stats
    sum_all_list.append(sum_all)
    strong_list.append(strong)

# === Compute summary statistics ===
def compute_stats(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }

summary_stats = {
    "summary_statistics": {
        "sum_all": compute_stats(sum_all_list),
        "strong": compute_stats(strong_list)
    }
}

# === Save full output with stats at the end ===
output_data = data + [summary_stats]

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2)

print("✅ Processed segments saved with summary statistics.")
