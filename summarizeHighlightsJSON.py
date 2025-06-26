"""import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_JSON = "output_segments_with_hhmmss-PORT-SPAIN.json"
OUTPUT_JSON = "highlights-PORT-SPAIN.json"
MIN_DURATION = 4.0  # seconds



# === Load data ===
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# === Extract saliencies ===
sum_saliencies = [
    float(seg.get("sum_all", 0))
    for seg in data
    if float(seg.get("endTime", 0)) - float(seg.get("startTime", 0)) > MIN_DURATION
]

print("🔍 Summary of raw sum_saliencies:")
print(f"  Count: {len(sum_saliencies)}")
print(f"  Min  : {min(sum_saliencies):.6f}")
print(f"  Max  : {max(sum_saliencies):.6f}")
print(f"  Mean : {np.mean(sum_saliencies):.6f}")
print(f"  Std  : {np.std(sum_saliencies):.6f}")

# === Plot and save with high resolution on Y-axis ===
plt.figure(figsize=(12, 6))
plt.scatter(range(len(sum_saliencies)), sum_saliencies, alpha=0.6, s=10)
plt.title("Scatter Plot of sum_saliencies per Segment")
plt.xlabel("Segment Index")
plt.ylabel("sum_saliencies")
plt.grid(True)

# Set precise ticks on Y-axis
y_max = max(sum_saliencies)
plt.yticks(np.arange(0, y_max + 0.1, 0.1))  # Every 0.1 until slightly beyond max
plt.ylim(0, y_max + 0.1)

plt.tight_layout()
plot_path = "saliency_scatter_plot.png"
plt.savefig(plot_path, dpi=300)
print(f"📊 Saliency scatter plot saved to '{plot_path}'")

# === Prompt for standard deviation multiplier ===
try:
    threshold = float(input("Enter the saliency score threshold): ").strip())
except ValueError:
    print("❌ Invalid input for K. Please enter a numeric value.")
    exit(1)


# === Filter highlights ===
highlights = [
    segment for segment in data
    if float(segment.get("sum_all", 0)) >= threshold and
       (float(segment["endTime"]) - float(segment["startTime"])) > MIN_DURATION
]

# === Save result ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(highlights, f, indent=2)

print(f"✅ Extracted {len(highlights)} highlight segments to: {OUTPUT_JSON}")
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_JSON = "output_segments_with_hhmmss-PORT-SPAIN.json"
OUTPUT_JSON = "highlights-PORT-SPAIN.json"
MIN_DURATION = 4.0  # seconds
TOP_PERCENT = 6  # percentage of top saliency segments to keep

# === Load data ===
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# === Filter valid segments and extract saliencies ===
valid_segments = [
    seg for seg in data
    if float(seg.get("endTime", 0)) - float(seg.get("startTime", 0)) > MIN_DURATION
]
sum_saliencies = [float(seg.get("sum_all", 0)) for seg in valid_segments]

# === Stats ===
print("🔍 Summary of raw sum_saliencies:")
print(f"  Count: {len(sum_saliencies)}")
print(f"  Min  : {min(sum_saliencies):.6f}")
print(f"  Max  : {max(sum_saliencies):.6f}")
print(f"  Mean : {np.mean(sum_saliencies):.6f}")
print(f"  Std  : {np.std(sum_saliencies):.6f}")

# === Plot and save ===
plt.figure(figsize=(12, 6))
plt.scatter(range(len(sum_saliencies)), sum_saliencies, alpha=0.6, s=10)
plt.title("Scatter Plot of sum_saliencies per Segment")
plt.xlabel("Segment Index")
plt.ylabel("sum_saliencies")
plt.grid(True)

y_max = max(sum_saliencies)
plt.yticks(np.arange(0, y_max + 0.1, 0.1))
plt.ylim(0, y_max + 0.1)

plt.tight_layout()
plot_path = "saliency_scatter_plot.png"
plt.savefig(plot_path, dpi=300)
print(f"📊 Saliency scatter plot saved to '{plot_path}'")

# === Keep top X% ===
num_top = max(1, int(len(valid_segments) * TOP_PERCENT / 100))
top_segments = sorted(valid_segments, key=lambda x: float(x.get("sum_all", 0)), reverse=True)[:num_top]

# === Save ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(top_segments, f, indent=2)

print(f"✅ Extracted top {TOP_PERCENT}% ({len(top_segments)} segments) to: {OUTPUT_JSON}")
