import json
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime


def hhmmss_to_seconds(hhmmss: str) -> int:
    """Convert 'hh:mm:ss' string to total seconds."""
    t = datetime.strptime(hhmmss, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second


def load_segments(path):
    with open(path) as f:
        return [
            (hhmmss_to_seconds(d["startTime_hhmmss"]), hhmmss_to_seconds(d["endTime_hhmmss"]))
            for d in json.load(f)
        ]



def compute_global_iou(preds, gts):
    """Calculate IoU based on total overlapping time vs total union time."""
    pred_mask = np.zeros(max(e for _, e in preds + gts) + 1, dtype=bool)
    gt_mask = np.zeros_like(pred_mask)

    for s, e in preds:
        pred_mask[s:e + 1] = True
    for s, e in gts:
        gt_mask[s:e + 1] = True

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def relaxed_match(pred, gt, threshold):
    """Return True if prediction covers at least `threshold` proportion of GT."""
    overlap = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    gt_length = gt[1] - gt[0]
    return overlap / gt_length >= threshold


def evaluate(preds, gts, mode="strict"):
    pred_mask = np.zeros(max(e for _, e in preds + gts) + 1, dtype=bool)
    gt_mask = np.zeros_like(pred_mask)

    threshold = 0.75 if mode == "flexible" else (0.2 if mode == "superflexible" else None)

    if threshold is not None:
        matched_gts = set()
        for ps, pe in preds:
            for idx, (gs, ge) in enumerate(gts):
                if idx not in matched_gts and relaxed_match((ps, pe), (gs, ge), threshold):
                    pred_mask[gs:ge + 1] = True
                    gt_mask[gs:ge + 1] = True
                    matched_gts.add(idx)
                    break
            else:
                pred_mask[ps:pe + 1] = True
    else:
        for s, e in preds:
            pred_mask[s:e + 1] = True
        for s, e in gts:
            gt_mask[s:e + 1] = True

    for s, e in gts:
        gt_mask[s:e + 1] = True

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    auc = roc_auc_score(gt_mask, pred_mask) if len(set(gt_mask)) > 1 else None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "global_iou": compute_global_iou(preds, gts),
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# === Run evaluation ===
print("Choose Evaluation Mode:\n 1. Strict (exact second-based overlap)\n 2. Flexible (≥75% coverage equals full match)\n 3. Super Flexible (≥20% coverage equals full match)")
mode_input = input("Enter 1, 2 or 3: ").strip()
mode = "flexible" if mode_input == "2" else "superflexible" if mode_input == "3" else "strict"

ground_truth = load_segments("highlights-GROUNDTRUTH.json")
predictions = load_segments("highlights-PORTUGAL-SPAIN-final.json")

metrics = evaluate(predictions, ground_truth, mode=mode)

print(f"\n📊 Evaluation Summary ({mode.capitalize()} Mode)")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:>12}: {v:.4f}")
    else:
        print(f"{k:>12}: {v}")
