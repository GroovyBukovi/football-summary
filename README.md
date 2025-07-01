# ⚽ Multimodal Football Highlight Detection

This project explores the automatic summarization of football matches by leveraging **multimodal signals** — specifically **audio emotion analysis**, **text-based event extraction**, and basic **visual analysis**. The system identifies emotionally and semantically salient moments in match recordings and compiles them into concise highlight reels.

## Project Summary

We use:
- **Audio**: Emotional saliency computed via the Behavioral Signals API.
- **Text**: Commentary transcribed via Whisper and analyzed using LLMs for key events.
- **Evaluation**: Comparison with official highlights using flexible IoU-based matching.

Evaluation metrics include:
- Precision, Recall, F1
- Global IoU
- AUC
- TP / FP / FN counts

## Results Summary

| Evaluation Mode     | Precision | Recall | F1 Score | AUC  |
|---------------------|-----------|--------|----------|------|
| Strict              | 0.6606    | 0.3529 | 0.4601   | 0.6730 |
| Flexible (≥75%)     | 0.6818    | 0.3676 | 0.4777   | 0.6805 |
| Super Flexible (≥20%) | 0.7578    | 0.4755 | 0.5843   | 0.7348 |

> Audio saliency emerged as the strongest standalone signal for emotionally meaningful highlights.
