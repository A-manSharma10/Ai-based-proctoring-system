# Research Report: Single-Modal vs Multimodal Proctoring

## 1. Abstract
This study compares the detection efficiency of Single-Modal (Video-only) and Multimodal (Video, Audio, Behavior) AI systems for online proctoring.

## 2. Experimental Results
| Metric | Single-Modal | Multimodal |
|--------|-------------|------------|
| precision | 0.75 | 0.9333 |
| recall | 0.4 | 0.9333 |
| f1_score | 0.5217 | 0.9333 |
| accuracy | 0.3529 | 0.875 |
| false_alert_rate | 0.1176 | 0.0625 |
| detection_latency_sec | 0.24 | 0.249 |
| fps_performance | 30.0 | 30.0 |

## 3. Analysis & Discussion
- **Accuracy**: Multimodal proctoring captures non-visual cheating events (e.g., audio, off-screen gaze), leading to higher accuracy.
- **False Alert Rate**: Adding multiple modalities provides redundancy, lowering the false alert rate.
- **Performance**: Analyzing multiple streams may impact FPS, but the trade-off significantly boosts reliability.
