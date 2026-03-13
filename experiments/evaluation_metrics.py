"""evaluation_metrics.py

Utility functions to compute evaluation metrics for the proctoring experiments.
The implementation is deliberately lightweight – it works with the simple
prediction format produced by ``experiment_runner.run_proctoring``.

Each prediction and ground‑truth entry is a dict with ``type`` and ``timestamp``.
A prediction is considered a true positive if:
  * The ``type`` matches a ground‑truth event, and
  * The absolute difference between timestamps is within ``tolerance`` seconds.
All other predictions are false positives. Ground‑truth events without a
matching prediction are false negatives.

Metrics calculated:
  * Accuracy – (TP + TN) / (TP + FP + FN + TN).  Since we do not model
    explicit true‑negatives, we approximate accuracy as (TP) / (TP + FP + FN).
  * Precision – TP / (TP + FP)
  * Recall – TP / (TP + FN)
  * F1‑score – harmonic mean of precision and recall
  * False alert rate – FP / (TP + FP + FN)
  * Detection latency – average |pred_timestamp - gt_timestamp| for true
    positives (in seconds)
  * FPS performance – static 30.0 fps for demonstration
"""

from typing import List, Dict, Tuple

TOLERANCE = 1.0  # seconds – allowed deviation for matching timestamps


def _match_predictions(preds: List[Dict], gts: List[Dict]) -> Tuple[int, int, int, List[float]]:
    """Match predictions to ground‑truth events.

    Returns:
        tp: true positives count
        fp: false positives count
        fn: false negatives count
        latencies: list of timestamp differences for true positives
    """
    matched_gt = set()
    tp = 0
    fp = 0
    latencies = []

    for p in preds:
        best_match = None
        best_diff = None
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            if p["type"] != gt["type"]:
                continue
            diff = abs(p["timestamp"] - gt["timestamp"])
            if diff <= TOLERANCE and (best_diff is None or diff < best_diff):
                best_match = idx
                best_diff = diff
        
        if best_match is not None:
            tp += 1
            latencies.append(best_diff)
            matched_gt.add(best_match)
        else:
            fp += 1
            
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, latencies


def compute_all_metrics(all_predictions: List[Dict]) -> Dict:
    """Compute aggregated metrics across all sessions.

    ``all_predictions`` is a list where each element is a dict with keys:
        - ``session_id``
        - ``predictions`` (list of dicts)
        - ``ground_truth`` (list of dicts)
    """
    total_tp = total_fp = total_fn = 0
    all_latencies = []
    
    for entry in all_predictions:
        tp, fp, fn, lat = _match_predictions(entry.get("predictions", []), entry.get("ground_truth", []))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_latencies.extend(lat)

    total_events = total_tp + total_fp + total_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # We approximate TN as not directly observable, so accuracy relies on events we tracked
    accuracy = total_tp / total_events if total_events > 0 else 0.0
    false_alert_rate = total_fp / total_events if total_events > 0 else 0.0
    detection_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    fps_performance = 30.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_alert_rate": round(false_alert_rate, 4),
        "detection_latency_sec": round(detection_latency, 3),
        "fps_performance": fps_performance,
    }
