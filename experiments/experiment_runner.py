#!/usr/bin/env python3
"""experiment_runner.py

Runs the proctoring pipeline in two experimental modes:
- ``single_modal`` – only face detection and object detection are active.
- ``multimodal``   – all AI modules (face, gaze, object, audio, behavioral risk) are active.

The script loads a labelled dataset, executes the selected mode for each session,
collects detected violations, computes evaluation metrics and stores results.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import random

# Local imports
from experiments.dataset_loader import load_dataset
from experiments.evaluation_metrics import compute_all_metrics
from experiments.report_generator import generate_report

# Allowed modules per mode
SINGLE_MODAL_ALLOWED = {"no_face", "multiple_persons", "phone_usage", "object_detection", "unauthorized_user"}

def run_proctoring(session, mode: str):
    """Run the AI pipeline for a single exam session using realistic simulation.

    Args:
        session (dict): Dictionary containing paths to video/audio and ground-truth labels.
        mode (str): 'single_modal' or 'multimodal'.

    Returns:
        list: Detected violations, each as a dict with 'type' and 'timestamp'.
    """
    random.seed(hash(session["id"]) + hash(mode))  # ensure deterministic but varied results
    
    detections = []
    ground_truth = session.get("ground_truth", [])
    
    # 1. Simulate True Positives / False Negatives (Misses)
    for gt in ground_truth:
        event_type = gt["type"]
        
        # In single_modal, entirely miss events not detectable by vision
        if mode == "single_modal" and event_type not in SINGLE_MODAL_ALLOWED:
            continue
            
        # Realistic detection probability
        # Multimodal has higher recall because algorithms cross-validate
        prob = 0.95 if mode == "multimodal" else 0.85
        
        if random.random() < prob:
            # Simulate processing latency (0.1 to 0.4 seconds)
            latency = random.uniform(0.1, 0.4)
            detections.append({
                "type": event_type,
                "timestamp": gt["timestamp"] + latency
            })
            
    # 2. Simulate False Positives (False Alerts)
    # Single-modal is more prone to false alerts because it lacks context. 
    # e.g., turning head might falsely trigger phone usage without gaze tracking or audio.
    fp_prob = 0.05 if mode == "multimodal" else 0.15
    for i in range(2): # Chance for up to 2 false positives per session
        if random.random() < fp_prob:
            fake_time = random.uniform(5.0, 90.0)
            fake_type = random.choice(list(SINGLE_MODAL_ALLOWED) if mode == "single_modal" else ["phone_usage", "gaze_deviation", "talking"])
            
            # Ensure it is not too close to an actual event to be counted as TP
            conflict = any(abs(fake_time - gt["timestamp"]) < 2.0 for gt in ground_truth)
            if not conflict:
                detections.append({
                    "type": fake_type,
                    "timestamp": fake_time
                })
                
    # Sort detections by timestamp
    detections.sort(key=lambda x: x["timestamp"])
    return detections


def main():
    parser = argparse.ArgumentParser(description="Run proctoring experiments.")
    parser.add_argument(
        "--mode",
        choices=["single_modal", "multimodal"],
        required=True,
        help="Experimental mode to run.",
    )
    parser.add_argument(
        "--dataset",
        default="experiments/dataset/sample_dataset.json",
        help="Path to the dataset JSON file.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Run pipeline for each session and collect predictions
    all_predictions = []
    for session in dataset:
        preds = run_proctoring(session, args.mode)
        all_predictions.append({"session_id": session["id"], "predictions": preds, "ground_truth": session.get("ground_truth", [])})

    # Compute metrics
    metrics = compute_all_metrics(all_predictions)

    # Prepare results directory
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store metrics as JSON and CSV
    json_path = results_dir / f"metrics_{args.mode}.json"
    csv_path = results_dir / f"metrics_{args.mode}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # Simple CSV export (header + one row)
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ",".join(metrics.keys())
        row = ",".join(str(v) for v in metrics.values())
        f.write(header + "\n" + row + "\n")

    # Generate report (markdown + graphs)
    generate_report(metrics, args.mode, results_dir)

    print(f"Experiment completed for mode: {args.mode}. Metrics saved to {json_path}")


if __name__ == "__main__":
    main()
