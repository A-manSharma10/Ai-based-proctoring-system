"""dataset_loader.py

Utility to load a labelled dataset for proctoring experiments.
The dataset is expected to be a JSON file with the following structure:

[
    {
        "id": "session_001",
        "video_path": "path/to/video.mp4",
        "audio_path": "path/to/audio.wav",
        "ground_truth": [
            {"type": "no_face", "timestamp": 12.3},
            {"type": "phone_usage", "timestamp": 45.7},
            ...
        ]
    },
    ...
]

Each entry in ``ground_truth`` represents a cheating event with its type and the
timestamp (in seconds) when it occurred.
"""
import json
from pathlib import Path
from typing import List, Dict


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load the dataset JSON file.

    Args:
        dataset_path: Path to the JSON file.

    Returns:
        List of session dictionaries.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Basic validation
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of session objects.")
    return data
