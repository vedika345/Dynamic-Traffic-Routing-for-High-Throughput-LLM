"""
Lightweight request logger.

Every call to /chat is appended as one JSON line to logs/requests.jsonl.
This log is what scripts/benchmark.py and the Streamlit dashboard read
to show routing statistics (which models get used, latency, fallback
rate, etc.).
"""

import json
import os
import time

LOG_FILE = "logs/requests.jsonl"


def log_request(prompt: str, result: dict, total_latency: float) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp": time.time(),
        "prompt": prompt[:200],
        "task_type": result["task_type"],
        "complexity": result["complexity"],
        "model_key": result["model_key"],
        "model_used": result["model_used"],
        "fallback_used": result["fallback_used"],
        "model_latency": result["latency"],
        "total_latency": total_latency,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
