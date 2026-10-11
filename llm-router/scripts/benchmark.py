"""
Benchmark / smoke-test the running router.

Sends a small batch of prompts covering different task types and
complexity levels, then prints which model handled each one, the
latency, and whether a fallback occurred.

Make sure the API is running first:
    uvicorn app.main:app --reload

Then in another terminal:
    python scripts/benchmark.py
"""

import json
import statistics
import time

import requests

ROUTER_URL = "http://localhost:8000/chat"

TEST_PROMPTS = [
    ("Hi, how are you?", "simple chat"),
    ("What's the capital of France?", "simple chat"),
    ("Write a Python function to compute Fibonacci numbers recursively.", "code"),
    ("Explain step by step how to prove the Pythagorean theorem.", "math reasoning (hard)"),
    ("Write a short poem about the ocean at night.", "creative"),
    ("Summarize the plot of Romeo and Juliet in three sentences.", "summarization"),
    ("Derive the quadratic formula and explain each step in detail.", "math reasoning (hard)"),
    ("Debug this code: def add(a, b): return a + b +", "code"),
]


def run_benchmark():
    results = []

    for prompt, expected in TEST_PROMPTS:
        try:
            resp = requests.post(ROUTER_URL, json={"prompt": prompt}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            results.append({
                "prompt": prompt,
                "expected_category": expected,
                "task_type": data["task_type"],
                "complexity": round(data["complexity"], 3),
                "model_used": data["model_used"],
                "latency_sec": round(data["latency"], 2),
                "fallback_used": data["fallback_used"],
            })
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})

        time.sleep(0.3)

    print(json.dumps(results, indent=2))

    latencies = [r["latency_sec"] for r in results if "latency_sec" in r]
    fallback_count = sum(1 for r in results if r.get("fallback_used"))

    if latencies:
        print(f"\nAverage latency: {statistics.mean(latencies):.2f}s")
        print(f"Fallback rate:  {fallback_count}/{len(results)}")


if __name__ == "__main__":
    run_benchmark()
