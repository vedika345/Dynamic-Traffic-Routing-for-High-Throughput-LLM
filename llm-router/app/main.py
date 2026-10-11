"""
FastAPI entrypoint for the LLM Router.

Run with:
    uvicorn app.main:app --reload

Endpoints:
    GET  /health   -> simple liveness check
    POST /chat     -> route a prompt to the right local LLM and return its reply
    GET  /stats    -> summary stats from logs/requests.jsonl
"""

import json
import os
import time
from collections import Counter

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.classifier import TaskClassifier
from app.model_pool import ModelPool
from app.router import Router
from app.logger import log_request, LOG_FILE

CONFIG_PATH = "config/router_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

app = FastAPI(title="LLM Router", version="1.0.0")

# These are created once at startup and reused for every request.
classifier = TaskClassifier()
model_pool = ModelPool(config)
router = Router(
    classifier,
    model_pool,
    complexity_threshold=config["routing_thresholds"]["complexity_high"],
)


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    model_used: str
    model_key: str
    task_type: str
    complexity: float
    latency: float
    fallback_used: bool


@app.get("/")
def root():
    return {
        "message": "LLM Router is running.",
        "try": ["/docs", "/health", "/stats"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    start = time.time()
    try:
        result = router.route(req.prompt)
    except RuntimeError as e:
        # All models in the fallback chain failed.
        raise HTTPException(status_code=503, detail=str(e))

    total_latency = time.time() - start
    log_request(req.prompt, result, total_latency)

    return ChatResponse(
        response=result["response"],
        model_used=result["model_used"],
        model_key=result["model_key"],
        task_type=result["task_type"],
        complexity=result["complexity"],
        latency=result["latency"],
        fallback_used=result["fallback_used"],
    )


@app.get("/stats")
def stats():
    """Quick summary of routing behaviour from the request log."""
    if not os.path.exists(LOG_FILE):
        return {"total_requests": 0}

    model_counter = Counter()
    task_counter = Counter()
    fallback_count = 0
    total_latency = 0.0
    n = 0

    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            model_counter[entry["model_used"]] += 1
            task_counter[entry["task_type"]] += 1
            fallback_count += int(entry["fallback_used"])
            total_latency += entry["total_latency"]
            n += 1

    return {
        "total_requests": n,
        "model_usage": dict(model_counter),
        "task_type_usage": dict(task_counter),
        "fallback_count": fallback_count,
        "avg_total_latency_sec": (total_latency / n) if n else 0,
    }
