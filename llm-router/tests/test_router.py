"""
Unit tests for the routing *policy* (app/router.py).

These tests do NOT need Ollama running -- the classifier and
model pool are replaced with simple fakes so we can test the
decision logic in isolation.

Run from the project root:
    pytest
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.router import Router  # noqa: E402


class FakeClassifier:
    """Returns a fixed (task_type, complexity) regardless of input."""

    def __init__(self, task_type, complexity):
        self.task_type = task_type
        self.complexity = complexity

    def classify(self, prompt):
        return self.task_type, self.complexity


def make_router(task_type, complexity, threshold=0.6):
    classifier = FakeClassifier(task_type, complexity)
    model_pool = MagicMock()
    model_pool.generate.return_value = {
        "response": "fake response",
        "model_used": "fake-model",
        "latency": 0.01,
        "fallback_used": False,
    }
    router = Router(classifier, model_pool, complexity_threshold=threshold)
    return router, model_pool


def test_code_always_routes_to_code_model():
    router, model_pool = make_router("code", complexity=0.1)
    result = router.route("write some code")
    assert result["model_key"] == "code"
    model_pool.generate.assert_called_once_with("code", "write some code")


def test_simple_general_chat_routes_to_small_model():
    router, _ = make_router("general_chat", complexity=0.1)
    result = router.route("hi")
    assert result["model_key"] == "simple_chat"


def test_complex_general_chat_routes_to_general_model():
    router, _ = make_router("general_chat", complexity=0.9)
    result = router.route("a very long, complex question...")
    assert result["model_key"] == "general"


def test_easy_math_routes_to_general_model():
    router, _ = make_router("math_reasoning", complexity=0.2)
    result = router.route("what is 5 + 7?")
    assert result["model_key"] == "general"


def test_hard_math_routes_to_complex_reasoning_model():
    router, _ = make_router("math_reasoning", complexity=0.9)
    result = router.route("prove this theorem step by step")
    assert result["model_key"] == "complex_reasoning"


def test_creative_routes_to_creative_model():
    router, _ = make_router("creative", complexity=0.3)
    result = router.route("write a poem")
    assert result["model_key"] == "creative"


def test_summarization_routes_to_general_model():
    router, _ = make_router("summarization", complexity=0.5)
    result = router.route("summarize this article")
    assert result["model_key"] == "general"


def test_result_includes_classification_metadata():
    router, _ = make_router("code", complexity=0.42)
    result = router.route("write a function")
    assert result["task_type"] == "code"
    assert result["complexity"] == 0.42
    assert "model_used" in result
    assert "latency" in result
