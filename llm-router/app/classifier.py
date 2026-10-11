"""
TaskClassifier
==============

Classifies an incoming prompt along two independent axes:

1. task_type  -> which *kind* of work is being asked for
                 ("code", "math_reasoning", "creative",
                  "summarization", "general_chat")
                 Decided via embedding similarity using a small
                 sentence-transformer model (zero-shot, no training needed).

2. complexity -> a 0.0-1.0 score for "how hard / expensive" the request is.
                 Decided by a small logistic-regression model trained on
                 data/training_prompts.csv (see scripts/train_classifier.py).
                 If that model hasn't been trained yet, falls back to a
                 simple heuristic so the router still works out of the box.
"""

import os
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.features import extract_complexity_features

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# One short natural-language description per task category.
# The classifier picks the category whose description is most
# *semantically similar* to the user's prompt.
TASK_CATEGORIES = {
    "code": "Writing, debugging, explaining, or reviewing source code, "
            "programming, scripts, functions, algorithms, software engineering",
    "math_reasoning": "Solving math problems, logical reasoning, step by step "
                       "proofs, calculations, puzzles, derivations",
    "creative": "Creative writing, stories, poems, brainstorming, dialogue, "
                 "fiction, songs",
    "summarization": "Summarizing, condensing, or extracting key points from "
                      "text, articles, or documents",
    "general_chat": "General conversation, simple factual questions, "
                     "greetings, definitions, casual chat",
}

DEFAULT_COMPLEXITY_MODEL_PATH = "models/complexity_clf.joblib"


class TaskClassifier:
    def __init__(self, complexity_model_path: str = DEFAULT_COMPLEXITY_MODEL_PATH):
        # Loads/downloads the embedding model the first time it's run.
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.category_names = list(TASK_CATEGORIES.keys())
        self.category_embeddings = self.embedder.encode(list(TASK_CATEGORIES.values()))

        self.complexity_model = None
        if os.path.exists(complexity_model_path):
            self.complexity_model = joblib.load(complexity_model_path)

    # ------------------------------------------------------------------
    # Task type
    # ------------------------------------------------------------------
    def classify_task_type(self, prompt: str) -> str:
        prompt_emb = self.embedder.encode([prompt])
        sims = cosine_similarity(prompt_emb, self.category_embeddings)[0]
        best_idx = int(np.argmax(sims))
        return self.category_names[best_idx]

    # ------------------------------------------------------------------
    # Complexity
    # ------------------------------------------------------------------
    def classify_complexity(self, prompt: str) -> float:
        features = extract_complexity_features(prompt)

        if self.complexity_model is not None:
            proba = self.complexity_model.predict_proba(features)[0][1]
            return float(proba)

        # ---- Fallback heuristic (used if the model hasn't been trained yet) ----
        word_count, _sentence_count, has_code, has_multistep, _q = features[0]
        score = min(word_count / 100.0, 1.0) * 0.5
        score += has_code * 0.2
        score += has_multistep * 0.3
        return float(min(score, 1.0))

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------
    def classify(self, prompt: str):
        task_type = self.classify_task_type(prompt)
        complexity = self.classify_complexity(prompt)
        return task_type, complexity
