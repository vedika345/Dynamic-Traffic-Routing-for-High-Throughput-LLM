"""
Feature extraction for the prompt-complexity classifier.

Kept in its own module (no heavy imports like sentence-transformers)
so it can be reused by both:
  - app/classifier.py        (at inference time)
  - scripts/train_classifier.py  (at training time)
"""

import re
import numpy as np


def extract_complexity_features(prompt: str) -> np.ndarray:
    """
    Turn a raw prompt into a small numeric feature vector that's
    predictive of "how hard/expensive" the request likely is.

    Returns a (1, 5) numpy array with columns:
      [word_count, sentence_count, has_code, has_multistep_language, question_count]
    """
    text = prompt.strip()

    word_count = len(text.split())

    sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    sentence_count = max(sentence_count, 1)

    has_code = 1 if re.search(
        r"```|def |class |import |function|SELECT |for \(|while \(", text
    ) else 0

    has_multistep = 1 if re.search(
        r"step by step|step-by-step|explain in detail|in detail|derive|prove|"
        r"analyze|analyse|compare and contrast|walk me through|design a",
        text,
        re.IGNORECASE,
    ) else 0

    question_count = text.count("?")

    return np.array([[word_count, sentence_count, has_code, has_multistep, question_count]])
