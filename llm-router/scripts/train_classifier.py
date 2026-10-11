"""
Train the complexity classifier.

Reads data/training_prompts.csv (columns: prompt, complexity[0/1]),
extracts numeric features for each prompt (see app/features.py),
trains a logistic regression model, and saves it to
models/complexity_clf.joblib so app/classifier.py can load it.

Run from the project root:
    python scripts/train_classifier.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Allow running this script directly (so `app` package is importable)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.features import extract_complexity_features  # noqa: E402

DATA_PATH = "data/training_prompts.csv"
MODEL_OUT = "models/complexity_clf.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    X = np.vstack([extract_complexity_features(p) for p in df["prompt"]])
    y = df["complexity"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluation on held-out test split:")
    print(classification_report(y_test, model.predict(X_test)))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved trained complexity model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
