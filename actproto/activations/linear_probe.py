"""Multinomial logistic regression on frozen activation vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def train_probe(vectors: np.ndarray, labels: list[str]) -> tuple[LogisticRegression, LabelEncoder]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    clf = LogisticRegression(max_iter=400)
    clf.fit(vectors, y)
    return clf, encoder


def evaluate_probe(clf: LogisticRegression, encoder: LabelEncoder, vectors: np.ndarray, labels: list[str]) -> dict[str, Any]:
    y = encoder.transform(labels)
    pred = clf.predict(vectors)
    acc = float(accuracy_score(y, pred))
    labels_order = list(encoder.classes_)
    cm = confusion_matrix(y, pred, labels=list(range(len(labels_order)))).tolist()
    return {
        "accuracy": acc,
        "n": len(labels),
        "n_classes": len(labels_order),
        "classes": labels_order,
        "confusion_matrix": cm,
    }


def save_probe(path: str | Path, clf: LogisticRegression, encoder: LabelEncoder) -> None:
    import pickle

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pickle.dumps({"clf": clf, "encoder": encoder}))
