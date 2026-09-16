"""Planet vs. false-positive classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

DEFAULT_MODEL = Path("models/exovet.joblib")
NON_FEATURE_COLUMNS = {"toi", "tic_id", "label"}


@dataclass
class VettingModel:
    classifier: HistGradientBoostingClassifier
    feature_names: list[str]

    def predict_proba(self, features: pd.DataFrame | dict[str, float]) -> np.ndarray:
        """Probability that each candidate is a planet."""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        return self.classifier.predict_proba(features[self.feature_names])[:, 1]

    def save(self, path: Path = DEFAULT_MODEL) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = DEFAULT_MODEL) -> VettingModel:
        return joblib.load(path)


def _make_classifier(seed: int) -> HistGradientBoostingClassifier:
    # Handles NaN features natively, so unmeasurable diagnostics need no imputation.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed
    )


def train(dataset: pd.DataFrame, folds: int = 5, seed: int = 0) -> tuple[VettingModel, dict]:
    """Fit on the full dataset and report cross-validated metrics."""
    feature_names = [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS]
    X, y = dataset[feature_names], dataset["label"].astype(int)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = cross_val_predict(_make_classifier(seed), X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {
        "n": len(y),
        "n_planets": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, oof)),
        "average_precision": float(average_precision_score(y, oof)),
    }

    classifier = _make_classifier(seed).fit(X, y)
    return VettingModel(classifier, feature_names), metrics
