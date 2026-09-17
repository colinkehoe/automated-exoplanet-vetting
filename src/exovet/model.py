"""Planet vs. false-positive classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from exovet.diagnostics.stellar import BACKFILLED_FEATURES

DEFAULT_MODEL = Path("models/exovet.joblib")
NON_FEATURE_COLUMNS = {"toi", "tic_id", "label"}


@dataclass
class VettingModel:
    classifier: CalibratedClassifierCV
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


def _make_classifier(feature_names: list[str], seed: int) -> CalibratedClassifierCV:
    # Median-impute catalog stellar values whose missingness leaks the label
    # (see BACKFILLED_FEATURES). Everything else keeps its NaN, which the
    # booster handles natively, so unmeasurable diagnostics need no imputation.
    impute = [c for c in feature_names if c in BACKFILLED_FEATURES]
    prepare = ColumnTransformer(
        [("impute", SimpleImputer(strategy="median", keep_empty_features=True), impute)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    booster = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed
    )
    # Sigmoid calibration over star-grouped internal folds: the raw booster is
    # overconfident near 0 and 1, and averaging the fold models also helps a
    # little. Fit with metadata routing so the folds receive the star groups.
    return CalibratedClassifierCV(
        make_pipeline(prepare, booster),
        method="sigmoid",
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed),
    )


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS]


def fit(dataset: pd.DataFrame, seed: int = 0) -> VettingModel:
    feature_names = feature_columns(dataset)
    classifier = _make_classifier(feature_names, seed)
    with sklearn.config_context(enable_metadata_routing=True):
        classifier.fit(
            dataset[feature_names], dataset["label"].astype(int), groups=dataset["tic_id"]
        )
    return VettingModel(classifier, feature_names)


def cross_validated_proba(dataset: pd.DataFrame, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold planet probabilities.

    Folds keep all TOIs of a star together, so sibling candidates never sit on
    both sides of a split.
    """
    feature_names = feature_columns(dataset)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    with sklearn.config_context(enable_metadata_routing=True):
        return cross_val_predict(
            _make_classifier(feature_names, seed),
            dataset[feature_names],
            dataset["label"].astype(int),
            cv=cv,
            method="predict_proba",
            params={"groups": dataset["tic_id"]},
        )[:, 1]


def rank_candidates(model: VettingModel, candidates: pd.DataFrame) -> pd.DataFrame:
    """TOIs with their planet probability, most planet-like first."""
    ranked = candidates.copy()
    ranked["planet_probability"] = model.predict_proba(ranked)
    columns = ["toi", "tic_id", "planet_probability"]
    return ranked[columns + [c for c in ranked.columns if c not in columns]].sort_values(
        "planet_probability", ascending=False, ignore_index=True
    )


def score(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def train(dataset: pd.DataFrame, folds: int = 5, seed: int = 0) -> tuple[VettingModel, dict]:
    """Fit on the full dataset and report cross-validated metrics."""
    y = dataset["label"].astype(int)
    metrics = {
        "n": len(y),
        "n_planets": int(y.sum()),
        **score(y, cross_validated_proba(dataset, folds, seed)),
    }
    return fit(dataset, seed), metrics
