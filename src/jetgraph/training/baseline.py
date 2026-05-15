"""Baseline classifiers built from physics-inspired jet observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from jetgraph.physics.observables import compute_jet_observables


FEATURE_NAMES = ("multiplicity", "total_pt", "mass", "eta_width", "phi_width")


@dataclass(frozen=True)
class ObservableDataset:
    """Feature matrix and labels for the baseline classifiers."""

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]


@dataclass
class BaselineResult:
    """Metrics and curves for one trained baseline model."""

    name: str
    model: object
    accuracy: float
    roc_auc: float
    classification_report: str
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray


def load_observable_dataset(
    input_path: str | Path,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> ObservableDataset:
    """Load baseline observables and labels from a processed JetGraph NPZ file."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run scripts/01_load_qg_dataset.py first."
        )

    with np.load(path) as data:
        if "labels" not in data:
            raise KeyError(f"Processed dataset is missing required array: labels")

        labels = np.asarray(data["labels"]).astype(int)
        observables = _load_or_compute_observables(data, feature_names)

    X = np.column_stack([observables[name] for name in feature_names]).astype(float)
    y = labels.reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature rows ({X.shape[0]}) do not match label count ({y.shape[0]})."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix contains NaN or infinite values.")

    return ObservableDataset(X=X, y=y, feature_names=feature_names)


def build_baseline_models(random_state: int = 42) -> dict[str, object]:
    """Construct the three requested sklearn baseline classifiers."""

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=random_state),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def train_and_evaluate_baselines(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
    models: Mapping[str, object] | None = None,
) -> dict[str, BaselineResult]:
    """Split data, train the baseline models, and compute classification metrics."""

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split

    model_map = dict(models or build_baseline_models(random_state=random_state))
    stratify = y if np.unique(y).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    results: dict[str, BaselineResult] = {}
    for name, model in model_map.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = positive_class_scores(model, X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        results[name] = BaselineResult(
            name=name,
            model=model,
            accuracy=float(accuracy_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_score)),
            classification_report=classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["gluon", "quark"],
                digits=3,
                zero_division=0,
            ),
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            y_pred=y_pred,
            y_score=y_score,
        )

    return results


def positive_class_scores(
    model: object,
    X: np.ndarray,
    *,
    positive_label: int = 1,
) -> np.ndarray:
    """Return classifier scores for the positive class used by ROC AUC."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive_index = _class_index(classes, positive_label)
        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))

    raise TypeError(f"Model {model!r} does not expose probabilities or scores.")


def format_summary_table(results: Mapping[str, BaselineResult]) -> str:
    """Format accuracy and ROC AUC values as a compact text table."""

    rows = [("Model", "Accuracy", "ROC AUC")]
    rows.extend(
        (name, f"{result.accuracy:.4f}", f"{result.roc_auc:.4f}")
        for name, result in results.items()
    )
    widths = [max(len(row[i]) for row in rows) for i in range(3)]

    lines = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows[1:]
    )
    return "\n".join(lines)


def _load_or_compute_observables(
    data: Mapping[str, np.ndarray],
    feature_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    missing = [name for name in feature_names if name not in data]
    if missing:
        if "jets" not in data:
            missing_names = ", ".join(missing)
            raise KeyError(
                f"Processed dataset is missing observables ({missing_names}) "
                "and does not contain jets for recomputation."
            )
        observables = compute_jet_observables(np.asarray(data["jets"]))
    else:
        observables = {}

    return {
        name: np.asarray(data[name] if name in data else observables[name]).reshape(-1)
        for name in feature_names
    }


def _class_index(classes: np.ndarray, label: int) -> int:
    matches = np.flatnonzero(classes == label)
    if matches.size == 0:
        raise ValueError(f"Positive label {label} is not present in model classes.")
    return int(matches[0])
