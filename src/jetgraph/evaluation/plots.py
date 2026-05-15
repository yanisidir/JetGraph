"""Plotting helpers for JetGraph model evaluation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def plot_roc_curves(results: Mapping[str, object], output_path: str | Path) -> Path:
    """Save ROC curves for a collection of evaluated models."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for name, result in results.items():
        ax.plot(
            result.fpr,
            result.tpr,
            linewidth=2,
            label=f"{name} (AUC = {result.roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Quark/Gluon Baseline ROC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importances(
    models: Mapping[str, object],
    feature_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Save feature importances for tree-based baseline models."""

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    importance_items = [
        (name, importances)
        for name, model in models.items()
        if (importances := get_feature_importances(model)) is not None
    ]
    if not importance_items:
        raise ValueError("No tree-based feature importances were found to plot.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(importance_items),
        figsize=(6 * len(importance_items), 4.5),
        dpi=150,
        squeeze=False,
    )

    for ax, (model_name, importances) in zip(axes.ravel(), importance_items):
        importances = np.asarray(importances)
        order = np.argsort(importances)
        ordered_names = np.asarray(feature_names)[order]

        ax.barh(ordered_names, importances[order], color="#3676b8")
        ax.set_title(model_name)
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Baseline Feature Importances", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def get_feature_importances(model: object) -> np.ndarray | None:
    """Return feature importances from an estimator or pipeline, if available."""

    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None
    return np.asarray(importances)


def _configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "jetgraph-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg", force=True)
