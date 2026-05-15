#!/usr/bin/env python
"""Create a simple ROC AUC comparison plot for JetGraph models."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODEL_AUCS = {
    "Logistic Regression": 0.8579,
    "Random Forest": 0.8451,
    "Gradient Boosting": 0.8616,
    "EdgeConv raw": 0.8580,
    "EdgeConv raw no pid": 0.8533,
    "EdgeConv physics": 0.8102,
    "EdgeConv hybrid": 0.8617,
    "EdgeConv hybrid no pid": 0.8585,
}
DEFAULT_OUTPUT = Path("figures/model_comparison.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC AUC scores for JetGraph baseline and GNN models."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_model_comparison(MODEL_AUCS, resolve_project_path(args.output))

    print("Model comparison")
    print("----------------")
    for model_name, auc in MODEL_AUCS.items():
        print(f"{model_name:>19}: ROC AUC = {auc:.4f}")
    print(f"\nSaved comparison plot: {output_path}")


def plot_model_comparison(model_aucs: dict[str, float], output_path: Path) -> Path:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(model_aucs)
    auc_values = list(model_aucs.values())
    colors = [
        "#4c78a8",
        "#72b7b2",
        "#f58518",
        "#54a24b",
        "#9d755d",
        "#e45756",
        "#b279a2",
        "#ff9da6",
    ]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    bars = ax.bar(model_names, auc_values, color=colors)

    ax.set_ylabel("ROC AUC")
    ax.set_title("JetGraph Model Comparison")
    ax.set_ylim(0.78, 0.88)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=24)

    for bar, auc in zip(bars, auc_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.001,
            f"{auc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "jetgraph-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg", force=True)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
