#!/usr/bin/env python
"""Train physics-observable baseline classifiers for quark/gluon tagging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.evaluation.plots import plot_feature_importances, plot_roc_curves
from jetgraph.training.baseline import (
    FEATURE_NAMES,
    format_summary_table,
    load_observable_dataset,
    train_and_evaluate_baselines,
)


DEFAULT_INPUT = Path("data/processed/qg_jets_sample.npz")
DEFAULT_ROC_OUTPUT = Path("figures/baseline_roc.png")
DEFAULT_IMPORTANCE_OUTPUT = Path("figures/baseline_feature_importance.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline classifiers on JetGraph physics observables."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processed NPZ dataset path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=DEFAULT_ROC_OUTPUT,
        help=f"Path for the ROC curve PNG. Defaults to {DEFAULT_ROC_OUTPUT}.",
    )
    parser.add_argument(
        "--importance-output",
        type=Path,
        default=DEFAULT_IMPORTANCE_OUTPUT,
        help=(
            "Path for the tree-model feature importance PNG. "
            f"Defaults to {DEFAULT_IMPORTANCE_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of jets held out for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and stochastic classifiers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_observable_dataset(resolve_project_path(args.input))
    results = train_and_evaluate_baselines(
        dataset.X,
        dataset.y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    roc_path = plot_roc_curves(results, resolve_project_path(args.roc_output))
    importance_path = plot_feature_importances(
        {name: result.model for name, result in results.items()},
        dataset.feature_names,
        resolve_project_path(args.importance_output),
    )

    print_run_summary(dataset, results)
    print(f"\nSaved ROC curve: {roc_path}")
    print(f"Saved feature importances: {importance_path}")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_run_summary(dataset, results) -> None:
    print("Baseline observable classifier summary")
    print("--------------------------------------")
    print(f"jets:     {dataset.X.shape[0]}")
    print(f"features: {', '.join(FEATURE_NAMES)}")
    print()
    print(format_summary_table(results))

    print("\nClassification reports")
    print("----------------------")
    for name, result in results.items():
        print(f"\n{name}")
        print(result.classification_report)


if __name__ == "__main__":
    main()
