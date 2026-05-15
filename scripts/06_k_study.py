#!/usr/bin/env python
"""Study the impact of kNN graph connectivity on EdgeConv performance."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.graphs.build_graphs import (  # noqa: E402
    HYBRID_FEATURE_MODE,
    build_graph_dataset,
    save_graph_dataset,
)
from jetgraph.training.train_gnn import (  # noqa: E402
    build_model,
    configure_matplotlib,
    evaluate_gnn,
    load_graphs,
    make_data_loaders,
    resolve_device,
    set_random_seed,
    split_graphs,
    train_gnn,
)


DEFAULT_K_VALUES = (4, 8, 12, 16)
DEFAULT_INPUT = Path("data/processed/qg_jets_sample.npz")
DEFAULT_RESULTS = Path("data/processed/k_study_results.csv")
DEFAULT_PLOT = Path("figures/k_study_auc.png")


@dataclass(frozen=True)
class KStudyResult:
    """One row of the k-connectivity study."""

    k: int
    graph_path: Path
    test_accuracy: float
    test_roc_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an EdgeConv GNN study over kNN graph connectivity values."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processed NPZ input path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Connectivity values to study. Defaults to 4 8 12 16.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for each k value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Graphs per mini-batch.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for the EdgeConv classifier.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation fraction of the graph dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test fraction of the graph dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits and training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Output CSV path. Defaults to {DEFAULT_RESULTS}.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help=f"Output AUC plot path. Defaults to {DEFAULT_PLOT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_path = resolve_project_path(args.input)

    print("k-nearest-neighbour connectivity study")
    print("--------------------------------------")
    print(f"k values: {', '.join(str(k) for k in args.k_values)}")
    print(f"feature mode: {HYBRID_FEATURE_MODE}")
    print(f"device: {device}")
    print()

    results: list[KStudyResult] = []
    for k in args.k_values:
        graph_path = graph_dataset_path(k)
        ensure_hybrid_graph_dataset(input_path, graph_path, k)

        print(f"Training EdgeConv for k={k}...")
        result = train_and_evaluate_k(args, graph_path, k, device)
        results.append(result)
        print(
            f"k={k}: test accuracy={result.test_accuracy:.4f}, "
            f"test AUC={result.test_roc_auc:.4f}"
        )
        print()

    csv_path = save_results_csv(results, resolve_project_path(args.results))
    plot_path = plot_k_study_auc(results, resolve_project_path(args.plot))

    print("Summary")
    print("-------")
    print(format_summary_table(results))
    print(f"\nSaved results: {csv_path}")
    print(f"Saved AUC plot: {plot_path}")


def train_and_evaluate_k(
    args: argparse.Namespace,
    graph_path: Path,
    k: int,
    device,
) -> KStudyResult:
    set_random_seed(args.random_state)
    graphs = load_graphs(graph_path)
    train_graphs, val_graphs, test_graphs = split_graphs(
        graphs,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_loader, val_loader, test_loader = make_data_loaders(
        train_graphs,
        val_graphs,
        test_graphs,
        batch_size=args.batch_size,
    )

    model = build_model(input_dim=graphs[0].x.shape[1], hidden_dim=args.hidden_dim)
    train_gnn(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        verbose=False,
    )
    test_result = evaluate_gnn(model, test_loader, device=device)

    return KStudyResult(
        k=k,
        graph_path=graph_path,
        test_accuracy=test_result.accuracy,
        test_roc_auc=test_result.roc_auc,
    )


def ensure_hybrid_graph_dataset(input_path: Path, graph_path: Path, k: int) -> None:
    if graph_path.exists():
        print(f"Using existing graph dataset for k={k}: {graph_path}")
        return

    print(f"Building hybrid graph dataset for k={k}: {graph_path}")
    with np.load(input_path) as data:
        if "jets" not in data or "labels" not in data:
            raise KeyError("Processed dataset must contain 'jets' and 'labels'.")
        graphs = build_graph_dataset(
            np.asarray(data["jets"]),
            np.asarray(data["labels"]),
            k=k,
            feature_mode=HYBRID_FEATURE_MODE,
        )
    save_graph_dataset(graphs, graph_path)


def graph_dataset_path(k: int) -> Path:
    return resolve_project_path(Path(f"data/processed/qg_graphs_k{k}_hybrid.pt"))


def save_results_csv(results: list[KStudyResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["k", "test_accuracy", "test_roc_auc", "graph_path"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "k": result.k,
                    "test_accuracy": f"{result.test_accuracy:.6f}",
                    "test_roc_auc": f"{result.test_roc_auc:.6f}",
                    "graph_path": str(result.graph_path),
                }
            )
    return output_path


def plot_k_study_auc(results: list[KStudyResult], output_path: Path) -> Path:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k_values = [result.k for result in results]
    auc_values = [result.test_roc_auc for result in results]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(k_values, auc_values, marker="o", linewidth=2, color="#4c78a8")
    ax.set_xlabel("k nearest neighbours")
    ax.set_ylabel("Test ROC AUC")
    ax.set_title("EdgeConv Connectivity Study")
    ax.set_xticks(k_values)
    ax.set_ylim(min(auc_values) - 0.01, max(auc_values) + 0.01)
    ax.grid(alpha=0.25)

    for k, auc in zip(k_values, auc_values, strict=True):
        ax.text(k, auc + 0.001, f"{auc:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def format_summary_table(results: list[KStudyResult]) -> str:
    rows = [("k", "Test Accuracy", "Test ROC AUC")]
    rows.extend(
        (str(result.k), f"{result.test_accuracy:.4f}", f"{result.test_roc_auc:.4f}")
        for result in results
    )
    widths = [max(len(row[index]) for row in rows) for index in range(3)]
    lines = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows[1:]
    )
    return "\n".join(lines)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
