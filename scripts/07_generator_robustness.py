#!/usr/bin/env python
"""Study EdgeConv robustness across Pythia and Herwig qg_jets samples."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.data.load_qg import load_qg_jets, save_qg_sample  # noqa: E402
from jetgraph.graphs.build_graphs import (  # noqa: E402
    DEFAULT_K,
    HYBRID_FEATURE_MODE,
    build_graph_dataset,
    save_graph_dataset,
)
from jetgraph.physics.observables import compute_jet_observables  # noqa: E402
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


GENERATORS = ("pythia", "herwig")
DEFAULT_NUM_JETS = 10_000
DEFAULT_RESULTS = Path("data/processed/generator_robustness_results.csv")
DEFAULT_PLOT = Path("figures/generator_robustness.png")


@dataclass(frozen=True)
class GeneratorResult:
    """One train-generator/test-generator robustness result."""

    train_generator: str
    test_generator: str
    test_accuracy: float
    test_roc_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate EdgeConv across Pythia and Herwig jets."
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=DEFAULT_NUM_JETS,
        help=f"Number of jets per generator. Defaults to {DEFAULT_NUM_JETS}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for each generator-specific model.",
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
        help="Validation fraction for each generator dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test fraction for each generator dataset.",
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
        help=f"Output comparison plot path. Defaults to {DEFAULT_PLOT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("Pythia-vs-Herwig robustness study")
    print("---------------------------------")
    print(f"generators: {', '.join(GENERATORS)}")
    print(f"feature mode: {HYBRID_FEATURE_MODE}")
    print(f"k: {DEFAULT_K}")
    print(f"device: {device}")
    print()

    graph_paths = {
        generator: ensure_generator_graph_dataset(generator, args.num_jets)
        for generator in GENERATORS
    }
    splits = {
        generator: split_graphs(
            load_graphs(graph_path),
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        for generator, graph_path in graph_paths.items()
    }

    results: list[GeneratorResult] = []
    for train_generator in GENERATORS:
        set_random_seed(args.random_state)
        train_graphs, val_graphs, _ = splits[train_generator]
        train_loader, val_loader, _ = make_data_loaders(
            train_graphs,
            val_graphs,
            [],
            batch_size=args.batch_size,
        )

        input_dim = train_graphs[0].x.shape[1]
        model = build_model(input_dim=input_dim, hidden_dim=args.hidden_dim)
        print(f"Training on {train_generator}...")
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

        for test_generator in GENERATORS:
            _, _, test_graphs = splits[test_generator]
            test_loader = make_test_loader(test_graphs, args.batch_size)
            test_result = evaluate_gnn(model, test_loader, device=device)
            results.append(
                GeneratorResult(
                    train_generator=train_generator,
                    test_generator=test_generator,
                    test_accuracy=test_result.accuracy,
                    test_roc_auc=test_result.roc_auc,
                )
            )
            print(
                f"train {train_generator:>6} -> test {test_generator:<6}: "
                f"accuracy={test_result.accuracy:.4f}, AUC={test_result.roc_auc:.4f}"
            )
        print()

    csv_path = save_results_csv(results, resolve_project_path(args.results))
    plot_path = plot_generator_robustness(results, resolve_project_path(args.plot))

    print("Summary")
    print("-------")
    print(format_summary_table(results))
    print(f"\nSaved results: {csv_path}")
    print(f"Saved plot: {plot_path}")


def ensure_generator_graph_dataset(generator: str, num_jets: int) -> Path:
    graph_path = graph_dataset_path(generator)
    if graph_path.exists():
        print(f"Using existing {generator} graph dataset: {graph_path}")
        return graph_path

    sample_path = sample_dataset_path(generator)
    if sample_path.exists():
        print(f"Using existing {generator} array sample: {sample_path}")
        with np.load(sample_path) as data:
            jets = np.asarray(data["jets"])
            labels = np.asarray(data["labels"])
    else:
        print(f"Loading {generator} qg_jets sample with EnergyFlow...")
        jets, labels = load_qg_jets(
            num_jets=num_jets,
            generator=generator,
            cache_dir=resolve_project_path("data/raw/energyflow"),
        )
        observables = compute_jet_observables(jets)
        save_qg_sample(jets, labels, sample_path, observables=observables)

    print(f"Building {generator} hybrid graph dataset: {graph_path}")
    graphs = build_graph_dataset(
        jets,
        labels,
        k=DEFAULT_K,
        feature_mode=HYBRID_FEATURE_MODE,
    )
    save_graph_dataset(graphs, graph_path)
    return graph_path


def make_test_loader(graphs, batch_size):
    return DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)


def sample_dataset_path(generator: str) -> Path:
    return resolve_project_path(f"data/processed/qg_jets_{generator}_sample.npz")


def graph_dataset_path(generator: str) -> Path:
    return resolve_project_path(
        f"data/processed/qg_graphs_k8_hybrid_{generator}.pt"
    )


def save_results_csv(results: list[GeneratorResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "train_generator",
                "test_generator",
                "test_accuracy",
                "test_roc_auc",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "train_generator": result.train_generator,
                    "test_generator": result.test_generator,
                    "test_accuracy": f"{result.test_accuracy:.6f}",
                    "test_roc_auc": f"{result.test_roc_auc:.6f}",
                }
            )
    return output_path


def plot_generator_robustness(
    results: list[GeneratorResult],
    output_path: Path,
) -> Path:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        f"Train {result.train_generator}\nTest {result.test_generator}"
        for result in results
    ]
    auc_values = [result.test_roc_auc for result in results]
    colors = ["#4c78a8", "#e45756", "#e45756", "#4c78a8"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(labels, auc_values, color=colors)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Generator Robustness")
    ax.set_ylim(min(auc_values) - 0.03, max(auc_values) + 0.02)
    ax.grid(axis="y", alpha=0.25)

    for bar, auc in zip(bars, auc_values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.002,
            f"{auc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def format_summary_table(results: list[GeneratorResult]) -> str:
    rows = [("Train", "Test", "Accuracy", "ROC AUC")]
    rows.extend(
        (
            result.train_generator,
            result.test_generator,
            f"{result.test_accuracy:.4f}",
            f"{result.test_roc_auc:.4f}",
        )
        for result in results
    )
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
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
