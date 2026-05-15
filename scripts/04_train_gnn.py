#!/usr/bin/env python
"""Train the first EdgeConv GNN for JetGraph quark/gluon tagging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.training.train_gnn import (
    build_model,
    evaluate_gnn,
    load_graphs,
    make_data_loaders,
    plot_roc_curve,
    resolve_device,
    save_checkpoint,
    set_random_seed,
    split_graphs,
    train_gnn,
)


DEFAULT_INPUT = Path("data/processed/qg_graphs_k8.pt")
DEFAULT_ROC_OUTPUT = Path("figures/gnn_roc.png")
DEFAULT_CHECKPOINT = Path("models/gnn_edgeconv_k8.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple EdgeConv GNN on JetGraph particle graphs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Graph dataset path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
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
        help="Hidden dimension for EdgeConv and classifier layers.",
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
        help="Validation fraction of the full graph dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test fraction of the full graph dataset.",
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
        help="Optional torch device, for example 'cpu', 'cuda', or 'mps'.",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=DEFAULT_ROC_OUTPUT,
        help=f"Path for the test ROC curve PNG. Defaults to {DEFAULT_ROC_OUTPUT}.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path for the model checkpoint. Defaults to {DEFAULT_CHECKPOINT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)
    device = resolve_device(args.device)

    graphs = load_graphs(resolve_project_path(args.input))
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

    input_dim = graphs[0].x.shape[1]
    model = build_model(input_dim=input_dim, hidden_dim=args.hidden_dim)

    print("GNN training setup")
    print("------------------")
    print(f"graphs: {len(graphs)}")
    print(f"split:  train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
    print(f"device: {device}")
    print()

    history = train_gnn(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    test_result = evaluate_gnn(model, test_loader, device=device)
    roc_path = plot_roc_curve(test_result, resolve_project_path(args.roc_output))
    checkpoint_path = save_checkpoint(
        model,
        resolve_project_path(args.checkpoint),
        history=history,
        test_result=test_result,
        config={
            "input": str(resolve_project_path(args.input)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "device": str(device),
        },
    )

    print("\nTest set results")
    print("----------------")
    print(f"accuracy: {test_result.accuracy:.4f}")
    print(f"ROC AUC:  {test_result.roc_auc:.4f}")
    print()
    print(test_result.classification_report)
    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved checkpoint: {checkpoint_path}")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
