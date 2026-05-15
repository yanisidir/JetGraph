"""Training and evaluation helpers for the first JetGraph GNN."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from jetgraph.models.gnn import EdgeConvJetClassifier


@dataclass
class EpochMetrics:
    """Metrics reported after one training epoch."""

    epoch: int
    train_loss: float
    val_accuracy: float
    val_auc: float


@dataclass
class EvaluationResult:
    """Graph-level classification metrics and ROC curve arrays."""

    accuracy: float
    roc_auc: float
    classification_report: str
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray


def load_graphs(path: str | Path) -> list[Data]:
    """Load a list of PyTorch Geometric ``Data`` objects from disk."""

    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph dataset not found: {graph_path}. "
            "Run scripts/03_build_graph_dataset.py first."
        )
    return torch.load(graph_path, weights_only=False)


def split_graphs(
    graphs: Sequence[Data],
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Stratify graphs into train, validation, and test sets."""

    from sklearn.model_selection import train_test_split

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to < 1.")

    graphs = list(graphs)
    labels = np.asarray([int(graph.y.view(-1)[0]) for graph in graphs])

    train_graphs, temp_graphs, train_labels, temp_labels = train_test_split(
        graphs,
        labels,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=labels,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_graphs, test_graphs = train_test_split(
        temp_graphs,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_labels,
    )

    return train_graphs, val_graphs, test_graphs


def make_data_loaders(
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyG DataLoaders for train, validation, and test splits."""

    return (
        DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(val_graphs, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_gnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> list[EpochMetrics]:
    """Train the GNN and report validation metrics after each epoch."""

    device = resolve_device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate_gnn(model, val_loader, device=device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_accuracy=val_result.accuracy,
            val_auc=val_result.roc_auc,
        )
        history.append(metrics)
        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val accuracy: {val_result.accuracy:.4f} | "
                f"val AUC: {val_result.roc_auc:.4f}"
            )

    return history


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss per graph."""

    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        num_graphs = labels.numel()
        total_loss += float(loss.detach().cpu()) * num_graphs
        total_graphs += num_graphs

    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate_gnn(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str | None = None,
) -> EvaluationResult:
    """Evaluate a GNN on a graph DataLoader."""

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
        roc_curve,
    )

    device = resolve_device(device)
    model.eval()
    model.to(device)

    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probabilities = torch.softmax(logits, dim=1)

        labels.append(batch.y.view(-1).long().cpu().numpy())
        scores.append(probabilities[:, 1].cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    y_pred = np.concatenate(predictions)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["gluon", "quark"],
            digits=3,
            zero_division=0,
        ),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
    )


def save_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    *,
    history: Sequence[EpochMetrics],
    test_result: EvaluationResult,
    config: dict,
) -> Path:
    """Save the trained model state and lightweight training metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "history": [metrics.__dict__ for metrics in history],
            "test_accuracy": test_result.accuracy,
            "test_roc_auc": test_result.roc_auc,
            "config": config,
        },
        path,
    )
    return path


def plot_roc_curve(result: EvaluationResult, output_path: str | Path) -> Path:
    """Save a ROC curve image for the GNN test-set result."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(
        result.fpr,
        result.tpr,
        linewidth=2,
        label=f"EdgeConv GNN (AUC = {result.roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Quark/Gluon GNN ROC")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def build_model(input_dim: int = 4, hidden_dim: int = 64, dropout: float = 0.1) -> nn.Module:
    """Factory for the default EdgeConv classifier."""

    return EdgeConvJetClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=2,
        dropout=dropout,
    )


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve an explicit device string or choose CUDA/CPU automatically."""

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_matplotlib() -> None:
    """Keep Matplotlib cache files in a writable temporary directory."""

    cache_root = Path(tempfile.gettempdir()) / "jetgraph-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg", force=True)
