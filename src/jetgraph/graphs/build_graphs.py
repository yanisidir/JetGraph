"""Build PyTorch Geometric graphs from particle-level jets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from jetgraph.physics.observables import wrap_delta_phi


DEFAULT_K = 8
RAW_FEATURE_MODE = "raw"
RAW_NO_PID_FEATURE_MODE = "raw_no_pid"
PHYSICS_FEATURE_MODE = "physics"
HYBRID_FEATURE_MODE = "hybrid"
HYBRID_NO_PID_FEATURE_MODE = "hybrid_no_pid"
FEATURE_MODES = (
    RAW_FEATURE_MODE,
    RAW_NO_PID_FEATURE_MODE,
    PHYSICS_FEATURE_MODE,
    HYBRID_FEATURE_MODE,
    HYBRID_NO_PID_FEATURE_MODE,
)
RAW_NODE_FEATURE_NAMES = ("pt", "eta", "phi", "pid")
RAW_NO_PID_NODE_FEATURE_NAMES = ("pt", "eta", "phi")
PHYSICS_NODE_FEATURE_NAMES = (
    "log_pt",
    "pt_fraction",
    "delta_eta",
    "delta_phi",
    "delta_r",
    "pid",
)
HYBRID_NODE_FEATURE_NAMES = (
    "pt",
    "log_pt",
    "pt_fraction",
    "eta",
    "phi",
    "delta_eta",
    "delta_phi",
    "delta_r",
    "pid",
)
HYBRID_NO_PID_NODE_FEATURE_NAMES = (
    "pt",
    "log_pt",
    "pt_fraction",
    "eta",
    "phi",
    "delta_eta",
    "delta_phi",
    "delta_r",
)
REQUIRED_INPUT_COLUMNS = len(RAW_NODE_FEATURE_NAMES)


@dataclass(frozen=True)
class GraphDatasetSummary:
    """Compact summary for a graph dataset."""

    num_graphs: int
    average_num_nodes: float
    average_num_edges: float


def load_graph_dataset(
    input_path: str | Path,
    *,
    k: int = DEFAULT_K,
    feature_mode: str = RAW_FEATURE_MODE,
) -> list[Data]:
    """Load a processed JetGraph NPZ file and convert all jets to graphs."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run scripts/01_load_qg_dataset.py first."
        )

    with np.load(path) as data:
        if "jets" not in data or "labels" not in data:
            raise KeyError("Processed dataset must contain 'jets' and 'labels' arrays.")

        jets = np.asarray(data["jets"])
        labels = np.asarray(data["labels"])

    return build_graph_dataset(jets, labels, k=k, feature_mode=feature_mode)


def build_graph_dataset(
    jets: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = DEFAULT_K,
    feature_mode: str = RAW_FEATURE_MODE,
) -> list[Data]:
    """Convert dense padded jets and labels into PyG ``Data`` objects."""

    validate_feature_mode(feature_mode)
    jets = np.asarray(jets)
    labels = np.asarray(labels).reshape(-1)

    if jets.ndim != 3 or jets.shape[-1] < REQUIRED_INPUT_COLUMNS:
        raise ValueError(
            "jets must have shape (n_jets, n_particles, n_features>=4)."
        )
    if jets.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Number of jets ({jets.shape[0]}) does not match labels ({labels.shape[0]})."
        )

    return [
        build_particle_graph(jet, label, k=k, feature_mode=feature_mode)
        for jet, label in zip(jets, labels, strict=True)
    ]


def build_particle_graph(
    jet: np.ndarray,
    label: int | float,
    *,
    k: int = DEFAULT_K,
    feature_mode: str = RAW_FEATURE_MODE,
) -> Data:
    """Build one particle-level graph after removing padded constituents."""

    validate_feature_mode(feature_mode)
    particles = remove_padded_particles(jet)
    node_features = build_node_features(particles, feature_mode=feature_mode)

    x = torch.as_tensor(node_features, dtype=torch.float32)
    edge_index = knn_edge_index_eta_phi(particles[:, 1:3], k=k)
    y = torch.as_tensor([int(label)], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0])


def remove_padded_particles(jet: np.ndarray) -> np.ndarray:
    """Keep the node features for particles with positive transverse momentum."""

    jet = np.asarray(jet)
    if jet.ndim != 2 or jet.shape[-1] < REQUIRED_INPUT_COLUMNS:
        raise ValueError("jet must have shape (n_particles, n_features>=4).")

    particles = jet[jet[:, 0] > 0.0, :REQUIRED_INPUT_COLUMNS]
    return np.asarray(particles, dtype=np.float32)


def build_node_features(
    particles: np.ndarray,
    *,
    feature_mode: str = RAW_FEATURE_MODE,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build raw or physics-motivated node features for one unpadded jet."""

    validate_feature_mode(feature_mode)
    particles = np.asarray(particles, dtype=np.float32)

    if feature_mode == RAW_FEATURE_MODE:
        return particles
    if feature_mode == RAW_NO_PID_FEATURE_MODE:
        return particles[:, : len(RAW_NO_PID_NODE_FEATURE_NAMES)]

    if particles.size == 0:
        return np.empty((0, len(node_feature_names(feature_mode))), dtype=np.float32)

    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]
    pid = particles[:, 3]
    total_pt = np.maximum(pt.sum(), eps)

    jet_eta = (pt * eta).sum() / total_pt
    mean_sin = (pt * np.sin(phi)).sum() / total_pt
    mean_cos = (pt * np.cos(phi)).sum() / total_pt
    jet_phi = np.arctan2(mean_sin, mean_cos)

    delta_eta = eta - jet_eta
    delta_phi = wrap_delta_phi(phi - jet_phi)
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
    log_pt = np.log(np.maximum(pt, eps))
    pt_fraction = pt / total_pt

    if feature_mode == HYBRID_FEATURE_MODE:
        return np.column_stack(
            [
                pt,
                log_pt,
                pt_fraction,
                eta,
                phi,
                delta_eta,
                delta_phi,
                delta_r,
                pid,
            ]
        ).astype(np.float32)

    if feature_mode == HYBRID_NO_PID_FEATURE_MODE:
        return np.column_stack(
            [
                pt,
                log_pt,
                pt_fraction,
                eta,
                phi,
                delta_eta,
                delta_phi,
                delta_r,
            ]
        ).astype(np.float32)

    return np.column_stack(
        [
            log_pt,
            pt_fraction,
            delta_eta,
            delta_phi,
            delta_r,
            pid,
        ]
    ).astype(np.float32)


def node_feature_names(feature_mode: str = RAW_FEATURE_MODE) -> tuple[str, ...]:
    """Return node feature names for the selected feature mode."""

    validate_feature_mode(feature_mode)
    if feature_mode == RAW_FEATURE_MODE:
        return RAW_NODE_FEATURE_NAMES
    if feature_mode == RAW_NO_PID_FEATURE_MODE:
        return RAW_NO_PID_NODE_FEATURE_NAMES
    if feature_mode == HYBRID_FEATURE_MODE:
        return HYBRID_NODE_FEATURE_NAMES
    if feature_mode == HYBRID_NO_PID_FEATURE_MODE:
        return HYBRID_NO_PID_NODE_FEATURE_NAMES
    return PHYSICS_NODE_FEATURE_NAMES


def validate_feature_mode(feature_mode: str) -> None:
    if feature_mode not in FEATURE_MODES:
        allowed = ", ".join(FEATURE_MODES)
        raise ValueError(f"feature_mode must be one of: {allowed}")


def knn_edge_index_eta_phi(coords: np.ndarray, *, k: int = DEFAULT_K) -> torch.Tensor:
    """Build directed kNN edges from eta-phi coordinates.

    Phi differences are wrapped to ``[-pi, pi)`` before distances are computed.
    For jets with one or zero particles, the returned edge set is empty.
    """

    if k < 0:
        raise ValueError("k must be non-negative.")

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_particles, 2).")

    num_nodes = coords.shape[0]
    if num_nodes <= 1 or k == 0:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k, num_nodes - 1)
    eta = coords[:, 0]
    phi = coords[:, 1]

    delta_eta = eta[:, None] - eta[None, :]
    delta_phi = wrap_delta_phi(phi[:, None] - phi[None, :])
    distances = delta_eta**2 + delta_phi**2
    np.fill_diagonal(distances, np.inf)

    nearest = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
    row_index = np.arange(num_nodes)[:, None]
    nearest_order = np.argsort(distances[row_index, nearest], axis=1)
    nearest = nearest[row_index, nearest_order]

    sources = np.repeat(np.arange(num_nodes), k_eff)
    targets = nearest.reshape(-1)
    edge_index = np.stack([sources, targets], axis=0)

    return torch.as_tensor(edge_index, dtype=torch.long)


def save_graph_dataset(graphs: list[Data], output_path: str | Path) -> Path:
    """Save a list of PyG graph objects with ``torch.save``."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)
    return path


def summarize_graph_dataset(graphs: list[Data]) -> GraphDatasetSummary:
    """Summarize graph count, node count, and edge count."""

    if not graphs:
        return GraphDatasetSummary(
            num_graphs=0,
            average_num_nodes=0.0,
            average_num_edges=0.0,
        )

    node_counts = np.asarray([graph.num_nodes for graph in graphs], dtype=float)
    edge_counts = np.asarray(
        [graph.edge_index.shape[1] for graph in graphs],
        dtype=float,
    )
    return GraphDatasetSummary(
        num_graphs=len(graphs),
        average_num_nodes=float(node_counts.mean()),
        average_num_edges=float(edge_counts.mean()),
    )
