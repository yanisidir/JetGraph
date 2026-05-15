"""Jet and graph visualization helpers."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jetgraph.graphs.build_graphs import knn_edge_index_eta_phi, remove_padded_particles
from jetgraph.physics.observables import wrap_delta_phi


@dataclass(frozen=True)
class SelectedJet:
    """One selected jet for visualization."""

    index: int
    label: int
    name: str
    particles: np.ndarray

    @property
    def multiplicity(self) -> int:
        return int(self.particles.shape[0])


def load_jet_arrays(input_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dense jets and labels from a processed JetGraph NPZ file."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed jet sample not found: {path}")

    with np.load(path) as data:
        if "jets" not in data or "labels" not in data:
            raise KeyError("Processed sample must contain 'jets' and 'labels'.")
        return np.asarray(data["jets"]), np.asarray(data["labels"]).astype(int)


def select_representative_jets(
    jets: np.ndarray,
    labels: np.ndarray,
) -> dict[str, SelectedJet]:
    """Select one quark and one gluon jet with near-median multiplicity."""

    return {
        "gluon": select_representative_jet(jets, labels, label=0, name="gluon"),
        "quark": select_representative_jet(jets, labels, label=1, name="quark"),
    }


def select_representative_jet(
    jets: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    name: str,
) -> SelectedJet:
    """Select a class member closest to that class's median multiplicity."""

    indices = np.flatnonzero(labels == label)
    if indices.size == 0:
        raise ValueError(f"No jets found for label {label}.")

    multiplicities = np.asarray(
        [remove_padded_particles(jets[index]).shape[0] for index in indices]
    )
    target = np.median(multiplicities)
    local_index = int(np.argmin(np.abs(multiplicities - target)))
    jet_index = int(indices[local_index])

    return SelectedJet(
        index=jet_index,
        label=label,
        name=name,
        particles=remove_padded_particles(jets[jet_index]),
    )


def plot_jet_display(jet: SelectedJet, output_path: str | Path) -> Path:
    """Save an eta-phi particle-cloud display for one jet."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    pt = jet.particles[:, 0]
    eta = jet.particles[:, 1]
    phi = jet.particles[:, 2]

    scatter = ax.scatter(
        eta,
        phi,
        s=marker_sizes(pt),
        c=pt,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Particle pT")

    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    ax.set_title(f"{jet.name.capitalize()} jet, multiplicity = {jet.multiplicity}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_centered_jet_display(jet: SelectedJet, output_path: str | Path) -> Path:
    """Save a pT-centered delta_eta-delta_phi particle-cloud display."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pt = jet.particles[:, 0]
    delta_eta, delta_phi = centered_eta_phi(jet.particles)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    scatter = ax.scatter(
        delta_eta,
        delta_phi,
        s=marker_sizes(pt),
        c=pt,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Particle pT")

    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel("delta_eta")
    ax.set_ylabel("delta_phi")
    ax.set_title(
        f"Centered {jet.name} jet, multiplicity = {jet.multiplicity}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_jet_graph(jet: SelectedJet, output_path: str | Path, *, k: int = 8) -> Path:
    """Save an eta-phi display with kNN graph edges overlaid."""

    configure_matplotlib()
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pt = jet.particles[:, 0]
    eta = jet.particles[:, 1]
    phi = jet.particles[:, 2]
    edge_index = knn_edge_index_eta_phi(jet.particles[:, 1:3], k=k).numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for source, target in edge_index.T:
        ax.plot(
            [eta[source], eta[target]],
            [phi[source], phi[target]],
            color="0.45",
            linewidth=0.5,
            alpha=0.35,
            zorder=1,
        )

    scatter = ax.scatter(
        eta,
        phi,
        s=marker_sizes(pt),
        c=pt,
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    fig.colorbar(scatter, ax=ax, label="Particle pT")

    ax.set_xlabel("eta")
    ax.set_ylabel("phi")
    ax.set_title(
        f"{jet.name.capitalize()} jet kNN graph, k = {k}, "
        f"multiplicity = {jet.multiplicity}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def centered_eta_phi(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eta and phi offsets from the pT-weighted jet axis."""

    pt = particles[:, 0]
    eta = particles[:, 1]
    phi = particles[:, 2]
    total_pt = np.maximum(pt.sum(), 1e-12)

    jet_eta = (pt * eta).sum() / total_pt
    mean_sin = (pt * np.sin(phi)).sum() / total_pt
    mean_cos = (pt * np.cos(phi)).sum() / total_pt
    jet_phi = np.arctan2(mean_sin, mean_cos)

    return eta - jet_eta, wrap_delta_phi(phi - jet_phi)


def marker_sizes(pt: np.ndarray) -> np.ndarray:
    """Map particle pT to visually stable scatter marker areas."""

    pt = np.asarray(pt, dtype=float)
    if pt.size == 0:
        return pt
    scaled = np.sqrt(pt / np.max(pt))
    return 25.0 + 275.0 * scaled


def configure_matplotlib() -> None:
    """Keep Matplotlib cache files in a writable temporary directory."""

    cache_root = Path(tempfile.gettempdir()) / "jetgraph-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg", force=True)
