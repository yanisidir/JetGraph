"""Utilities for loading and saving the EnergyFlow quark/gluon jet dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_NUM_JETS = 10_000
DEFAULT_OUTPUT_PATH = Path("data/processed/qg_jets_sample.npz")


def load_qg_jets(
    num_jets: int = DEFAULT_NUM_JETS,
    *,
    pad: bool = True,
    ncol: int = 4,
    generator: str = "pythia",
    with_bc: bool = False,
    cache_dir: str | Path = "data/raw/energyflow",
) -> tuple[np.ndarray, np.ndarray]:
    """Load quark/gluon jets from EnergyFlow.

    Parameters
    ----------
    num_jets:
        Number of jets to load. EnergyFlow accepts ``-1`` for the full dataset.
    pad:
        If true, pad jets to a common number of particles. The observables in
        this project expect padded, dense arrays.
    ncol:
        Number of per-particle columns to retain. The default EnergyFlow order is
        ``(pT, rapidity/eta, phi, particle_id)``.
    generator:
        Monte Carlo generator, usually ``"pythia"`` or ``"herwig"``.
    with_bc:
        Whether to use the separate dataset that includes bottom and charm jets.
    cache_dir:
        Directory where EnergyFlow stores downloaded dataset files.
    """

    if num_jets == 0 or num_jets < -1:
        raise ValueError("num_jets must be positive, or -1 to load all events")

    try:
        from energyflow.datasets import qg_jets
    except ImportError as exc:
        raise ImportError(
            "EnergyFlow is required to load qg_jets. Install the project "
            "environment or run `pip install energyflow`."
        ) from exc

    jets, labels = qg_jets.load(
        num_data=num_jets,
        pad=pad,
        ncol=ncol,
        generator=generator,
        with_bc=with_bc,
        cache_dir=str(cache_dir),
    )

    return np.asarray(jets), np.asarray(labels)


def save_qg_sample(
    jets: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    observables: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Save processed jets, labels, and optional observables to a compressed NPZ."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "jets": np.asarray(jets),
        "labels": np.asarray(labels),
    }

    if observables:
        arrays.update(
            {name: np.asarray(values) for name, values in observables.items()}
        )

    np.savez_compressed(path, **arrays)
    return path
