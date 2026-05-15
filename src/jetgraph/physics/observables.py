"""Physics-inspired jet observables for padded particle arrays."""

from __future__ import annotations

import numpy as np


def particle_mask(jets: np.ndarray, *, pt_column: int = 0) -> np.ndarray:
    """Return a mask that excludes padded particles with zero transverse momentum."""

    jets = _as_particle_array(jets)
    return jets[..., pt_column] > 0.0


def jet_multiplicity(jets: np.ndarray) -> np.ndarray:
    """Count non-padded particles in each jet."""

    return particle_mask(jets).sum(axis=1)


def total_pt(jets: np.ndarray) -> np.ndarray:
    """Compute scalar constituent pT sum for each jet."""

    jets = _as_particle_array(jets)
    mask = particle_mask(jets)
    return np.where(mask, jets[..., 0], 0.0).sum(axis=1)


def jet_mass(jets: np.ndarray) -> np.ndarray:
    """Approximate jet mass from massless constituent four-vectors.

    The EnergyFlow qg_jets columns are interpreted as ``(pT, eta, phi, pid)``.
    Constituents are treated as massless, so ``E = pT cosh(eta)`` and
    ``pz = pT sinh(eta)``.
    """

    jets = _as_particle_array(jets)
    mask = particle_mask(jets)

    pt = np.where(mask, jets[..., 0], 0.0)
    eta = jets[..., 1]
    phi = jets[..., 2]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = pt * np.cosh(eta)

    jet_energy = energy.sum(axis=1)
    jet_px = px.sum(axis=1)
    jet_py = py.sum(axis=1)
    jet_pz = pz.sum(axis=1)

    mass2 = jet_energy**2 - jet_px**2 - jet_py**2 - jet_pz**2
    return np.sqrt(np.maximum(mass2, 0.0))


def pt_weighted_eta_width(jets: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Compute the pT-weighted RMS width in eta for each jet."""

    jets = _as_particle_array(jets)
    mask = particle_mask(jets)

    pt = np.where(mask, jets[..., 0], 0.0)
    eta = jets[..., 1]
    pt_sum = pt.sum(axis=1)

    mean_eta = (pt * eta).sum(axis=1) / np.maximum(pt_sum, eps)
    variance = (pt * (eta - mean_eta[:, None]) ** 2).sum(axis=1) / np.maximum(
        pt_sum, eps
    )
    return np.sqrt(np.maximum(variance, 0.0))


def pt_weighted_phi_width(jets: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Compute the pT-weighted RMS width in phi for each jet."""

    jets = _as_particle_array(jets)
    mask = particle_mask(jets)

    pt = np.where(mask, jets[..., 0], 0.0)
    phi = jets[..., 2]
    pt_sum = pt.sum(axis=1)

    mean_sin = (pt * np.sin(phi)).sum(axis=1) / np.maximum(pt_sum, eps)
    mean_cos = (pt * np.cos(phi)).sum(axis=1) / np.maximum(pt_sum, eps)
    mean_phi = np.arctan2(mean_sin, mean_cos)

    dphi = wrap_delta_phi(phi - mean_phi[:, None])
    variance = (pt * dphi**2).sum(axis=1) / np.maximum(pt_sum, eps)
    return np.sqrt(np.maximum(variance, 0.0))


def compute_jet_observables(jets: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the baseline observable set used by the initial JetGraph sample."""

    return {
        "multiplicity": jet_multiplicity(jets),
        "total_pt": total_pt(jets),
        "mass": jet_mass(jets),
        "eta_width": pt_weighted_eta_width(jets),
        "phi_width": pt_weighted_phi_width(jets),
    }


def summarize_array(values: np.ndarray) -> dict[str, float]:
    """Return compact summary statistics for a one-dimensional array."""

    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def wrap_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    """Wrap angular differences to the interval [-pi, pi)."""

    return (delta_phi + np.pi) % (2.0 * np.pi) - np.pi


def _as_particle_array(jets: np.ndarray) -> np.ndarray:
    jets = np.asarray(jets)
    if jets.ndim != 3 or jets.shape[-1] < 3:
        raise ValueError(
            "jets must be a dense array with shape (n_jets, n_particles, n_features>=3)"
        )
    return jets
