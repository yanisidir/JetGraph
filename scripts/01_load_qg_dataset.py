#!/usr/bin/env python
"""Load a small EnergyFlow qg_jets sample and compute baseline observables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.data.load_qg import (
    DEFAULT_NUM_JETS,
    DEFAULT_OUTPUT_PATH,
    load_qg_jets,
    save_qg_sample,
)
from jetgraph.physics.observables import compute_jet_observables, summarize_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load EnergyFlow qg_jets and save a processed JetGraph sample."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=None,
        help=f"Number of jets to load. Defaults to config value or {DEFAULT_NUM_JETS}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output NPZ path. Defaults to config value or {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory used by EnergyFlow to cache downloaded files.",
    )
    parser.add_argument(
        "--generator",
        choices=("pythia", "herwig"),
        default=None,
        help="Monte Carlo generator used by EnergyFlow.",
    )
    parser.add_argument(
        "--ncol",
        type=int,
        default=None,
        help="Number of per-particle columns to load.",
    )
    parser.add_argument(
        "--with-bc",
        action="store_true",
        help="Use the separate qg_jets dataset that includes bottom/charm jets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})

    num_jets = args.num_jets or int(dataset_cfg.get("num_jets", DEFAULT_NUM_JETS))
    output = resolve_project_path(
        args.output or paths_cfg.get("processed_output", DEFAULT_OUTPUT_PATH)
    )
    cache_dir = resolve_project_path(
        args.cache_dir or dataset_cfg.get("cache_dir", "data/raw/energyflow")
    )
    generator = args.generator or dataset_cfg.get("generator", "pythia")
    ncol = args.ncol or int(dataset_cfg.get("ncol", 4))
    pad = bool(dataset_cfg.get("pad", True))
    with_bc = bool(args.with_bc or dataset_cfg.get("with_bc", False))

    jets, labels = load_qg_jets(
        num_jets=num_jets,
        pad=pad,
        ncol=ncol,
        generator=generator,
        with_bc=with_bc,
        cache_dir=cache_dir,
    )

    observables = compute_jet_observables(jets)
    saved_path = save_qg_sample(jets, labels, output, observables=observables)

    print_dataset_summary(jets, labels)
    print_observable_summary(observables)
    print(f"\nSaved processed sample to: {saved_path}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read the YAML config. Install the project "
            "environment from environment.yml or run `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return config


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def print_dataset_summary(jets, labels) -> None:
    print("Dataset summary")
    print("---------------")
    print(f"jets shape:   {jets.shape}")
    print(f"labels shape: {labels.shape}")

    label_counts = label_distribution(labels)
    print("label distribution:")
    for label, count in label_counts.items():
        name = "gluon" if label == 0 else "quark" if label == 1 else "unknown"
        print(f"  {label} ({name}): {count}")


def print_observable_summary(observables: dict[str, Any]) -> None:
    print("\nObservable summaries")
    print("--------------------")
    for name, values in observables.items():
        stats = summarize_array(values)
        print(
            f"{name:>12}: "
            f"mean={stats['mean']:.4g}, "
            f"std={stats['std']:.4g}, "
            f"min={stats['min']:.4g}, "
            f"median={stats['median']:.4g}, "
            f"max={stats['max']:.4g}"
        )


def label_distribution(labels) -> dict[int, int]:
    import numpy as np

    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


if __name__ == "__main__":
    main()
