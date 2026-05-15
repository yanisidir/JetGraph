#!/usr/bin/env python
"""Build a PyTorch Geometric graph dataset from processed qg_jets arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.graphs.build_graphs import (
    DEFAULT_K,
    FEATURE_MODES,
    RAW_FEATURE_MODE,
    load_graph_dataset,
    save_graph_dataset,
    summarize_graph_dataset,
)


DEFAULT_INPUT = Path("data/processed/qg_jets_sample.npz")
DEFAULT_RAW_OUTPUT = Path("data/processed/qg_graphs_k8.pt")
DEFAULT_RAW_NO_PID_OUTPUT = Path("data/processed/qg_graphs_k8_raw_no_pid.pt")
DEFAULT_PHYSICS_OUTPUT = Path("data/processed/qg_graphs_k8_physics.pt")
DEFAULT_HYBRID_OUTPUT = Path("data/processed/qg_graphs_k8_hybrid.pt")
DEFAULT_HYBRID_NO_PID_OUTPUT = Path("data/processed/qg_graphs_k8_hybrid_no_pid.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PyTorch Geometric particle graphs for JetGraph."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processed NPZ dataset path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PyTorch graph dataset path. Defaults depend on feature mode.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=RAW_FEATURE_MODE,
        help="Node feature mode. Defaults to raw.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of nearest neighbours per particle. Defaults to {DEFAULT_K}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.feature_mode, args.k)

    graphs = load_graph_dataset(
        resolve_project_path(args.input),
        k=args.k,
        feature_mode=args.feature_mode,
    )
    output_path = save_graph_dataset(graphs, resolve_project_path(output))
    summary = summarize_graph_dataset(graphs)

    print("Graph dataset summary")
    print("---------------------")
    print(f"feature mode:  {args.feature_mode}")
    print(f"graphs:        {summary.num_graphs}")
    print(f"average nodes: {summary.average_num_nodes:.2f}")
    print(f"average edges: {summary.average_num_edges:.2f}")
    print(f"saved to:      {output_path}")


def default_output_path(feature_mode: str, k: int) -> Path:
    if k == DEFAULT_K and feature_mode == "raw":
        return DEFAULT_RAW_OUTPUT
    if k == DEFAULT_K and feature_mode == "raw_no_pid":
        return DEFAULT_RAW_NO_PID_OUTPUT
    if k == DEFAULT_K and feature_mode == "physics":
        return DEFAULT_PHYSICS_OUTPUT
    if k == DEFAULT_K and feature_mode == "hybrid":
        return DEFAULT_HYBRID_OUTPUT
    if k == DEFAULT_K and feature_mode == "hybrid_no_pid":
        return DEFAULT_HYBRID_NO_PID_OUTPUT

    suffixes = {
        "raw": "",
        "raw_no_pid": "_raw_no_pid",
        "physics": "_physics",
        "hybrid": "_hybrid",
        "hybrid_no_pid": "_hybrid_no_pid",
    }
    suffix = suffixes[feature_mode]
    return Path(f"data/processed/qg_graphs_k{k}{suffix}.pt")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
