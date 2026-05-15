#!/usr/bin/env python
"""Create eta-phi jet displays and kNN graph visualizations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jetgraph.evaluation.visualize_jets import (  # noqa: E402
    load_jet_arrays,
    plot_centered_jet_display,
    plot_jet_display,
    plot_jet_graph,
    select_representative_jets,
)


DEFAULT_INPUT = Path("data/processed/qg_jets_sample.npz")
DEFAULT_K = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize representative quark and gluon jets in eta-phi space."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Processed NPZ jet sample. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"kNN connectivity for graph displays. Defaults to {DEFAULT_K}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jets, labels = load_jet_arrays(resolve_project_path(args.input))
    selected = select_representative_jets(jets, labels)

    outputs = [
        plot_jet_display(
            selected["quark"],
            resolve_project_path("figures/jet_display_quark.png"),
        ),
        plot_jet_display(
            selected["gluon"],
            resolve_project_path("figures/jet_display_gluon.png"),
        ),
        plot_centered_jet_display(
            selected["quark"],
            resolve_project_path("figures/jet_display_quark_centered.png"),
        ),
        plot_centered_jet_display(
            selected["gluon"],
            resolve_project_path("figures/jet_display_gluon_centered.png"),
        ),
        plot_jet_graph(
            selected["quark"],
            resolve_project_path("figures/jet_graph_quark_k8.png"),
            k=args.k,
        ),
        plot_jet_graph(
            selected["gluon"],
            resolve_project_path("figures/jet_graph_gluon_k8.png"),
            k=args.k,
        ),
    ]

    print("Jet visualization summary")
    print("-------------------------")
    for name, jet in selected.items():
        print(f"{name:>5}: index={jet.index}, multiplicity={jet.multiplicity}")
    print()
    for output in outputs:
        print(f"saved: {output}")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
