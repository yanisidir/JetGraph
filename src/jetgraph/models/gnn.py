"""Simple EdgeConv GNN for quark/gluon jet tagging."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import EdgeConv, global_mean_pool


class EdgeConvJetClassifier(nn.Module):
    """Two-layer EdgeConv classifier for graph-level jet labels."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.edge_conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.edge_conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.edge_conv1(x, edge_index)
        x = torch.relu(x)
        x = self.edge_conv2(x, edge_index)
        x = torch.relu(x)
        graph_embedding = global_mean_pool(x, batch)
        return self.classifier(graph_embedding)
