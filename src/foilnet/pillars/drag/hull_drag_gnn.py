"""A GraphSAGE encoder + global pooling predicting total drag from hull surface geometry."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool


class HullDragGNN(nn.Module):
    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.head(pooled)
