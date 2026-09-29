"""Petit réseau résiduel pour les données 2D jouets."""

import torch
import torch.nn as nn

from .embeddings import TimeClassEmbedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.emb = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return h + self.ff(self.norm(h) + self.emb(emb))


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        data_dim: int = 2,
        hidden: int = 256,
        n_blocks: int = 4,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed = TimeClassEmbedding(hidden, num_classes)
        self.inp = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, data_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embed(t, y)
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(h)
