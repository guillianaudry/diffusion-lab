import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Encodage sinusoïdal du pas de temps (comme les positions d'un transformer)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class TimeClassEmbedding(nn.Module):
    """Embedding de t (+ classe optionnelle, avec un indice supplémentaire pour le jeton ∅)."""

    def __init__(self, emb_dim: int, num_classes: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.time = nn.Sequential(
            SinusoidalEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.label = nn.Embedding(num_classes + 1, emb_dim) if num_classes is not None else None

    def forward(self, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.time(t)
        if self.label is not None:
            if y is None:  # pas de condition fournie : jeton ∅
                y = torch.full_like(t, self.num_classes)
            emb = emb + self.label(y)
        return emb
