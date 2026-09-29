from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    """Moyenne mobile exponentielle des poids (notes/01_ddpm.md, section 9)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.detach(), 1.0 - self.decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """Enregistre un lot d'images dans [-1, 1] sous forme de grille PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = ((images.detach().cpu().clamp(-1, 1) + 1) / 2)
    B, C, H, W = images.shape
    ncol = nrow
    nrows = (B + ncol - 1) // ncol
    grid = torch.ones(C, nrows * (H + 2), ncol * (W + 2))
    for k in range(B):
        r, c = divmod(k, ncol)
        grid[:, r * (H + 2) + 1 : r * (H + 2) + 1 + H, c * (W + 2) + 1 : c * (W + 2) + 1 + W] = images[k]
    arr = grid.permute(1, 2, 0).numpy()
    plt.imsave(path, arr.squeeze(-1) if C == 1 else arr, cmap="gray" if C == 1 else None)
