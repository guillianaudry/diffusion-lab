"""Plannings de bruit (beta_t). Voir notes/01_ddpm.md, section 6.

Toutes les fonctions renvoient un tenseur float64 de taille T, indexé de 0 à T-1
(l'indice 0 du code correspond au pas t=1 des notes).
"""

import math

import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Planning linéaire de DDPM, remis à l'échelle pour que T != 1000 reste comparable."""
    scale = 1000.0 / T
    return torch.linspace(scale * beta_start, scale * beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Planning cosinus de Improved DDPM (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=max_beta)


SCHEDULES = {
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule,
}


def get_beta_schedule(name: str, T: int) -> torch.Tensor:
    if name not in SCHEDULES:
        raise ValueError(f"Planning inconnu '{name}'. Choix : {list(SCHEDULES)}")
    return SCHEDULES[name](T)
