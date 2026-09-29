"""diffusion_lab — implémentations pédagogiques de modèles de diffusion."""

from .analytic import GaussianMixtureDenoiser
from .diffusion import GaussianDiffusion
from .schedules import cosine_beta_schedule, get_beta_schedule, linear_beta_schedule

__all__ = [
    "GaussianDiffusion",
    "GaussianMixtureDenoiser",
    "get_beta_schedule",
    "linear_beta_schedule",
    "cosine_beta_schedule",
]
