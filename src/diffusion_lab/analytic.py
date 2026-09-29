"""Débruiteur optimal exact pour un mélange de gaussiennes isotropes.

Si x_0 ~ Σ_k π_k N(μ_k, s² I), alors sous le processus direct
    x_t | k ~ N(√ᾱ_t μ_k, (ᾱ_t s² + 1 − ᾱ_t) I),
donc le score de p_t, et le bruit optimal ε*(x_t, t) = −√(1−ᾱ_t) ∇log p_t(x_t),
ont une forme fermée (notes/01_ddpm.md, section 8).

Utilité :
  * tester les échantillonneurs sans aucun entraînement (tests/test_samplers.py) ;
  * étudier la guidance sur un cas où tout est calculable (notes/04_guidance.md).

Le module respecte l'interface des réseaux : forward(x_t, t, y) -> ε.
L'indice de classe num_classes (= K) est le jeton ∅ (mélange complet).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianMixtureDenoiser(nn.Module):
    def __init__(
        self,
        alphas_cumprod: torch.Tensor,
        means: torch.Tensor,
        std: float,
        weights: torch.Tensor | None = None,
    ):
        super().__init__()
        K = means.shape[0]
        if weights is None:
            weights = torch.full((K,), 1.0 / K)
        self.num_classes = K
        self.std = float(std)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("means", means.float())
        self.register_buffer("log_weights", weights.float().log())

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        ab = self.alphas_cumprod[t].unsqueeze(1)                        # (B, 1)
        mu_t = ab.sqrt().unsqueeze(1) * self.means.unsqueeze(0)          # (B, K, D)
        var_t = ab * self.std**2 + (1 - ab)                              # (B, 1)
        sq_dist = ((x_t.unsqueeze(1) - mu_t) ** 2).sum(-1)               # (B, K)
        logits = self.log_weights.unsqueeze(0) - sq_dist / (2 * var_t)
        if y is not None:
            # Conditionné à la classe y : on ne garde que la composante y (sauf jeton ∅).
            conditioned = y < self.num_classes
            onehot = torch.zeros_like(logits, dtype=torch.bool)
            rows = conditioned.nonzero(as_tuple=True)[0]
            onehot[rows, y[rows]] = True
            mask = onehot | ~conditioned.unsqueeze(1)
            logits = logits.masked_fill(~mask, float("-inf"))
        resp = torch.softmax(logits, dim=1)                              # (B, K)
        mean_t = (resp.unsqueeze(-1) * mu_t).sum(1)                      # (B, D)
        score = -(x_t - mean_t) / var_t
        return -(1 - ab).sqrt() * score

    def sample_data(self, n: int, y: torch.Tensor | None = None, generator=None) -> torch.Tensor:
        """Échantillons exacts de la loi des données (ou d'une composante)."""
        if y is None:
            y = torch.multinomial(self.log_weights.exp(), n, replacement=True, generator=generator)
        noise = torch.randn(n, self.means.shape[1], generator=generator, device=self.means.device)
        return self.means[y] + self.std * noise
