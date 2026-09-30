"""Diffusion gaussienne discrète : DDPM, DDIM et classifier-free guidance.

Notes associées :
    notes/01_ddpm.md                  processus direct, perte, échantillonneur DDPM
    notes/03_ddim_echantillonnage.md  échantillonneur DDIM
    notes/04_guidance.md              classifier-free guidance

Convention d'indices : les tableaux sont indexés de 0 à T-1 ; l'indice i du code
correspond au pas t = i + 1 des notes.

Interface attendue du réseau :
    model(x_t, t, y) -> tenseur de même forme que x_t
        t : LongTensor (B,) d'indices dans [0, T-1]
        y : None, ou LongTensor (B,) de classes dans [0, num_classes],
            l'indice num_classes étant le jeton « sans condition » (∅) de la CFG.
    model.num_classes : int ou None
"""

from __future__ import annotations

import torch
import torch.nn as nn

PREDICTIONS = ("eps", "x0", "v")


def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Prend a[t] et le remet en forme (B, 1, 1, ...) pour la diffusion sur x."""
    return a[t].reshape(t.shape[0], *([1] * (x.ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, betas: torch.Tensor, prediction: str = "eps"):
        super().__init__()
        if prediction not in PREDICTIONS:
            raise ValueError(f"prediction doit être dans {PREDICTIONS}")
        self.prediction = prediction
        self.T = int(betas.shape[0])

        # Calculs en float64 puis stockage en float32 (précision des produits cumulés).
        betas = betas.to(torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        def buf(name: str, value: torch.Tensor) -> None:
            self.register_buffer(name, value.to(torch.float32), persistent=False)

        buf("betas", betas)
        buf("alphas_cumprod", alphas_cumprod)
        buf("alphas_cumprod_prev", alphas_cumprod_prev)
        buf("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        buf("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        # Postérieur q(x_{t-1} | x_t, x_0) — notes/01_ddpm.md, section 2.
        buf("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        buf("posterior_mean_coef1", betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod))
        buf("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # Processus direct
    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise
        )

    # ------------------------------------------------------------------
    # Conversions entre paramétrisations — notes/01_ddpm.md, section 7
    # ------------------------------------------------------------------
    def training_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.prediction == "eps":
            return noise
        if self.prediction == "x0":
            return x0
        a = _extract(self.sqrt_alphas_cumprod, t, x0)
        s = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return a * noise - s * x0  # v

    def predict_x0_and_eps(self, model_out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        a = _extract(self.sqrt_alphas_cumprod, t, x_t)
        s = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        if self.prediction == "eps":
            eps = model_out
            x0 = (x_t - s * eps) / a
        elif self.prediction == "x0":
            x0 = model_out
            eps = (x_t - a * x0) / s
        else:  # v
            x0 = a * x_t - s * model_out
            eps = s * x_t + a * model_out
        return x0, eps

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
        p_uncond: float = 0.0,
    ):
        """Perte L_simple par exemple : renvoie (perte de chaque exemple (B,), pas t tirés (B,)).

        Utile pour suivre la perte en fonction du niveau de bruit.
        Si y est fourni et p_uncond > 0, la condition est remplacée par le jeton ∅
        avec probabilité p_uncond (entraînement pour la classifier-free guidance).
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if y is not None and p_uncond > 0:
            drop = torch.rand(B, device=x0.device) < p_uncond
            y = torch.where(drop, torch.full_like(y, model.num_classes), y)
        out = model(x_t, t, y)
        per_example = ((out - self.training_target(x0, noise, t)) ** 2).flatten(1).mean(1)
        return per_example, t

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
        p_uncond: float = 0.0,
    ) -> torch.Tensor:
        """Perte L_simple moyenne (régression MSE sur la cible choisie)."""
        per_example, _ = self.training_losses(model, x0, y, p_uncond)
        return per_example.mean()

    # ------------------------------------------------------------------
    # Prédictions (avec guidance optionnelle)
    # ------------------------------------------------------------------
    def model_predictions(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: bool = False,
    ):
        """Renvoie (x0_hat, eps_hat).

        Classifier-free guidance (notes/04_guidance.md) :
            out = out(∅) + s · (out(c) − out(∅)).
        La combinaison est linéaire, donc valable quelle que soit la paramétrisation.
        """
        if y is not None and guidance_scale != 1.0:
            y_null = torch.full_like(y, model.num_classes)
            out = model(torch.cat([x_t, x_t]), torch.cat([t, t]), torch.cat([y, y_null]))
            out_cond, out_uncond = out.chunk(2)
            out = out_uncond + guidance_scale * (out_cond - out_uncond)
        else:
            out = model(x_t, t, y)
        x0, eps = self.predict_x0_and_eps(out, x_t, t)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
            a = _extract(self.sqrt_alphas_cumprod, t, x_t)
            s = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
            eps = (x_t - a * x0) / s
        return x0, eps

    # ------------------------------------------------------------------
    # Échantillonnage DDPM — notes/01_ddpm.md, section 5
    # ------------------------------------------------------------------
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        i: int,
        var_type: str = "posterior",
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_{t-1} ~ N(μ̃_t(x_t, x̂_0), σ_t² I), avec σ_t² = β̃_t ('posterior') ou β_t ('beta')."""
        mean = self.posterior_mean_coef1[i] * x0_hat + self.posterior_mean_coef2[i] * x_t
        if i == 0:
            return mean
        var = self.posterior_variance[i] if var_type == "posterior" else self.betas[i]
        if noise is None:
            noise = torch.randn_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: bool = False,
        var_type: str = "posterior",
        x_T: torch.Tensor | None = None,
        return_trajectory: bool = False,
        return_pred_x0: bool = False,
    ):
        """Échantillonnage DDPM.

        return_trajectory : renvoie aussi [x_T, x_{T-1}, ..., x_0] (T + 1 éléments).
        return_pred_x0    : renvoie aussi les x̂_0 prédits à chaque pas (T éléments,
                            le k-ième est prédit à partir de trajectory[k]).
        """
        device = self.betas.device
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
        trajectory, pred_x0 = [x], []
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0_hat, _ = self.model_predictions(model, x, t, y, guidance_scale, clip_x0)
            if return_pred_x0:
                pred_x0.append(x0_hat)
            x = self.ddpm_step(x, x0_hat, i, var_type)
            if return_trajectory:
                trajectory.append(x)
        return self._pack(x, trajectory, pred_x0, return_trajectory, return_pred_x0)

    @staticmethod
    def _pack(x, trajectory, pred_x0, return_trajectory, return_pred_x0):
        out = (x,)
        if return_trajectory:
            out += (trajectory,)
        if return_pred_x0:
            out += (pred_x0,)
        return out if len(out) > 1 else x

    # ------------------------------------------------------------------
    # Échantillonnage DDIM — notes/03_ddim_echantillonnage.md
    # ------------------------------------------------------------------
    def ddim_timesteps(self, steps: int) -> list[int]:
        """Sous-suite décroissante de `steps` indices régulièrement espacés dans [0, T-1]."""
        ts = torch.linspace(0, self.T - 1, steps).round().long().unique()
        return ts.flip(0).tolist()

    def ddim_step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        eps_hat: torch.Tensor,
        i: int,
        i_prev: int,
        eta: float = 0.0,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Un pas DDIM de l'indice i vers i_prev (i_prev = -1 signifie x_0)."""
        ab = self.alphas_cumprod[i]
        ab_prev = self.alphas_cumprod[i_prev] if i_prev >= 0 else torch.ones_like(ab)
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        direction = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_hat
        x_prev = ab_prev.sqrt() * x0_hat + direction
        if eta > 0 and i_prev >= 0:
            if noise is None:
                noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        steps: int = 50,
        eta: float = 0.0,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: bool = False,
        x_T: torch.Tensor | None = None,
        return_trajectory: bool = False,
        return_pred_x0: bool = False,
    ):
        """Échantillonnage DDIM (mêmes options de retour que p_sample_loop).

        Les indices de temps parcourus sont donnés par self.ddim_timesteps(steps) :
        trajectory[k] et pred_x0[k] correspondent à l'indice ddim_timesteps(steps)[k].
        """
        device = self.betas.device
        x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
        trajectory, pred_x0 = [x], []
        ts = self.ddim_timesteps(steps)
        for i, i_prev in zip(ts, ts[1:] + [-1]):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x0_hat, eps_hat = self.model_predictions(model, x, t, y, guidance_scale, clip_x0)
            if return_pred_x0:
                pred_x0.append(x0_hat)
            x = self.ddim_step(x, x0_hat, eps_hat, i, i_prev, eta)
            if return_trajectory:
                trajectory.append(x)
        return self._pack(x, trajectory, pred_x0, return_trajectory, return_pred_x0)
