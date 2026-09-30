"""Validation des échantillonneurs avec le débruiteur optimal exact (aucun entraînement).

Les données suivent un mélange de 2 gaussiennes en 2D. Avec le vrai ε*, DDPM et DDIM
doivent retrouver les poids, moyennes et écarts-types du mélange.
"""

import pytest
import torch

from diffusion_lab import GaussianDiffusion, GaussianMixtureDenoiser, get_beta_schedule

MEANS = torch.tensor([[-1.5, 0.0], [1.5, 0.5]])
STD = 0.3


@pytest.fixture(scope="module")
def setup():
    d = GaussianDiffusion(get_beta_schedule("linear", 1000))
    oracle = GaussianMixtureDenoiser(d.alphas_cumprod, MEANS, STD)
    return d, oracle


def _check_mixture(x, tol_mean=0.05, tol_std=0.05, tol_frac=0.05):
    comp = (x[:, 0] > 0).long()
    frac = comp.float().mean()
    assert abs(frac - 0.5) < tol_frac
    for k in range(2):
        xk = x[comp == k]
        assert torch.allclose(xk.mean(0), MEANS[k], atol=tol_mean), (k, xk.mean(0))
        assert torch.allclose(xk.std(0), torch.full((2,), STD), atol=tol_std), (k, xk.std(0))


def test_ddpm_sampler_recovers_mixture(setup):
    torch.manual_seed(0)
    d, oracle = setup
    x = d.p_sample_loop(oracle, (4000, 2))
    _check_mixture(x)


@pytest.mark.parametrize("steps,eta", [(50, 0.0), (100, 1.0)])
def test_ddim_sampler_recovers_mixture(setup, steps, eta):
    torch.manual_seed(0)
    d, oracle = setup
    x = d.ddim_sample_loop(oracle, (4000, 2), steps=steps, eta=eta)
    _check_mixture(x)


def test_ddim_few_steps_discretization_bias(setup):
    """Même avec le score exact, 20 pas d'Euler (DDIM η=0) contractent la distribution :
    l'écart-type obtenu (≈ 0,24) est inférieur au vrai (0,30). C'est l'erreur de
    discrétisation de l'ODE (notes/03_ddim_echantillonnage.md, section 4)."""
    torch.manual_seed(0)
    d, oracle = setup
    x = d.ddim_sample_loop(oracle, (4000, 2), steps=20, eta=0.0)
    _check_mixture(x, tol_mean=0.1, tol_std=0.1)
    assert x[x[:, 0] > 0].std(0)[1] < STD - 0.03


def test_ddim_eta1_matches_ddpm_step(setup):
    """Avec η = 1 et des pas consécutifs, DDIM coïncide exactement avec DDPM (σ² = β̃)."""
    d, oracle = setup
    x = torch.randn(32, 2)
    noise = torch.randn(32, 2)
    for i in [1, 50, 400, 999]:
        t = torch.full((32,), i)
        x0_hat, eps_hat = d.model_predictions(oracle, x, t)
        a = d.ddpm_step(x, x0_hat, i, noise=noise)
        b = d.ddim_step(x, x0_hat, eps_hat, i, i - 1, eta=1.0, noise=noise)
        assert torch.allclose(a, b, atol=1e-4)


def test_conditional_guidance_scale_one(setup):
    """s = 1 : on échantillonne la composante demandée."""
    torch.manual_seed(0)
    d, oracle = setup
    y = torch.ones(2000, dtype=torch.long)
    x = d.ddim_sample_loop(oracle, (2000, 2), steps=50, y=y, guidance_scale=1.0)
    assert torch.allclose(x.mean(0), MEANS[1], atol=0.05)
    assert (x[:, 0] > 0).float().mean() > 0.99


def test_guidance_scale_zero_is_unconditional(setup):
    torch.manual_seed(0)
    d, oracle = setup
    y = torch.ones(4000, dtype=torch.long)
    x = d.ddim_sample_loop(oracle, (4000, 2), steps=50, y=y, guidance_scale=0.0)
    _check_mixture(x)


def test_trajectory_and_pred_x0_outputs(setup):
    """Options de retour utilisées pour les figures de trajectoire."""
    d, oracle = setup
    x, traj, x0s = d.ddim_sample_loop(oracle, (16, 2), steps=10, return_trajectory=True, return_pred_x0=True)
    assert len(traj) == 11 and len(x0s) == 10
    assert torch.equal(traj[-1], x)
    assert torch.allclose(x0s[-1], x)  # au dernier pas DDIM, x_0 = x̂_0
    per_example, t = d.training_losses(oracle, torch.randn(8, 2))
    assert per_example.shape == (8,) and t.shape == (8,)
