import pytest
import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_schedule_properties(name):
    betas = get_beta_schedule(name, 1000)
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas < 1).all()
    ab = torch.cumprod(1 - betas, 0)
    assert (ab[1:] < ab[:-1]).all()          # ᾱ_t strictement décroissant
    assert ab[-1] < 1e-3                      # x_T ≈ bruit pur


def test_q_sample_statistics():
    torch.manual_seed(0)
    d = GaussianDiffusion(get_beta_schedule("linear", 1000))
    x0 = torch.full((200_000, 1), 2.0)
    i = 300
    xt = d.q_sample(x0, torch.full((x0.shape[0],), i))
    ab = d.alphas_cumprod[i]
    assert torch.allclose(xt.mean(), 2.0 * ab.sqrt(), atol=0.01)
    assert torch.allclose(xt.var(), 1 - ab, atol=0.01)


@pytest.mark.parametrize("prediction", ["eps", "x0", "v"])
def test_parametrizations_roundtrip(prediction):
    """Si le réseau sortait la cible exacte, on doit retrouver x0 et ε."""
    torch.manual_seed(0)
    d = GaussianDiffusion(get_beta_schedule("cosine", 1000), prediction)
    x0, noise = torch.randn(64, 3), torch.randn(64, 3)
    t = torch.randint(0, 1000, (64,))
    xt = d.q_sample(x0, t, noise)
    target = d.training_target(x0, noise, t)
    x0_hat, eps_hat = d.predict_x0_and_eps(target, xt, t)
    assert torch.allclose(x0_hat, x0, atol=1e-3)
    assert torch.allclose(eps_hat, noise, atol=1e-3)


def test_posterior_mean_matches_eps_formula():
    """μ̃(x_t, x̂_0) = (x_t − β_t/√(1−ᾱ_t) ε̂)/√α_t  (notes/01_ddpm.md, section 4)."""
    d = GaussianDiffusion(get_beta_schedule("linear", 1000))
    xt, eps = torch.randn(16, 2), torch.randn(16, 2)
    for i in [1, 10, 500, 999]:
        t = torch.full((16,), i)
        x0_hat, _ = d.predict_x0_and_eps(eps, xt, t)
        mean = d.ddpm_step(xt, x0_hat, i, noise=torch.zeros_like(xt))
        beta, ab = d.betas[i], d.alphas_cumprod[i]
        expected = (xt - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()
        assert torch.allclose(mean, expected, atol=1e-4)


def test_ddim_timesteps_start_from_noise():
    """Le premier pas DDIM part toujours de x_T, même avec un seul pas."""
    d = GaussianDiffusion(get_beta_schedule("linear", 1000))
    assert d.ddim_timesteps(1) == [999]
    assert d.ddim_timesteps(2) == [999, 0]
    ts = d.ddim_timesteps(50)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 50
