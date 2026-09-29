import pytest
import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule
from diffusion_lab.models import MLPDenoiser, UNet


@pytest.mark.parametrize("num_classes", [None, 10])
def test_unet_shapes(num_classes):
    model = UNet(1, 32, (1, 2, 2), 1, 0.0, num_classes)
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,)) if num_classes else None
    assert model(x, t, y).shape == x.shape


def test_unet_cifar_shape():
    model = UNet(3, 32, (1, 2, 2, 2), 1, 0.0)
    x = torch.randn(2, 3, 32, 32)
    assert model(x, torch.tensor([0, 999])).shape == x.shape


def test_mlp_training_step_reduces_loss():
    torch.manual_seed(0)
    d = GaussianDiffusion(get_beta_schedule("linear", 100))
    model = MLPDenoiser(2, 64, 2, num_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.randn(256, 2) * 0.1 + 1.0
    y = torch.randint(0, 2, (256,))
    losses = []
    for _ in range(200):
        loss = d.training_loss(model, x0, y, p_uncond=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20
