"""Jeux de données : distributions 2D jouets et MNIST.

Les jeux 2D sont générés à la volée, sans téléchargement. Ils sont normalisés
(variance ≈ 1 par coordonnée), comme le suppose le processus direct.
Certains ont des étiquettes, pour expérimenter la guidance :
    moons (2 classes), 8gaussians (8 classes).
"""

from __future__ import annotations

import math

import torch

TOY_DATASETS = ("moons", "8gaussians", "swissroll", "checkerboard")


def _moons(n: int, g: torch.Generator):
    y = torch.randint(0, 2, (n,), generator=g)
    theta = torch.rand(n, generator=g) * math.pi
    outer = torch.stack([theta.cos(), theta.sin()], 1)
    inner = torch.stack([1 - theta.cos(), 0.5 - theta.sin()], 1)
    x = torch.where(y.unsqueeze(1) == 0, outer, inner)
    x = x + 0.06 * torch.randn(n, 2, generator=g)
    x = (x - torch.tensor([0.5, 0.25])) / torch.tensor([0.87, 0.50])
    return x, y


def _eight_gaussians(n: int, g: torch.Generator):
    y = torch.randint(0, 8, (n,), generator=g)
    angles = y.float() * (2 * math.pi / 8)
    centers = 2.0 * torch.stack([angles.cos(), angles.sin()], 1)
    x = centers + 0.2 * torch.randn(n, 2, generator=g)
    return x / 1.42, y


def _swissroll(n: int, g: torch.Generator):
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n, generator=g))
    x = torch.stack([t * t.cos(), t * t.sin()], 1) / 8.0
    x = x + 0.05 * torch.randn(n, 2, generator=g)
    return x, None


def _checkerboard(n: int, g: torch.Generator):
    x1 = torch.rand(n, generator=g) * 4 - 2
    x2 = torch.rand(n, generator=g) - torch.randint(0, 2, (n,), generator=g).float() * 2
    x2 = x2 + (torch.floor(x1) % 2)
    return torch.stack([x1, x2], 1) / 1.15, None


_GENERATORS = {
    "moons": _moons,
    "8gaussians": _eight_gaussians,
    "swissroll": _swissroll,
    "checkerboard": _checkerboard,
}

TOY_NUM_CLASSES = {"moons": 2, "8gaussians": 8, "swissroll": None, "checkerboard": None}


def sample_toy(name: str, n: int, seed: int | None = None):
    """Renvoie (x, y) ; y vaut None pour les jeux sans étiquettes."""
    if name not in _GENERATORS:
        raise ValueError(f"Jeu inconnu '{name}'. Choix : {TOY_DATASETS}")
    g = torch.Generator()
    if seed is None:  # tiré du générateur global, donc reproductible via set_seed
        seed = int(torch.randint(0, 2**62, (1,)).item())
    g.manual_seed(seed)
    return _GENERATORS[name](n, g)


def mnist_loader(batch_size: int = 128, root: str = "data", train: bool = True, num_workers: int = 2):
    """MNIST normalisé dans [-1, 1] (nécessite torchvision)."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(root, train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, drop_last=train)
