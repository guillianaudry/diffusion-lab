"""Entraîne un U-Net de diffusion sur MNIST (GPU recommandé).

Exemples :
    python scripts/train_mnist.py --epochs 20
    python scripts/train_mnist.py --epochs 20 --cond --p-uncond 0.1 --schedule cosine --prediction v

Le nombre de paramètres est affiché au démarrage. Un GPU est fortement recommandé :
sur CPU, une époque peut prendre très longtemps.
"""

import argparse
import json
import os
import time

import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule
from diffusion_lab.data import mnist_loader
from diffusion_lab.models import UNet
from diffusion_lab.utils import EMA, count_parameters, get_device, save_image_grid, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--prediction", default="eps", choices=["eps", "x0", "v"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--cond", action="store_true")
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    return p.parse_args()


@torch.no_grad()
def sample_grid(diffusion, net, args, device, path):
    n = 64
    if args.cond:
        n = 80  # une ligne de 8 images par chiffre
        y = torch.arange(10, device=device).repeat_interleave(8)
        imgs = diffusion.ddim_sample_loop(
            net, (n, 1, 28, 28), steps=args.ddim_steps, y=y, guidance_scale=args.guidance, clip_x0=True
        )
    else:
        imgs = diffusion.ddim_sample_loop(net, (n, 1, 28, 28), steps=args.ddim_steps, clip_x0=True)
    save_image_grid(imgs, path, nrow=8)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    name = args.name or f"mnist_{args.schedule}_{args.prediction}" + ("_cond" if args.cond else "")
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    loader = mnist_loader(args.batch_size, args.data_root)
    diffusion = GaussianDiffusion(get_beta_schedule(args.schedule, args.T), args.prediction).to(device)
    model = UNet(1, args.base_channels, (1, 2, 2), 2, 0.1, 10 if args.cond else None).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"{name} | {count_parameters(model):,} paramètres | device={device}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x0, y in loader:
            x0, y = x0.to(device), y.to(device)
            loss = diffusion.training_loss(
                model, x0, y if args.cond else None, p_uncond=args.p_uncond if args.cond else 0.0
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            total += loss.item() * x0.shape[0]
            count += x0.shape[0]
        print(f"époque {epoch:3d} | perte {total / count:.4f} | {time.time() - t0:.0f}s")
        sample_grid(diffusion, ema.model, args, device, os.path.join(out_dir, f"samples_ep{epoch:03d}.png"))
        torch.save(
            {"kind": "mnist", "config": vars(args), "model": model.state_dict(), "ema": ema.state_dict()},
            os.path.join(out_dir, "checkpoint.pt"),
        )
    print(f"Terminé. Résultats dans {out_dir}/")


if __name__ == "__main__":
    main()
