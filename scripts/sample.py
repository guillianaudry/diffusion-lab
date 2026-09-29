"""Échantillonne à partir d'un checkpoint (toy ou MNIST) avec l'échantillonneur de son choix.

Exemples :
    python scripts/sample.py runs/mnist_linear_eps/checkpoint.pt --sampler ddim --steps 50
    python scripts/sample.py runs/mnist_linear_eps_cond/checkpoint.pt --class-label 7 --guidance 4
    python scripts/sample.py runs/moons_linear_eps/checkpoint.pt --sampler ddpm
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule
from diffusion_lab.data import TOY_NUM_CLASSES
from diffusion_lab.models import MLPDenoiser, UNet
from diffusion_lab.utils import get_device, save_image_grid, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--steps", type=int, default=50, help="pas DDIM")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--class-label", type=int, default=None)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    if ckpt["kind"] == "toy":
        num_classes = TOY_NUM_CLASSES[cfg["dataset"]] if cfg["cond"] else None
        model = MLPDenoiser(2, cfg["hidden"], cfg["blocks"], num_classes)
        shape = (args.n, 2)
    else:
        num_classes = 10 if cfg["cond"] else None
        model = UNet(1, cfg["base_channels"], (1, 2, 2), 2, 0.1, num_classes)
        shape = (args.n, 1, 28, 28)
    model.load_state_dict(ckpt["model"] if args.no_ema else ckpt["ema"])
    model.to(device).eval()

    diffusion = GaussianDiffusion(get_beta_schedule(cfg["schedule"], cfg["T"]), cfg["prediction"]).to(device)
    y = None
    if args.class_label is not None:
        if num_classes is None:
            raise SystemExit("Ce modèle n'est pas conditionnel.")
        y = torch.full((args.n,), args.class_label, device=device, dtype=torch.long)

    is_image = ckpt["kind"] != "toy"
    kwargs = dict(y=y, guidance_scale=args.guidance, clip_x0=is_image)
    if args.sampler == "ddpm":
        x = diffusion.p_sample_loop(model, shape, **kwargs)
    else:
        x = diffusion.ddim_sample_loop(model, shape, steps=args.steps, eta=args.eta, **kwargs)

    out = args.output or os.path.join(os.path.dirname(args.checkpoint), f"sample_{args.sampler}.png")
    if is_image:
        save_image_grid(x, out)
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        x = x.cpu()
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5)
        ax.set_aspect("equal")
        fig.savefig(out, dpi=120)
    print(f"Enregistré : {out}")


if __name__ == "__main__":
    main()
