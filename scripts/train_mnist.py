"""Entraîne un U-Net de diffusion sur MNIST (GPU recommandé).

Exemples :
    python scripts/train_mnist.py --epochs 20
    python scripts/train_mnist.py --epochs 20 --cond --p-uncond 0.1 --schedule cosine --prediction v

Le nombre de paramètres est affiché au démarrage. Un GPU est fortement recommandé :
sur CPU, une époque peut prendre très longtemps.

Sorties dans runs/<nom>/ :
    forward.png                 bruitage progressif de vraies images (processus direct)
    loss.png                    profil de la perte + perte par niveau de bruit t (mis à jour à chaque époque)
    samples_epXXX.png           grille d'images générées à la fin de chaque époque
    trajectory_epXXX.png        trajectoire de génération : x_t et la prédiction x̂_0 au fil des pas
    trajectory.gif              animation de la génération (dernière époque)
    history.json                pertes enregistrées (pour refaire les figures)
    checkpoint.pt, config.json
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
from diffusion_lab.viz import (
    plot_generation_trajectory,
    plot_image_table,
    plot_training_curves,
    save_trajectory_gif,
)


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
    p.add_argument("--t-bins", type=int, default=20, help="nombre de tranches de t pour la perte par niveau de bruit")
    p.add_argument("--traj-samples", type=int, default=4, help="images montrées dans la figure de trajectoire")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    return p.parse_args()


def _labels(args, n, device):
    """Classes à générer (modèle conditionnel) : 0, 1, 2, ... en boucle."""
    if not args.cond:
        return None
    return torch.arange(n, device=device) % 10


@torch.no_grad()
def sample_grid(diffusion, net, args, device, path):
    n = 80 if args.cond else 64  # conditionnel : une ligne de 8 images par chiffre
    y = torch.arange(10, device=device).repeat_interleave(8) if args.cond else None
    imgs = diffusion.ddim_sample_loop(
        net, (n, 1, 28, 28), steps=args.ddim_steps, y=y, guidance_scale=args.guidance, clip_x0=True
    )
    save_image_grid(imgs, path, nrow=8)


@torch.no_grad()
def sample_trajectory(diffusion, net, args, device, n, x_T=None):
    """Génère n images par DDIM en gardant les états x_t et les prédictions x̂_0."""
    x, traj, x0s = diffusion.ddim_sample_loop(
        net,
        (n, 1, 28, 28),
        steps=args.ddim_steps,
        y=_labels(args, n, device),
        guidance_scale=args.guidance,
        clip_x0=True,
        x_T=x_T,
        return_trajectory=True,
        return_pred_x0=True,
    )
    t_labels = [i + 1 for i in diffusion.ddim_timesteps(args.ddim_steps)]  # numérotation des notes
    return traj, x0s, t_labels


@torch.no_grad()
def plot_forward(diffusion, x0, T, path):
    """Processus direct appliqué à de vraies images, à plusieurs niveaux de bruit."""
    ts = [0, T // 20, T // 10, T // 5, 3 * T // 10, T // 2, 7 * T // 10, T - 1]
    noise = torch.randn_like(x0)
    cols = [x0] + [diffusion.q_sample(x0, torch.full((x0.shape[0],), t, device=x0.device), noise) for t in ts]
    rows = [[c[b] for c in cols] for b in range(x0.shape[0])]
    titles = ["$x_0$"] + [f"t = {t + 1}" for t in ts]
    plot_image_table(path, rows, titles, title=r"Processus direct : $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$")


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

    # Processus direct sur quelques vraies images
    x_real, _ = next(iter(loader))
    plot_forward(diffusion, x_real[:6].to(device), args.T, os.path.join(out_dir, "forward.png"))

    # Même bruit de départ à chaque époque : on voit le modèle progresser sur les mêmes images.
    g = torch.Generator().manual_seed(1234)
    x_T_fixed = torch.randn(args.traj_samples, 1, 28, 28, generator=g).to(device)

    history = {"step_loss": [], "epoch_loss": [], "loss_by_t": [], "t_bins": args.t_bins}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        bin_sum = torch.zeros(args.t_bins, device=device)
        bin_count = torch.zeros(args.t_bins, device=device)
        total, count = 0.0, 0
        for x0, y in loader:
            x0, y = x0.to(device), y.to(device)
            per_example, t = diffusion.training_losses(
                model, x0, y if args.cond else None, p_uncond=args.p_uncond if args.cond else 0.0
            )
            loss = per_example.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            # Suivi de la perte, globale et par tranche de t
            bins = (t * args.t_bins) // args.T
            bin_sum += torch.bincount(bins, weights=per_example.detach().float(), minlength=args.t_bins).float()
            bin_count += torch.bincount(bins, minlength=args.t_bins).float()
            history["step_loss"].append(loss.item())
            total += loss.item() * x0.shape[0]
            count += x0.shape[0]

        epoch_loss = total / count
        history["epoch_loss"].append(epoch_loss)
        history["loss_by_t"].append((bin_sum / bin_count.clamp(min=1)).tolist())
        print(f"époque {epoch:3d} | perte {epoch_loss:.4f} | {time.time() - t0:.0f}s")

        # Figures de l'époque
        net = ema.model
        plot_training_curves(
            os.path.join(out_dir, "loss.png"),
            history["step_loss"],
            history["epoch_loss"],
            history["loss_by_t"],
            args.T,
            steps_per_epoch=len(loader),
        )
        sample_grid(diffusion, net, args, device, os.path.join(out_dir, f"samples_ep{epoch:03d}.png"))
        traj, x0s, t_labels = sample_trajectory(diffusion, net, args, device, args.traj_samples, x_T_fixed)
        plot_generation_trajectory(
            os.path.join(out_dir, f"trajectory_ep{epoch:03d}.png"),
            traj,
            x0s,
            t_labels,
            title=f"Génération DDIM ({args.ddim_steps} pas), époque {epoch}",
        )

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        torch.save(
            {"kind": "mnist", "config": vars(args), "model": model.state_dict(), "ema": ema.state_dict()},
            os.path.join(out_dir, "checkpoint.pt"),
        )

    # Animation finale : 16 images, x_t à gauche et x̂_0 à droite
    traj, x0s, _ = sample_trajectory(diffusion, ema.model, args, device, 16)
    save_trajectory_gif(os.path.join(out_dir, "trajectory.gif"), traj, x0s, nrow=4)
    print(f"Terminé. Résultats dans {out_dir}/")


if __name__ == "__main__":
    main()
