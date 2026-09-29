"""Entraîne un modèle de diffusion sur une distribution 2D et produit les figures.

Exemples :
    python scripts/train_toy.py --dataset moons
    python scripts/train_toy.py --dataset swissroll --schedule cosine --prediction v
    python scripts/train_toy.py --dataset 8gaussians --cond --p-uncond 0.2 --guidance 0 1 3 6
    python scripts/train_toy.py --plot-schedules            # trace seulement les plannings

Sorties dans runs/<nom>/ : checkpoint, courbe de perte, processus direct,
échantillons DDPM vs DDIM, effet du nombre de pas DDIM, trajectoires, guidance.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule
from diffusion_lab.data import TOY_DATASETS, TOY_NUM_CLASSES, sample_toy
from diffusion_lab.models import MLPDenoiser
from diffusion_lab.utils import EMA, count_parameters, get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="moons", choices=TOY_DATASETS)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--prediction", default="eps", choices=["eps", "x0", "v"])
    p.add_argument("--steps", type=int, default=10000, help="itérations d'entraînement")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--cond", action="store_true", help="modèle conditionné par la classe")
    p.add_argument("--p-uncond", type=float, default=0.2, help="proba. de jeton ∅ (CFG)")
    p.add_argument("--guidance", type=float, nargs="+", default=[0.0, 1.0, 3.0, 6.0])
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--plot-schedules", action="store_true")
    return p.parse_args()


def scatter(ax, x, title, color="C0", s=2):
    x = x.detach().cpu()
    ax.scatter(x[:, 0], x[:, 1], s=s, alpha=0.5, color=color)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_schedules(T, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name in ["linear", "cosine"]:
        d = GaussianDiffusion(get_beta_schedule(name, T))
        ab = d.alphas_cumprod
        axes[0].plot(ab.numpy(), label=name)
        axes[1].plot(torch.log(ab / (1 - ab)).numpy(), label=name)
    axes[0].set_title(r"$\bar\alpha_t$")
    axes[1].set_title(r"$\log \mathrm{SNR}(t) = \log \bar\alpha_t / (1-\bar\alpha_t)$")
    for ax in axes:
        ax.set_xlabel("t")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.plot_schedules:
        path = os.path.join(args.out, "schedules.png")
        plot_schedules(args.T, path)
        print(f"Figure enregistrée : {path}")
        return

    set_seed(args.seed)
    device = get_device(args.device)
    num_classes = TOY_NUM_CLASSES[args.dataset] if args.cond else None
    if args.cond and num_classes is None:
        raise SystemExit(f"Le jeu '{args.dataset}' n'a pas d'étiquettes ; choisir moons ou 8gaussians.")

    name = args.name or f"{args.dataset}_{args.schedule}_{args.prediction}" + ("_cond" if args.cond else "")
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    diffusion = GaussianDiffusion(get_beta_schedule(args.schedule, args.T), args.prediction).to(device)
    model = MLPDenoiser(2, args.hidden, args.blocks, num_classes).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    print(f"{name} | {count_parameters(model):,} paramètres | device={device}")

    # --- Entraînement -------------------------------------------------------
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x0, y = sample_toy(args.dataset, args.batch_size)
        x0 = x0.to(device)
        y = y.to(device) if (args.cond and y is not None) else None
        loss = diffusion.training_loss(model, x0, y, p_uncond=args.p_uncond if args.cond else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        losses.append(loss.item())
        if step % 1000 == 0 or step == 1:
            recent = sum(losses[-1000:]) / len(losses[-1000:])
            print(f"  pas {step:6d} | perte {recent:.4f} | {time.time() - t0:.0f}s")

    torch.save(
        {"kind": "toy", "config": vars(args), "model": model.state_dict(), "ema": ema.state_dict()},
        os.path.join(out_dir, "checkpoint.pt"),
    )
    net = ema.model
    n = args.n_samples
    data, data_y = sample_toy(args.dataset, n, seed=12345)

    # --- Courbe de perte ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3))
    w = max(1, min(100, len(losses)))
    smooth = torch.tensor(losses).unfold(0, w, w).mean(1)
    ax.plot(torch.arange(len(smooth)) * w, smooth)
    ax.set_xlabel("itération")
    ax.set_ylabel("perte (moy. glissante)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)

    # --- Processus direct ---------------------------------------------------
    ts = [0, args.T // 10, args.T // 4, args.T // 2, args.T - 1]
    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3))
    x0 = data.to(device)
    for ax, t in zip(axes, ts):
        xt = diffusion.q_sample(x0, torch.full((n,), t, device=device))
        scatter(ax, xt, f"q(x_t | x_0), t={t + 1}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "forward.png"), dpi=120)
    plt.close(fig)

    # --- DDPM vs DDIM (inconditionnel) -------------------------------------
    y_none = None
    x_ddpm, traj = diffusion.p_sample_loop(net, (n, 2), y_none, return_trajectory=True)
    x_ddim = diffusion.ddim_sample_loop(net, (n, 2), steps=args.ddim_steps, eta=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    scatter(axes[0], data, "données")
    scatter(axes[1], x_ddpm, f"DDPM ({args.T} pas)", "C1")
    scatter(axes[2], x_ddim, f"DDIM η=0 ({args.ddim_steps} pas)", "C2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "samples.png"), dpi=120)
    plt.close(fig)

    # --- Trajectoire inverse DDPM -------------------------------------------
    idx = [0, args.T // 2, 3 * args.T // 4, 9 * args.T // 10, args.T]
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3))
    for ax, k in zip(axes, idx):
        scatter(ax, traj[k], f"x_t, t={args.T - k}", "C1")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reverse_trajectory.png"), dpi=120)
    plt.close(fig)

    # --- Nombre de pas DDIM --------------------------------------------------
    steps_list = [2, 5, 10, 20, 50, 100]
    x_T = torch.randn(n, 2, device=device)
    fig, axes = plt.subplots(2, len(steps_list), figsize=(2.6 * len(steps_list), 5.4))
    for j, s in enumerate(steps_list):
        for r, eta in enumerate([0.0, 1.0]):
            xs = diffusion.ddim_sample_loop(net, (n, 2), steps=s, eta=eta, x_T=x_T)
            scatter(axes[r, j], xs, f"{s} pas, η={eta:g}", "C2" if eta == 0 else "C3")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ddim_steps.png"), dpi=120)
    plt.close(fig)

    # --- Guidance -------------------------------------------------------------
    if args.cond:
        target = 0
        y = torch.full((n,), target, device=device, dtype=torch.long)
        fig, axes = plt.subplots(1, len(args.guidance) + 1, figsize=(3 * (len(args.guidance) + 1), 3))
        scatter(axes[0], data[data_y == target], f"données, classe {target}")
        for ax, s in zip(axes[1:], args.guidance):
            xs = diffusion.ddim_sample_loop(net, (n, 2), steps=args.ddim_steps, y=y, guidance_scale=s)
            scatter(ax, xs, f"CFG s={s:g}", "C4")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "guidance.png"), dpi=120)
        plt.close(fig)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Terminé en {time.time() - t0:.0f}s. Résultats dans {out_dir}/")


if __name__ == "__main__":
    main()
