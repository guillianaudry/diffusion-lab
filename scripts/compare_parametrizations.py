"""Compare des modèles MNIST qui ne diffèrent que par la cible prédite (ε, x_0 ou v).

1. Entraîner les modèles avec les mêmes réglages (seule --prediction change) :
       python scripts/train_mnist.py --epochs 10 --prediction eps --name cmp_eps
       python scripts/train_mnist.py --epochs 10 --prediction v   --name cmp_v
       python scripts/train_mnist.py --epochs 10 --prediction x0  --name cmp_x0   # facultatif
2. Comparer :
       python scripts/compare_parametrizations.py --runs runs/cmp_eps runs/cmp_v runs/cmp_x0

Les pertes d'entraînement ne sont PAS comparables entre paramétrisations (les cibles
diffèrent). On compare donc des grandeurs communes, sur les images de test :

    x0_error_by_t.png      E‖x̂_0 − x_0‖² selon t  (stabilité de la reconstruction de l'image)
    eps_error_by_t.png     E‖ε̂ − ε‖² selon t
    steps_clip.png         images générées par DDIM avec 1, 2, 5, … pas (x̂_0 tronqué dans [−1, 1])
    steps_noclip.png       idem SANS troncature : révèle les instabilités que la troncature masque
    results.json           valeurs numériques

Les mêmes images de test, le même bruit et le même bruit de départ x_T sont utilisés pour
tous les modèles, pour que les différences ne viennent que de la paramétrisation.
"""

import argparse
import json
import os

import torch

from diffusion_lab import GaussianDiffusion, get_beta_schedule
from diffusion_lab.data import mnist_loader
from diffusion_lab.models import UNet
from diffusion_lab.utils import get_device, set_seed
from diffusion_lab.viz import plot_curves_by_t, plot_steps_comparison


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="dossiers contenant checkpoint.pt")
    p.add_argument("--labels", nargs="+", default=None, help="noms affichés (par défaut : prédiction du modèle)")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--n", type=int, default=16, help="images générées par case")
    p.add_argument("--n-test", type=int, default=512, help="images de test pour les erreurs selon t")
    p.add_argument("--t-points", type=int, default=40, help="nombre de valeurs de t évaluées")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default=os.path.join("runs", "compare"))
    return p.parse_args()


def load_run(run_dir, device, use_ema=True):
    ckpt = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device)
    cfg = ckpt["config"]
    model = UNet(1, cfg["base_channels"], (1, 2, 2), 2, 0.1, 10 if cfg["cond"] else None)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.to(device).eval()
    diffusion = GaussianDiffusion(get_beta_schedule(cfg["schedule"], cfg["T"]), cfg["prediction"]).to(device)
    return model, diffusion, cfg


@torch.no_grad()
def errors_by_t(diffusion, net, x0, noise, y, t_indices, chunk=128):
    """Erreurs quadratiques moyennes de x̂_0 et ε̂ (sans troncature ni guidance) pour chaque t."""
    err_x0, err_eps = [], []
    for i in t_indices:
        s_x0, s_eps = 0.0, 0.0
        for a in range(0, x0.shape[0], chunk):
            xb, nb = x0[a : a + chunk], noise[a : a + chunk]
            yb = y[a : a + chunk] if y is not None else None
            t = torch.full((xb.shape[0],), i, device=xb.device, dtype=torch.long)
            x_t = diffusion.q_sample(xb, t, nb)
            x0_hat, eps_hat = diffusion.model_predictions(net, x_t, t, yb)
            s_x0 += ((x0_hat - xb) ** 2).mean(dim=(1, 2, 3)).sum().item()
            s_eps += ((eps_hat - nb) ** 2).mean(dim=(1, 2, 3)).sum().item()
        err_x0.append(s_x0 / x0.shape[0])
        err_eps.append(s_eps / x0.shape[0])
    return err_x0, err_eps


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    runs = [load_run(r, device, not args.no_ema) for r in args.runs]
    labels = args.labels or [cfg["prediction"] for _, _, cfg in runs]
    T = runs[0][2]["T"]
    for _, _, cfg in runs:
        if cfg["T"] != T or cfg["schedule"] != runs[0][2]["schedule"]:
            print("Attention : les modèles n'ont pas le même T ou planning, la comparaison est moins propre.")

    # Données et bruits communs à tous les modèles
    loader = mnist_loader(args.n_test, args.data_root, train=False, num_workers=0)
    x0, y_test = next(iter(loader))
    x0, y_test = x0.to(device), y_test.to(device)
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(x0.shape, generator=g).to(device)
    x_T = torch.randn(args.n, 1, 28, 28, generator=g).to(device)
    t_indices = torch.linspace(0, T - 1, args.t_points).round().long().unique().tolist()

    results = {"t": [i + 1 for i in t_indices], "steps": args.steps, "models": {}}
    gen_clip, gen_noclip = {}, {}
    for (net, diffusion, cfg), label in zip(runs, labels):
        print(f"→ {label} ({cfg['prediction']}, planning {cfg['schedule']})")
        y = y_test if cfg["cond"] else None
        err_x0, err_eps = errors_by_t(diffusion, net, x0, noise, y, t_indices)
        results["models"][label] = {"prediction": cfg["prediction"], "x0_error": err_x0, "eps_error": err_eps}

        y_gen = (torch.arange(args.n, device=device) % 10) if cfg["cond"] else None
        scale = cfg["guidance"] if cfg["cond"] else 1.0
        gen_clip[label], gen_noclip[label] = [], []
        for s in args.steps:
            for clip, store in ((True, gen_clip), (False, gen_noclip)):
                imgs = diffusion.ddim_sample_loop(
                    net, (args.n, 1, 28, 28), steps=s, y=y_gen, guidance_scale=scale, clip_x0=clip, x_T=x_T
                )
                store[label].append(imgs.cpu())
        # Valeurs hors de [-1, 1] sans troncature : indicateur d'instabilité
        results["models"][label]["out_of_range_noclip"] = [
            ((im.abs() > 1).float().mean().item()) for im in gen_noclip[label]
        ]

    # Figures
    plot_curves_by_t(
        os.path.join(args.out, "x0_error_by_t.png"),
        results["t"],
        {k: v["x0_error"] for k, v in results["models"].items()},
        r"$E\,\|\hat{x}_0 - x_0\|^2$",
        "Erreur sur l'image reconstruite (images de test, sans troncature)",
    )
    plot_curves_by_t(
        os.path.join(args.out, "eps_error_by_t.png"),
        results["t"],
        {k: v["eps_error"] for k, v in results["models"].items()},
        r"$E\,\|\hat{\epsilon} - \epsilon\|^2$",
        "Erreur sur le bruit reconstruit (images de test)",
    )
    plot_steps_comparison(
        os.path.join(args.out, "steps_clip.png"), gen_clip, args.steps,
        title=r"DDIM ($\eta=0$), $\hat{x}_0$ tronqué dans [-1, 1], même $x_T$ pour tous",
    )
    plot_steps_comparison(
        os.path.join(args.out, "steps_noclip.png"), gen_noclip, args.steps,
        title=r"DDIM ($\eta=0$) SANS troncature de $\hat{x}_0$, même $x_T$ pour tous",
    )
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Résumé texte
    print("\nErreur E‖x̂_0 − x_0‖² à quelques niveaux de bruit :")
    picks = [0, len(t_indices) // 4, len(t_indices) // 2, 3 * len(t_indices) // 4, len(t_indices) - 1]
    print("modèle".ljust(10) + "".join(f"t={results['t'][p]}".rjust(12) for p in picks))
    for label, v in results["models"].items():
        print(label.ljust(10) + "".join(f"{v['x0_error'][p]:12.4g}" for p in picks))
    print("\nPart des pixels hors de [-1, 1] sans troncature, selon le nombre de pas :")
    print("modèle".ljust(10) + "".join(f"{s} pas".rjust(9) for s in args.steps))
    for label, v in results["models"].items():
        print(label.ljust(10) + "".join(f"{x:9.1%}" for x in v["out_of_range_noclip"]))
    print(f"\nFigures dans {args.out}/")


if __name__ == "__main__":
    main()
