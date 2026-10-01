"""Figures d'analyse : courbes de perte, trajectoires de génération, animations.

Les fonctions acceptent des tenseurs PyTorch ou des tableaux NumPy ; les images
sont supposées dans [-1, 1], au format (B, C, H, W).
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _to_display(img: np.ndarray) -> np.ndarray:
    """(C, H, W) dans [-1, 1] -> (H, W) ou (H, W, 3) dans [0, 1]."""
    img = np.clip((img + 1) / 2, 0, 1)
    return img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)


def image_grid(images, nrow: int = 8, pad: int = 2) -> np.ndarray:
    """Assemble (B, C, H, W) en une grille (H', W') ou (H', W', 3), valeurs dans [0, 1]."""
    images = to_numpy(images)
    B, C, H, W = images.shape
    nrows = (B + nrow - 1) // nrow
    grid = np.ones((C, nrows * (H + pad) + pad, nrow * (W + pad) + pad), dtype=np.float32)
    for k in range(B):
        r, c = divmod(k, nrow)
        top, left = pad + r * (H + pad), pad + c * (W + pad)
        grid[:, top : top + H, left : left + W] = np.clip((images[k] + 1) / 2, 0, 1)
    return grid[0] if C == 1 else grid.transpose(1, 2, 0)


# ----------------------------------------------------------------------
# Perte
# ----------------------------------------------------------------------
def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(
    path: str,
    step_losses,
    epoch_losses,
    loss_by_t,
    T: int,
    steps_per_epoch: int | None = None,
) -> None:
    """Deux panneaux :
    - gauche : perte à chaque itération (brute + moyenne glissante) et moyenne par époque ;
    - droite : perte moyenne par tranche de t, une courbe par époque (du clair au foncé).

    loss_by_t : liste (une entrée par époque) de tableaux (n_bins,) de pertes moyennes.
    """
    step_losses = np.asarray(step_losses, dtype=np.float64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    it = np.arange(1, len(step_losses) + 1)
    ax1.plot(it, step_losses, color="0.75", lw=0.5, label="par itération")
    window = max(1, min(200, len(step_losses) // 20))
    sm = _smooth(step_losses, window)
    ax1.plot(it[window - 1 :], sm, color="C0", lw=1.5, label=f"moyenne glissante ({window})")
    if steps_per_epoch and len(epoch_losses):
        ex = np.arange(1, len(epoch_losses) + 1) * steps_per_epoch
        ax1.plot(ex, epoch_losses, "o-", color="C3", ms=4, lw=1, label="moyenne par époque")
    ax1.set_yscale("log")
    ax1.set_xlabel("itération")
    ax1.set_ylabel("perte (échelle log)")
    ax1.set_title("Profil de la perte")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    if len(loss_by_t):
        n_bins = len(loss_by_t[0])
        centers = (np.arange(n_bins) + 0.5) * T / n_bins
        cmap = plt.get_cmap("viridis")
        n = len(loss_by_t)
        for e, curve in enumerate(loss_by_t):
            color = cmap(0.15 + 0.8 * e / max(1, n - 1))
            label = f"époque {e + 1}" if e in (0, n - 1) else None
            ax2.plot(centers, curve, color=color, lw=1.2, label=label)
        ax2.set_yscale("log")
        ax2.set_xlabel("pas de bruit t")
        ax2.set_ylabel("perte moyenne (échelle log)")
        ax2.set_title("Perte selon le niveau de bruit")
        ax2.grid(alpha=0.3, which="both")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves_by_t(path: str, t_values, curves: dict, ylabel: str, title: str, logy: bool = True) -> None:
    """Une courbe par modèle en fonction de t (curves : {nom: valeurs})."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (label, values) in enumerate(curves.items()):
        ax.plot(t_values, values, "o-", ms=3, lw=1.5, color=f"C{k}", label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("pas de bruit t")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_steps_comparison(path: str, results: dict, steps, nrow: int = 4, title: str | None = None) -> None:
    """Tableau modèles × nombre de pas ; chaque case est une grille d'images générées.

    results : {nom du modèle: [lot d'images pour steps[0], lot pour steps[1], ...]}.
    """
    names = list(results)
    fig, axes = plt.subplots(
        len(names), len(steps), figsize=(1.9 * len(steps) + 0.6, 1.9 * len(names) + 0.6), squeeze=False
    )
    for r, name in enumerate(names):
        for c, s in enumerate(steps):
            ax = axes[r, c]
            grid = image_grid(results[name][c], nrow)
            ax.imshow(grid, cmap="gray" if grid.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{s} pas", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Trajectoires
# ----------------------------------------------------------------------
def plot_image_table(path: str, rows, col_titles, row_titles=None, title: str | None = None) -> None:
    """Tableau d'images : rows[r][c] est une image (C, H, W) dans [-1, 1]."""
    rows = [[to_numpy(img) for img in row] for row in rows]
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(1.2 * nc + 0.8, 1.2 * nr + 0.6), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r, c]
            img = _to_display(rows[r][c])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if row_titles is not None and c == 0:
                ax.set_ylabel(row_titles[r], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_generation_trajectory(path: str, trajectory, pred_x0, t_labels, n_cols: int = 10, title=None) -> None:
    """Trajectoire de génération de quelques images.

    Pour chaque image, deux lignes :
      x_t   : l'état courant de l'échantillonneur ;
      x̂_0  : la prédiction du réseau à partir de cet état (E[x_0 | x_t]).
    trajectory : liste de S + 1 lots (B, C, H, W) ; pred_x0 : liste de S lots ;
    t_labels   : S valeurs de t (numérotation des notes, 1..T) associées à pred_x0.
    """
    S = len(pred_x0)
    cols = np.unique(np.linspace(0, S - 1, n_cols - 1).round().astype(int)).tolist()
    traj = [to_numpy(trajectory[k]) for k in cols] + [to_numpy(trajectory[S])]
    x0s = [to_numpy(pred_x0[k]) for k in cols] + [to_numpy(trajectory[S])]
    col_titles = [f"t = {t_labels[k]}" for k in cols] + ["résultat"]
    B = traj[0].shape[0]
    rows, row_titles = [], []
    for b in range(B):
        rows.append([x[b] for x in traj])
        row_titles.append(f"#{b + 1}  $x_t$")
        rows.append([x[b] for x in x0s])
        row_titles.append(f"#{b + 1}  $\\hat{{x}}_0$")
    plot_image_table(path, rows, col_titles, row_titles, title)


def save_trajectory_gif(
    path: str,
    trajectory,
    pred_x0=None,
    nrow: int = 4,
    scale: int = 2,
    max_frames: int = 40,
    duration_ms: int = 100,
    hold_ms: int = 2000,
) -> None:
    """Animation GIF de la génération : grille des x_t (et, à droite, des x̂_0 si fournis).

    Au plus max_frames images (le bruit se compresse mal : chaque image pèse lourd).
    """
    from PIL import Image

    frames = []
    S = len(trajectory) - 1
    keep = np.unique(np.linspace(0, S, min(max_frames, S + 1)).round().astype(int)).tolist()
    for k in keep:
        left = image_grid(trajectory[k], nrow)
        if pred_x0 is not None:
            right = image_grid(pred_x0[k] if k < S else trajectory[S], nrow)
            sep = np.ones((left.shape[0], 6) + left.shape[2:], dtype=np.float32)
            left = np.concatenate([left, sep, right], axis=1)
        arr = (left * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)  # niveaux de gris (2D) ou RGB (3D)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        frames.append(img.convert("P") if arr.ndim == 3 else img)
    durations = [duration_ms] * (len(frames) - 1) + [hold_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
