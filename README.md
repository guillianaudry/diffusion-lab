# diffusion-lab

Implémentations pédagogiques de modèles de diffusion en PyTorch, accompagnées de notes qui expliquent chaque méthode et ses variantes.

L'objectif est d'apprendre : le code privilégie la lisibilité et renvoie aux équations des notes, plutôt que la performance.

## Méthodes

| Méthode | Note | Implémentation | Statut |
|---|---|---|---|
| DDPM (ε, $x_0$, $v$ ; plannings linéaire et cosinus) | [01_ddpm](notes/01_ddpm.md) | `diffusion.py` → `training_loss`, `p_sample_loop` | ✅ |
| Score matching, SDE, ODE de flot | [02_score_sde](notes/02_score_sde.md) | — (théorie ; Langevin recuit à faire) | 📝 |
| DDIM (η quelconque, sous-suite de pas) | [03_ddim](notes/03_ddim_echantillonnage.md) | `diffusion.py` → `ddim_sample_loop` | ✅ |
| Classifier-free guidance | [04_guidance](notes/04_guidance.md) | `diffusion.py` → `p_uncond`, `guidance_scale` | ✅ |
| Débruiteur optimal exact (mélange de gaussiennes) | [01 §8](notes/01_ddpm.md) | `analytic.py` | ✅ |
| Flow matching / rectified flow | [05](notes/05_variantes.md) | — | ⏳ |
| EDM (préconditionnement, Heun) | [05](notes/05_variantes.md) | — | ⏳ |
| Improved DDPM (variance apprise) | [05](notes/05_variantes.md) | — | ⏳ |
| Diffusion latente | [05](notes/05_variantes.md) | — | ⏳ |
| DiT | [05](notes/05_variantes.md) | — | ⏳ |
| Distillation / consistency models | [05](notes/05_variantes.md) | — | ⏳ |

✅ implémenté et testé · 📝 note rédigée · ⏳ prévu

Commencer par [notes/00_vue_ensemble.md](notes/00_vue_ensemble.md).

## Structure

```
diffusion-lab/
├── notes/                    # explications (Markdown + LaTeX, rendu natif sur GitHub)
│   ├── 00_vue_ensemble.md
│   ├── 01_ddpm.md
│   ├── 02_score_sde.md
│   ├── 03_ddim_echantillonnage.md
│   ├── 04_guidance.md
│   ├── 05_variantes.md       # panorama + feuille de route
│   └── references.md
├── src/diffusion_lab/
│   ├── schedules.py          # plannings β_t
│   ├── diffusion.py          # GaussianDiffusion : perte, DDPM, DDIM, CFG
│   ├── analytic.py           # débruiteur exact pour mélange de gaussiennes
│   ├── data.py               # données 2D jouets, MNIST
│   ├── utils.py              # EMA, graines, grilles d'images
│   ├── viz.py                # figures : perte, trajectoires, animation GIF
│   └── models/
│       ├── embeddings.py     # embedding sinusoïdal du temps (+ classe)
│       ├── mlp.py            # réseau pour données 2D
│       └── unet.py           # U-Net pour images
├── scripts/
│   ├── train_toy.py          # entraînement 2D + toutes les figures
│   ├── train_mnist.py        # entraînement MNIST
│   ├── compare_parametrizations.py  # expérience ε / x0 / v
│   └── sample.py             # échantillonnage depuis un checkpoint
└── tests/                    # pytest
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision    # voir pytorch.org pour la version CUDA adaptée
pip install -e ".[dev]"
pytest                           # tourne sur CPU, sans données ni entraînement long
```

## Utilisation

```bash
# Données 2D (CPU suffisant, quelques minutes)
python scripts/train_toy.py --dataset moons
python scripts/train_toy.py --dataset swissroll --schedule cosine --prediction v
python scripts/train_toy.py --dataset 8gaussians --cond --guidance 0 1 3 6
python scripts/train_toy.py --plot-schedules

# MNIST (GPU recommandé)
python scripts/train_mnist.py --epochs 20
python scripts/train_mnist.py --epochs 20 --cond --schedule cosine --prediction v

# Échantillonner avec un autre échantillonneur
python scripts/sample.py runs/mnist_linear_eps/checkpoint.pt --sampler ddim --steps 20
python scripts/sample.py runs/mnist_cosine_v_cond/checkpoint.pt --class-label 3 --guidance 4
```

Les résultats sont écrits dans `runs/<nom>/` (ignoré par git).

## Expériences

**ε, $x_0$ ou $v$ : quelle cible prédire ?** (voir notes/01_ddpm.md, section 7)

```bash
python scripts/train_mnist.py --epochs 10 --prediction eps --name cmp_eps
python scripts/train_mnist.py --epochs 10 --prediction v   --name cmp_v
python scripts/train_mnist.py --epochs 10 --prediction x0  --name cmp_x0   # facultatif
python scripts/compare_parametrizations.py --runs runs/cmp_eps runs/cmp_v runs/cmp_x0
```

Produit dans `runs/compare/` l'erreur sur $\hat x_0$ et $\hat\epsilon$ selon $t$, et les images générées par DDIM avec 1 à 100 pas, avec et sans troncature de $\hat x_0$.

## Ajouter une méthode

1. Rédiger la note `notes/NN_nom.md` : intuition, équations clés, algorithme, pièges, exercices.
2. Implémenter dans `src/diffusion_lab/` en réutilisant l'interface des réseaux : `model(x_t, t, y) -> tenseur de même forme`.
3. Ajouter un test qui ne dépend pas d'un entraînement quand c'est possible (le débruiteur exact de `analytic.py` est fait pour ça).
4. Ajouter un script ou une option dans `scripts/`, et une figure de résultats.
5. Mettre à jour le tableau ci-dessus.

Une méthode = une branche = une pull request, pour garder un historique lisible de la progression.

## Journal de bord

| Date | Expérience | Observation |
|---|---|---|
| | | |
