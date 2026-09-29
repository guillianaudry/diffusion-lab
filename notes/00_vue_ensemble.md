# 00 — Vue d'ensemble des modèles de diffusion

## Le problème

Un modèle génératif cherche à apprendre une distribution $p_\text{data}(x)$ à partir d'exemples, pour pouvoir **échantillonner** de nouvelles données (et parfois évaluer une vraisemblance).

Les modèles de diffusion y parviennent en deux temps :

1. **Processus direct (forward)** : on détruit progressivement la donnée en ajoutant du bruit gaussien, jusqu'à obtenir un bruit pur $\mathcal{N}(0, I)$. Ce processus est **fixé** : aucun paramètre n'est appris.
2. **Processus inverse (reverse)** : on apprend un réseau qui, à chaque niveau de bruit, sait **débruiter un peu**. En partant d'un bruit pur et en appliquant ce débruitage pas à pas, on obtient une nouvelle donnée.

L'idée clé est qu'apprendre un gros saut « bruit → image » est difficile, alors qu'apprendre des milliers de petits débruitages est simple : chaque petit pas inverse est approximativement gaussien.

```
x_0 (donnée) ──q──▶ x_1 ──q──▶ … ──q──▶ x_T ≈ N(0, I)
x_0 ◀──p_θ── x_1 ◀──p_θ── … ◀──p_θ── x_T
```

## Trois points de vue sur le même objet

Tout l'intérêt (et toute la confusion) de la littérature vient de ce que le même modèle peut se lire de trois façons. Elles mènent à la même fonction de perte, à des pondérations près.

| Point de vue | Ce que prédit le réseau | Article fondateur | Note |
|---|---|---|---|
| **Modèle à variables latentes** (VAE hiérarchique, ELBO) | le bruit $\epsilon$ ajouté | DDPM (Ho et al., 2020) | [01](01_ddpm.md) |
| **Score matching** | le score $\nabla_x \log p_t(x)$ | NCSN (Song & Ermon, 2019) | [02](02_score_sde.md) |
| **Équations différentielles stochastiques** | le score, en temps continu | Score SDE (Song et al., 2021) | [02](02_score_sde.md) |

Le lien central : prédire le bruit revient à prédire le score, au facteur près $s_\theta(x_t, t) = -\epsilon_\theta(x_t, t) / \sqrt{1-\bar\alpha_t}$.

## Les grands axes de variantes

Une fois le cadre de base compris, presque toutes les variantes jouent sur l'un de ces leviers :

| Levier | Question | Exemples | Note |
|---|---|---|---|
| **Planning de bruit** | Comment répartir le bruit entre $t=0$ et $t=T$ ? | linéaire, cosinus, EDM | [01](01_ddpm.md), [05](05_variantes.md) |
| **Paramétrisation** | Que prédit le réseau ? | $\epsilon$, $x_0$, $v$, vitesse (flow matching) | [01](01_ddpm.md), [05](05_variantes.md) |
| **Échantillonneur** | Comment remonter le temps en peu de pas ? | DDPM, DDIM, Heun, DPM-Solver | [03](03_ddim_echantillonnage.md) |
| **Conditionnement** | Comment contrôler ce qu'on génère ? | classifier guidance, classifier-free guidance | [04](04_guidance.md) |
| **Espace** | Diffuser dans l'espace pixel ou un espace latent ? | Latent Diffusion / Stable Diffusion | [05](05_variantes.md) |
| **Architecture** | Quel réseau ? | U-Net, DiT (transformer) | [05](05_variantes.md) |
| **Nombre de pas** | Peut-on générer en 1 à 4 pas ? | distillation, consistency models | [05](05_variantes.md) |

## Notations utilisées dans toutes les notes

| Symbole | Signification |
|---|---|
| $x_0$ | donnée propre |
| $x_t$ | donnée bruitée au pas $t \in \{1, \dots, T\}$ |
| $\epsilon \sim \mathcal{N}(0, I)$ | bruit gaussien standard |
| $\mathcal{N}(x;\ \mu,\ \Sigma)$ | densité de la loi normale de moyenne $\mu$ et covariance $\Sigma$, évaluée en $x$ |
| $\beta_t$ | variance du bruit ajouté au pas $t$ |
| $\alpha_t = 1 - \beta_t$ | |
| $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ | fraction de « signal » (au carré) restant au pas $t$ |
| $\epsilon_\theta(x_t, t)$ | réseau qui prédit le bruit |
| $s_\theta(x_t, t)$ | réseau qui prédit le score |
| $c$ ou $y$ | conditionnement (classe, texte…) |

> **Attention aux indices.** Dans les notes, $t$ va de $1$ à $T$ comme dans l'article DDPM. Dans le code, les tableaux sont indexés de $0$ à $T-1$ : l'indice `t` du code correspond au pas $t+1$ des notes.

## Parcours de lecture conseillé

1. [01 — DDPM](01_ddpm.md) : le processus direct, la perte simplifiée, l'algorithme d'échantillonnage. Implémenter en parallèle `src/diffusion_lab/diffusion.py` et lancer `scripts/train_toy.py`.
2. [03 — DDIM et échantillonnage rapide](03_ddim_echantillonnage.md) : même modèle entraîné, 20 à 50 pas au lieu de 1000.
3. [04 — Guidance](04_guidance.md) : générer une classe choisie ; c'est ce qui rend les modèles texte-image utilisables.
4. [02 — Score et SDE](02_score_sde.md) : le cadre continu qui unifie tout ; indispensable pour lire EDM, DPM-Solver, flow matching.
5. [05 — Panorama des variantes](05_variantes.md) : feuille de route des prochaines implémentations.

Les références complètes sont dans [references.md](references.md).
