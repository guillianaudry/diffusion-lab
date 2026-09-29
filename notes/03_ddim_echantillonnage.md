# 03 — DDIM et échantillonnage rapide

> Song, Meng, Ermon (2020), *Denoising Diffusion Implicit Models*, arXiv:2010.02502.
> Code : `GaussianDiffusion.ddim_sample_loop` dans `src/diffusion_lab/diffusion.py`.

## 1. Le problème

DDPM nécessite autant d'évaluations du réseau que de pas d'entraînement ($T = 1000$). Générer une image prend donc 1000 passes d'un U-Net. Peut-on réutiliser **le même modèle entraîné** avec beaucoup moins de pas ?

## 2. L'observation clé

La perte $L_\text{simple}$ ne dépend que des marginales $q(x_t \mid x_0)$, pas de la chaîne de Markov qui les relie. Il existe donc toute une famille de processus directs, **non markoviens**, qui ont les mêmes marginales, et pour lesquels le même $\epsilon_\theta$ est optimal.

DDIM définit cette famille via le postérieur, indexé par $\sigma_t \ge 0$ :

$$
q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\Big(\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\ \ \sigma_t^2 I\Big).
$$

On vérifie par récurrence que $q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$ pour tout choix de $\sigma$.

## 3. La mise à jour DDIM

En remplaçant $x_0$ par la prédiction $\hat x_0$ et le bruit par $\hat\epsilon = \epsilon_\theta(x_t, t)$ :

$$
x_{t-1} =
\underbrace{\sqrt{\bar\alpha_{t-1}}\;\hat x_0}_{\text{donnée prédite}}
+ \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\hat\epsilon}_{\text{direction vers } x_t}
+ \underbrace{\sigma_t\, z}_{\text{bruit frais}},
\qquad
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.
$$

On paramètre la stochasticité par $\eta \in [0, 1]$ :

$$
\sigma_t(\eta) = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}.
$$

- $\eta = 1$ : $\sigma_t^2 = \tilde\beta_t$, on retrouve **exactement DDPM** (vérifié par `tests/test_samplers.py::test_ddim_eta1_matches_ddpm_step`).
- $\eta = 0$ : pas de bruit, l'échantillonnage devient **déterministe**. C'est le DDIM proprement dit (« implicit » : le modèle est une fonction du bruit initial, comme un GAN).

## 4. Sauter des pas

Rien dans la formule n'impose de passer de $t$ à $t-1$ : elle reste valable entre deux temps quelconques $\tau_i > \tau_{i-1}$ en remplaçant $\bar\alpha_{t-1}$ par $\bar\alpha_{\tau_{i-1}}$. On choisit une sous-suite de $S \ll T$ pas (par exemple 50 pas régulièrement espacés parmi 1000) et on applique la mise à jour.

**Ce que l'on perd en réduisant les pas.** Même avec le débruiteur *exact* (`analytic.py`), sur un mélange de deux gaussiennes d'écart-type 0,30 et avec le planning linéaire, DDIM $\eta=0$ produit un écart-type d'environ 0,28 en 50 pas mais d'environ 0,24 en 20 pas (`tests/test_samplers.py::test_ddim_few_steps_discretization_bias`). L'erreur ne vient pas du réseau mais de la discrétisation de l'ODE : les pas d'Euler trop grands contractent la distribution. C'est ce que corrigent les solveurs d'ordre supérieur et un meilleur choix des temps (section 7).

Ordres de grandeur (article DDIM, CIFAR-10) : avec $\eta = 0$, 50 pas donnent une qualité proche de 1000 pas de DDPM, alors qu'avec $\eta = 1$ la qualité s'effondre quand on réduit le nombre de pas. **Le bruit injecté aide quand on fait beaucoup de pas, et nuit quand on en fait peu.**

## 5. DDIM est un solveur d'ODE

En posant $\bar x = x/\sqrt{\bar\alpha}$ et $\bar\sigma = \sqrt{1-\bar\alpha}/\sqrt{\bar\alpha}$, la mise à jour à $\eta = 0$ s'écrit

$$
\bar x_{t-1} = \bar x_t + (\bar\sigma_{t-1} - \bar\sigma_t)\,\epsilon_\theta(x_t, t),
$$

soit **un pas d'Euler** sur l'ODE $d\bar x / d\bar\sigma = \epsilon_\theta$, qui est l'ODE de flot de probabilité de la note [02](02_score_sde.md) dans ces variables. Cette lecture ouvre la porte aux solveurs d'ordre supérieur.

## 6. Conséquences utiles

- **Encodage / inversion** : puisque l'échantillonnage est une fonction déterministe, on peut l'inverser en appliquant la mise à jour dans l'autre sens ($t \to t+1$). On obtient un code latent $x_T$ qui régénère l'image. Base de nombreuses méthodes d'édition (*null-text inversion*, *prompt-to-prompt*).
- **Interpolation sémantique** : interpoler (sphériquement) entre deux $x_T$ donne des transitions régulières entre images.
- **Cohérence** : le même $x_T$ donne une image similaire quel que soit le nombre de pas.

## 7. Au-delà de DDIM : solveurs d'ordre supérieur

| Solveur | Idée | Évaluations typiques |
|---|---|---|
| DDIM ($\eta=0$) | Euler sur l'ODE | 50–100 |
| Heun (Karras et al., EDM) | Euler + correction trapèze, ordre 2 | ~35 |
| DPM-Solver / DPM-Solver++ (Lu et al.) | intègre exactement la partie linéaire de l'ODE, ordres 2–3 | 10–20 |
| UniPC, DEIS | multi-pas, prédicteur-correcteur | 10–20 |

Le choix des temps compte presque autant que le solveur : espacement uniforme, quadratique, ou en $\log \mathrm{SNR}$ (voir EDM, note [05](05_variantes.md)).

## 8. Exercices

1. Vérifier que $\sigma_t(\eta=1)^2 = \tilde\beta_t$.
2. Sur `moons`, tracer la qualité (distance de Wasserstein ou simple inspection visuelle) en fonction du nombre de pas pour $\eta \in \{0;\ 0{,}5;\ 1\}$. Le script `train_toy.py` produit déjà la figure `ddim_steps.png` pour $\eta = 0$.
3. Implémenter `ddim_invert` (pas $t \to t+1$) et vérifier qu'encoder puis décoder un point retrouve le point de départ.
4. Implémenter l'échantillonneur de Heun en variables $(\bar x, \bar\sigma)$.
