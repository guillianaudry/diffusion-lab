# 05 — Panorama des variantes

Cette note sert de **feuille de route** : pour chaque variante, l'idée, l'équation qui la résume, et ce qu'il faudrait ajouter au dépôt pour l'implémenter. Le statut est tenu à jour dans le tableau du `README.md`.

---

## Improved DDPM — variance apprise et planning cosinus

> Nichol & Dhariwal (2021), arXiv:2102.09672.

- **Planning cosinus** (déjà implémenté, note [01](01_ddpm.md)).
- **Variance apprise** : le réseau sort en plus un vecteur $v$ et on interpole en échelle log entre les deux bornes :
  $\Sigma_\theta = \exp\big(v \log \beta_t + (1-v)\log\tilde\beta_t\big)$.
- **Perte hybride** : $L_\text{hybrid} = L_\text{simple} + \lambda\,L_\text{vlb}$ avec $\lambda = 0{,}001$ ; $L_\text{vlb}$ n'entraîne que la variance (gradient arrêté sur la moyenne).
- **Intérêt** : meilleure log-vraisemblance, et échantillonnage de bonne qualité en 50 à 100 pas avec l'échantillonneur DDPM stochastique.

*À implémenter* : sortie à `2 × canaux`, terme $L_\text{vlb}$ (KL entre gaussiennes + vraisemblance discrétisée pour $t=0$).

---

## EDM — « Elucidating the Design Space »

> Karras, Aittala, Aila, Laine (2022), arXiv:2206.00364.

Article de synthèse très recommandé : il sépare proprement les choix de conception qui étaient mélangés dans les travaux précédents.

- **Processus** : $x = x_0 + \sigma\,\epsilon$ (formulation VE), le temps est directement $\sigma$.
- **Préconditionnement** du réseau $F_\theta$ pour que ses entrées et sorties aient une variance unitaire à tout niveau de bruit :

$$
D_\theta(x; \sigma) = c_\text{skip}(\sigma)\,x + c_\text{out}(\sigma)\,F_\theta\big(c_\text{in}(\sigma)\,x;\ c_\text{noise}(\sigma)\big),
$$

$$
c_\text{skip} = \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2},\quad
c_\text{out} = \frac{\sigma\,\sigma_\text{data}}{\sqrt{\sigma^2+\sigma_\text{data}^2}},\quad
c_\text{in} = \frac{1}{\sqrt{\sigma^2+\sigma_\text{data}^2}},\quad
c_\text{noise} = \tfrac14 \ln\sigma.
$$

- **Distribution des niveaux de bruit à l'entraînement** : $\ln\sigma \sim \mathcal{N}(-1{,}2;\ 1{,}2^2)$, qui concentre l'effort sur les niveaux intermédiaires.
- **Échantillonnage** : solveur de Heun (ordre 2) sur l'ODE, avec des niveaux
  $\sigma_i = \big(\sigma_\text{max}^{1/\rho} + \tfrac{i}{N-1}(\sigma_\text{min}^{1/\rho} - \sigma_\text{max}^{1/\rho})\big)^\rho$, $\rho = 7$.
- **Résultat** : état de l'art sur CIFAR-10 à l'époque avec 35 évaluations.

*À implémenter* : `edm.py` avec préconditionnement, perte pondérée et échantillonneur de Heun. Réutilise les réseaux existants.

---

## Diffusion latente (Latent Diffusion Models, Stable Diffusion)

> Rombach et al. (2022), arXiv:2112.10752.

- Un **autoencodeur** (VAE avec perte perceptuelle et adversariale) compresse l'image d'un facteur spatial 4 à 8 ; la diffusion a lieu dans l'espace latent, bien plus petit.
- Le conditionnement texte passe par de la **cross-attention** entre les caractéristiques du U-Net et les tokens d'un encodeur de texte.
- **Intérêt** : un coût de calcul divisé par un ordre de grandeur, ce qui rend la haute résolution accessible.

*À implémenter* : entraîner un petit autoencodeur sur MNIST ou CIFAR-10 (ou charger un VAE pré-entraîné), puis réutiliser `GaussianDiffusion` tel quel sur les latents (penser à renormaliser les latents à variance 1).

---

## DiT — Diffusion Transformers

> Peebles & Xie (2022), arXiv:2212.09748.

- Remplace le U-Net par un **transformer** sur des patchs (des latents).
- Le temps et la classe modulent chaque bloc par **adaLN-Zero** (échelle et décalage de la LayerNorm prédits, initialisés à zéro).
- La qualité suit des lois d'échelle nettes avec les GFLOPs ; architecture de base de nombreux modèles récents.

*À implémenter* : `models/dit.py`, compatible avec l'interface `forward(x, t, y)`.

---

## Flow matching et rectified flow

> Lipman et al. (2022), *Flow Matching for Generative Modeling*, arXiv:2210.02747.
> Liu, Gong, Liu (2022), *Flow Straight and Fast* (rectified flow), arXiv:2209.03003.

- On apprend directement le **champ de vitesse** d'une ODE qui transporte le bruit vers les données. Avec le chemin linéaire (convention « $t=1$ = bruit ») :

$$
x_t = (1-t)\,x_0 + t\,\epsilon, \qquad
\mathcal{L} = \mathbb{E}_{t,\,x_0,\,\epsilon}\,\big\|v_\theta(x_t, t) - (\epsilon - x_0)\big\|^2 .
$$

- On échantillonne en intégrant $dx/dt = v_\theta$ de $t=1$ à $t=0$ (Euler ou Heun).
- **Lien avec la diffusion** : pour des chemins gaussiens, c'est une diffusion avec un planning et une paramétrisation particuliers ; la cible est proche de la $v$-paramétrisation.
- **Rectified flow / reflow** : ré-entraîner sur des couples (bruit, échantillon généré) redresse les trajectoires, ce qui permet de générer en très peu de pas.
- Utilisé par Stable Diffusion 3 et de nombreux modèles récents.

*À implémenter* : `flow_matching.py`. C'est la variante la plus simple à coder après DDPM, excellent deuxième projet.

---

## Génération en peu de pas : distillation et consistency models

> Salimans & Ho (2022), *Progressive Distillation*, arXiv:2202.00512.
> Song et al. (2023), *Consistency Models*, arXiv:2303.01469.

- **Distillation progressive** : un élève apprend à faire en 1 pas ce que le professeur fait en 2 ; on répète en divisant le nombre de pas par deux à chaque fois (1024 → 4 pas). C'est dans cet article qu'est introduite la $v$-paramétrisation.
- **Consistency models** : on apprend $f_\theta(x_t, t) \approx x_0$ pour tout point d'une même trajectoire de l'ODE, avec la contrainte $f_\theta(x, \epsilon) = x$. Entraînement par distillation (avec un modèle de diffusion existant) ou directement (*consistency training*). Génération en 1 ou 2 pas.

*À implémenter* : après avoir un bon modèle DDIM sur MNIST, la distillation progressive est un bon exercice.

---

## Autres lectures

- **VDM** — Kingma et al. (2021), *Variational Diffusion Models*, arXiv:2107.00630 : formulation en temps continu en fonction du SNR ; montre que la perte ne dépend du planning qu'à ses extrémités.
- **Diffusion discrète** (texte, graphes) : D3PM (Austin et al., 2021), diffusion masquée.
- **Guidance et contrôle** : ControlNet, IP-Adapter, inversion textuelle, DreamBooth, LoRA.

## Ordre d'implémentation suggéré

1. DDPM + DDIM + CFG sur données 2D puis MNIST (fait).
2. Flow matching (réutilise les réseaux, nouvelle perte et nouvel échantillonneur).
3. EDM : préconditionnement et Heun.
4. Improved DDPM : variance apprise.
5. Diffusion latente sur CIFAR-10.
6. DiT.
7. Distillation progressive ou consistency models.
