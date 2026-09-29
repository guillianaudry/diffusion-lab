# 04 — Guidance : générer ce que l'on veut

> Dhariwal & Nichol (2021), *Diffusion Models Beat GANs on Image Synthesis*, arXiv:2105.05233 (classifier guidance).
> Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*, arXiv:2207.12598.
> Code : `training_loss(..., p_uncond=...)` et `guidance_scale` dans `src/diffusion_lab/diffusion.py`.

## 1. Conditionner un modèle de diffusion

La façon la plus directe : donner la condition $c$ (classe, texte…) en entrée du réseau, $\epsilon_\theta(x_t, t, c)$, et entraîner sur des paires $(x_0, c)$. Dans le code, la classe est encodée par un embedding ajouté à l'embedding de temps.

En pratique, un modèle simplement conditionné produit des échantillons **corrects mais peu typiques** de la classe demandée. La guidance permet d'**accentuer** le conditionnement, au prix de la diversité.

## 2. Le principe commun : la règle de Bayes sur les scores

$$
\nabla_x \log p_t(x \mid c) = \nabla_x \log p_t(x) + \nabla_x \log p_t(c \mid x).
$$

Le score conditionnel est le score inconditionnel plus un terme qui pousse $x$ vers les régions où la condition est vraisemblable. La guidance consiste à **amplifier ce second terme** par un facteur $w$ :

$$
\nabla_x \log \tilde p_t(x \mid c) = \nabla_x \log p_t(x) + w\,\nabla_x \log p_t(c \mid x),
$$

ce qui revient à échantillonner $\tilde p(x \mid c) \propto p(x)\,p(c \mid x)^w$ (au moins heuristiquement : les scores des marginales bruitées ne se composent pas exactement ainsi). Pour $w > 1$, la distribution se concentre sur les $x$ où le classifieur est très confiant.

## 3. Classifier guidance

On entraîne un **classifieur sur données bruitées** $p_\phi(c \mid x_t, t)$ et on utilise son gradient. En paramétrisation $\epsilon$ ($\epsilon = -\sqrt{1-\bar\alpha_t}\cdot$ score) :

$$
\hat\epsilon(x_t, t, c) = \epsilon_\theta(x_t, t) - w\,\sqrt{1-\bar\alpha_t}\;\nabla_{x_t} \log p_\phi(c \mid x_t, t).
$$

Inconvénients : un second réseau à entraîner sur des données bruitées, et le gradient d'un classifieur peut ressembler à une attaque adverse (il améliore le score du classifieur sans améliorer l'image).

## 4. Classifier-free guidance (CFG)

Idée : obtenir le classifieur implicite à partir du modèle de diffusion lui-même, puisque

$$
\nabla_x \log p(c \mid x) = \nabla_x \log p(x \mid c) - \nabla_x \log p(x).
$$

**Entraînement** : un seul réseau apprend les deux scores. Avec une probabilité $p_\text{uncond}$ (typiquement 0,1 à 0,2), on remplace la condition par un jeton « vide » $\varnothing$. Dans le code, ce jeton est la classe d'indice `num_classes`.

**Échantillonnage** : on évalue le réseau deux fois (avec et sans condition) et on extrapole :

$$
\boxed{\ \tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + s\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big)\ }
$$

- $s = 0$ : inconditionnel ;
- $s = 1$ : conditionnel simple, sans guidance ;
- $s > 1$ : guidance ; valeurs typiques 3 à 8 pour la génération texte-image.

> **Deux conventions coexistent.** Ho & Salimans écrivent $\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t, c) - w\,\epsilon_\theta(x_t, \varnothing)$, soit $s = 1 + w$. La plupart des bibliothèques (dont ce dépôt, via `guidance_scale`) utilisent $s$. Toujours vérifier laquelle un article emploie avant de comparer des valeurs.

Comme la combinaison est linéaire, on peut l'appliquer indifféremment à la sortie $\epsilon$, $x_0$ ou $v$ du réseau : c'est ce que fait `model_predictions`. Les deux évaluations se font en un seul passage en concaténant le lot conditionnel et le lot inconditionnel.

## 5. Le compromis qualité / diversité

| $s$ | Effet |
|---|---|
| faible | grande diversité, échantillons parfois peu reconnaissables |
| moyen | meilleur FID en général |
| fort | échantillons très typiques, saturés, peu variés ; artefacts possibles |

Exemple calculable avec le débruiteur exact (`analytic.py`) : deux gaussiennes d'écart-type 0,30 centrées en $(-1{,}5;\ 0)$ et $(1{,}5;\ 0{,}5)$. En demandant la seconde avec $s = 4$ (DDIM, 50 pas), la moyenne obtenue est d'environ $(1{,}88;\ 0{,}57)$ et l'écart-type selon $x$ tombe à 0,16 : la guidance **repousse** les échantillons loin de l'autre classe (au-delà du vrai centre) et **réduit** la diversité dans la direction qui sépare les classes. L'écart-type dans la direction orthogonale reste presque inchangé.

Sur les données 2D du dépôt (`8gaussians` avec étiquettes), `train_toy.py --cond --guidance 1 3 6` montre concrètement l'effet : quand $s$ augmente, les points se concentrent au cœur de la gaussienne demandée, voire la dépassent (extrapolation hors du support).

## 6. Variantes et améliorations

- **Negative prompt** : remplacer $\varnothing$ par une condition à éviter.
- **Guidance dynamique / par intervalle** : n'appliquer la guidance que sur une plage de niveaux de bruit (Kynkäänniemi et al., 2024) ; réduit la perte de diversité.
- **CFG rescale** (Lin et al., 2023) : corrige la sur-saturation aux grandes échelles.
- **Distillation de guidance** (Meng et al., 2023) : un seul modèle qui intègre $s$ en entrée, pour ne faire qu'une passe.

## 7. Exercices

1. Montrer que la combinaison CFG appliquée à la sortie $v$ équivaut à l'appliquer sur $\epsilon$.
2. Avec le débruiteur analytique d'un mélange de gaussiennes (`analytic.py`), comparer la distribution obtenue avec $s = 3$ à la loi $p(x)\,p(c|x)^{3}$ : dans quelle mesure l'heuristique est-elle exacte ?
3. Entraîner avec $p_\text{uncond} \in \{0{,}05;\ 0{,}2;\ 0{,}5\}$ et observer l'effet sur la qualité inconditionnelle et conditionnelle.
