# 01 — DDPM : Denoising Diffusion Probabilistic Models

> Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
> Code : `src/diffusion_lab/diffusion.py` (`q_sample`, `training_loss`, `p_sample_loop`), `src/diffusion_lab/schedules.py`.

## 1. Le processus direct

On définit une chaîne de Markov qui ajoute un peu de bruit gaussien à chaque pas :

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big), \qquad t = 1, \dots, T.
$$

Le facteur $\sqrt{1-\beta_t}$ n'est pas décoratif : il rétrécit le signal pour que la variance totale reste bornée (si $\mathrm{Var}(x_{t-1}) = 1$, alors $\mathrm{Var}(x_t) = (1-\beta_t) + \beta_t = 1$). C'est pour cela qu'on parle de processus *variance preserving* (VP).

### Échantillonnage direct à n'importe quel pas

En posant $\alpha_t = 1-\beta_t$ et $\bar\alpha_t = \prod_{s \le t} \alpha_s$, une récurrence sur les sommes de gaussiennes indépendantes donne une forme fermée :

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\big)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ \ \epsilon \sim \mathcal{N}(0, I).
$$

*Esquisse :* $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t}$ et $x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}$. Les deux bruits se combinent en un seul bruit de variance $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$. On itère.

**Conséquence pratique :** pour entraîner, on n'a jamais besoin de simuler la chaîne. On tire $t$ au hasard et on fabrique $x_t$ en une ligne (`q_sample`).

Quand $\bar\alpha_T \approx 0$, $x_T \approx \mathcal{N}(0, I)$ quelle que soit la donnée de départ. Le rapport signal/bruit $\mathrm{SNR}(t) = \bar\alpha_t / (1-\bar\alpha_t)$ décroît de $+\infty$ à $\approx 0$.

## 2. Le postérieur du processus direct

Le vrai processus inverse $q(x_{t-1} \mid x_t)$ est inaccessible (il dépend de toute la distribution des données). Mais **conditionné à $x_0$**, il est gaussien et calculable par la règle de Bayes :

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\big),
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t.
$$

Dans le code, ces deux coefficients sont `posterior_mean_coef1` et `posterior_mean_coef2`, et $\tilde\beta_t$ est `posterior_variance`.

## 3. Le modèle inverse et la borne variationnelle (ELBO)

On paramètre le processus inverse par des gaussiennes :

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2 I\big), \qquad p(x_T) = \mathcal{N}(0, I).
$$

Comme pour un VAE, on maximise une borne inférieure de la log-vraisemblance. Elle se décompose en une somme de KL entre gaussiennes :

$$
-\log p_\theta(x_0) \le
\underbrace{D_\text{KL}\big(q(x_T|x_0)\,\|\,p(x_T)\big)}_{L_T\ \text{(constant)}}
+ \sum_{t=2}^{T} \underbrace{\mathbb{E}_q\, D_\text{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}}
\underbrace{-\ \mathbb{E}_q \log p_\theta(x_0|x_1)}_{L_0}.
$$

Avec une variance $\sigma_t^2$ fixée, chaque KL entre gaussiennes se réduit à un écart entre moyennes :

$$
L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\,\big\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\big\|^2\right] + C.
$$

Le modèle apprend donc à **prédire la moyenne du postérieur**, c'est-à-dire à deviner où était la donnée un pas plus tôt.

## 4. La paramétrisation $\epsilon$ et la perte simplifiée

En remplaçant $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ dans $\tilde\mu_t$, on obtient :

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right).
$$

Ho et al. choisissent donc de faire prédire **le bruit** au réseau et de poser
$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\big)$.
La perte devient :

$$
L_{t-1} = \mathbb{E}\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\,\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2\right].
$$

Le résultat empirique principal de l'article : **supprimer la pondération** donne de meilleurs échantillons.

$$
\boxed{\ L_\text{simple} = \mathbb{E}_{t \sim \mathcal{U}\{1..T\},\ x_0,\ \epsilon}\ \big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\big)\big\|^2\ }
$$

C'est une simple régression : « voici une image bruitée et le niveau de bruit, retrouve le bruit ». Supprimer la pondération donne relativement plus de poids aux grands $t$ (bruit fort), qui sont les plus utiles pour la qualité perceptuelle.

## 5. Les deux algorithmes

**Entraînement** (`GaussianDiffusion.training_loss`)

```
répéter
    x_0 ~ données
    t ~ Uniforme{1, …, T}
    ε ~ N(0, I)
    descente de gradient sur ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖²
```

**Échantillonnage** (`GaussianDiffusion.p_sample_loop`)

```
x_T ~ N(0, I)
pour t = T, …, 1
    z ~ N(0, I) si t > 1, sinon z = 0
    x_{t−1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)) + σ_t z
renvoyer x_0
```

Dans le code, on passe par la prédiction $\hat x_0$ (puis $\tilde\mu_t(x_t, \hat x_0)$), ce qui est mathématiquement identique et permet de **tronquer** $\hat x_0$ dans $[-1, 1]$ pour les images (`clip_x0=True`). Cette astuce stabilise nettement l'échantillonnage.

### Choix de la variance $\sigma_t^2$

Deux choix sont justifiés, et donnent des résultats similaires en pratique avec $T = 1000$ :

- $\sigma_t^2 = \beta_t$ : optimal si $x_0 \sim \mathcal{N}(0, I)$ (`var_type="beta"`) ;
- $\sigma_t^2 = \tilde\beta_t$ : optimal si $x_0$ est un point fixe (`var_type="posterior"`, défaut).

Improved DDPM (note [05](05_variantes.md)) apprend une interpolation entre les deux, ce qui devient important quand on échantillonne en peu de pas.

## 6. Plannings de bruit

**Linéaire** (DDPM) : $\beta_t$ croît linéairement de $10^{-4}$ à $0{,}02$ avec $T = 1000$. Pour un autre $T$, on multiplie les bornes par $1000/T$ afin de garder un $\bar\alpha_T$ comparable.

**Cosinus** (Nichol & Dhariwal, 2021) : on définit directement $\bar\alpha_t$

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right), \qquad s = 0{,}008,
$$

puis $\beta_t = 1 - \bar\alpha_t/\bar\alpha_{t-1}$, tronqué à $0{,}999$. Avec le planning linéaire, $\bar\alpha_t$ chute très vite et les derniers pas sont presque du bruit pur, donc gaspillés ; le cosinus détruit l'information plus régulièrement, ce qui aide surtout pour les petites images (32×32, 28×28).

`scripts/train_toy.py --plot-schedules` trace $\bar\alpha_t$ et le SNR des deux plannings.

## 7. Autres paramétrisations : $x_0$ et $v$

Puisque $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, connaître $x_t$ et l'une des deux quantités donne l'autre. Le réseau peut donc prédire au choix :

| Cible | Définition | Récupérer $\hat x_0$ | Récupérer $\hat\epsilon$ |
|---|---|---|---|
| $\epsilon$ | bruit | $(x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ | — |
| $x_0$ | donnée propre | — | $(x_t - \sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$ |
| $v$ | $\sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | $\sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v$ | $\sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,\hat v$ |

- Prédire $\epsilon$ est mal conditionné à bruit très fort : quand $\bar\alpha_t \to 0$, diviser par $\sqrt{\bar\alpha_t}$ amplifie la moindre erreur sur $\hat x_0$.
- Prédire $x_0$ est mal conditionné à bruit faible (il faut recopier l'entrée presque parfaitement).
- La $v$-paramétrisation (Salimans & Ho, 2022) interpole entre les deux et reste stable partout ; elle est devenue le choix par défaut de nombreux modèles récents.

Toutes les trois sont implémentées (`prediction="eps" | "x0" | "v"`). Comparer les trois sur la même tâche est un bon premier exercice.

## 8. Lien avec le score (aperçu de la note 02)

Le score de la loi de transition est $\nabla_{x_t} \log q(x_t \mid x_0) = -\epsilon / \sqrt{1-\bar\alpha_t}$. La formule de Tweedie donne alors

$$
\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1-\bar\alpha_t)\,\nabla \log p_t(x_t)}{\sqrt{\bar\alpha_t}},
$$

et le débruiteur optimal vérifie $\epsilon^*(x_t, t) = -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t)$. **Entraîner DDPM, c'est faire du *denoising score matching* à tous les niveaux de bruit.**

Pour une donnée qui suit un mélange de gaussiennes, ce débruiteur optimal a une forme fermée : c'est ce qu'implémente `src/diffusion_lab/analytic.py`, utilisé dans les tests pour valider les échantillonneurs indépendamment de tout entraînement.

## 9. Points pratiques

- **Normaliser les données** : images dans $[-1, 1]$, données 2D centrées réduites. Le processus direct suppose une variance de l'ordre de 1.
- **Encodage du temps** : embedding sinusoïdal de $t$ (comme les positions d'un transformer) suivi d'un petit MLP, injecté dans chaque bloc du réseau.
- **EMA des poids** (décroissance 0,999 à 0,9999) : les échantillons issus des poids moyennés sont nettement meilleurs. Implémenté dans `utils.EMA`.
- **Hyperparamètres de référence** : $T = 1000$, Adam, lr $2\cdot10^{-4}$, batch 128.
- **La perte stagne vite** et reste bruitée, c'est normal : l'erreur irréductible dépend de $t$. Juger sur les échantillons, pas sur la courbe.

## 10. Exercices

1. Montrer que $\mathrm{Var}(x_t) = 1$ si $\mathrm{Var}(x_0) = 1$ (cas scalaire).
2. Retrouver $\tilde\mu_t$ et $\tilde\beta_t$ en complétant le carré dans $q(x_t|x_{t-1})\,q(x_{t-1}|x_0)$.
3. Vérifier numériquement que `p_sample_loop` avec le débruiteur analytique retrouve les moyennes du mélange (`tests/test_samplers.py`).
4. Entraîner le jeu `moons` avec `--prediction eps`, `x0` et `v` ; comparer les échantillons à nombre de pas d'entraînement égal.
5. Tracer la perte par tranche de $t$ : où le réseau se trompe-t-il le plus ?
