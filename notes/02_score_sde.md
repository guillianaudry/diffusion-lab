# 02 — Score matching et formulation SDE

> Song & Ermon (2019), *Generative Modeling by Estimating Gradients of the Data Distribution* (NCSN), arXiv:1907.05600.
> Song et al. (2021), *Score-Based Generative Modeling through Stochastic Differential Equations*, arXiv:2011.13456.
> Vincent (2011), *A Connection Between Score Matching and Denoising Autoencoders*.

## 1. Le score

Le **score** d'une densité est le gradient de son log : $s(x) = \nabla_x \log p(x)$. C'est un champ de vecteurs qui pointe vers les régions de forte densité.

Deux avantages majeurs :

- il ne dépend pas de la constante de normalisation : $\nabla_x \log \frac{\tilde p(x)}{Z} = \nabla_x \log \tilde p(x)$. On évite donc le problème central des modèles à énergie ;
- connaître le score suffit pour échantillonner, grâce à la **dynamique de Langevin** :

$$
x_{k+1} = x_k + \frac{\eta}{2}\,\nabla_x \log p(x_k) + \sqrt{\eta}\,z_k, \qquad z_k \sim \mathcal{N}(0, I).
$$

Quand $\eta \to 0$ et $k \to \infty$, $x_k$ suit $p$.

## 2. Apprendre le score : denoising score matching

On ne peut pas régresser directement sur $\nabla \log p_\text{data}$, qui est inconnu. Vincent (2011) montre qu'en bruitant les données, $\tilde x = x + \sigma \epsilon$, on peut régresser sur le score **de la loi de bruitage**, qui est connu :

$$
\mathbb{E}_{x,\,\epsilon}\left\| s_\theta(x + \sigma\epsilon) - \left(-\frac{\epsilon}{\sigma}\right) \right\|^2 .
$$

L'optimum est $s_\theta^*(\tilde x) = \nabla_{\tilde x} \log p_\sigma(\tilde x)$, le score de la distribution **bruitée** $p_\sigma = p_\text{data} * \mathcal{N}(0, \sigma^2 I)$.

C'est exactement la perte DDPM : avec $\sigma_t = \sqrt{1-\bar\alpha_t}$, prédire $-\epsilon/\sigma_t$, c'est prédire $\epsilon$ au facteur $-1/\sigma_t$ près.

## 3. Pourquoi plusieurs niveaux de bruit (NCSN)

Avec un seul petit $\sigma$, deux problèmes :

1. **Régions vides** : loin des données, on n'a aucun exemple, donc le score appris y est faux, et c'est justement là que démarre Langevin.
2. **Modes séparés** : Langevin traverse très mal les zones de faible densité entre deux modes et ne retrouve pas leurs poids relatifs.

Un grand bruit remplit l'espace et relie les modes, un petit bruit donne de la précision. NCSN apprend donc un seul réseau $s_\theta(x, \sigma)$ pour une suite décroissante $\sigma_1 > \dots > \sigma_L$, puis échantillonne par **Langevin recuit** : quelques pas de Langevin à chaque niveau, du plus bruité au moins bruité. DDPM et NCSN sont deux discrétisations du même principe.

## 4. Le passage au temps continu : les SDE

On remplace la suite discrète de niveaux de bruit par une équation différentielle stochastique (SDE) :

$$
dx = f(x, t)\,dt + g(t)\,dw, \qquad t \in [0, 1].
$$

**Résultat clé (Anderson, 1982)** : le processus renversé en temps est lui aussi une SDE, qui ne dépend de la donnée qu'à travers le score :

$$
dx = \big[f(x, t) - g(t)^2\,\nabla_x \log p_t(x)\big]\,dt + g(t)\,d\bar w .
$$

Il suffit donc d'apprendre $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ pour tout $t$, puis de résoudre cette SDE de $t = 1$ vers $t = 0$.

### Les deux familles classiques

| SDE | Équation | Discrétisation | Loi de $x_t$ sachant $x_0$ |
|---|---|---|---|
| **VP** (variance preserving) | $dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$ | DDPM | $\mathcal{N}\big(\sqrt{\bar\alpha(t)}\,x_0,\ (1-\bar\alpha(t))I\big)$, $\bar\alpha(t) = e^{-\int_0^t \beta}$ |
| **VE** (variance exploding) | $dx = \sqrt{\tfrac{d\,\sigma^2(t)}{dt}}\,dw$ | NCSN | $\mathcal{N}\big(x_0,\ \sigma^2(t)\,I\big)$ |

En VP le signal est atténué et la variance reste proche de 1. En VE le signal est intact et on ajoute un bruit de plus en plus grand ($\sigma_\text{max}$ de l'ordre de la distance maximale entre points de données).

## 5. L'ODE de flot de probabilité

Pour toute SDE, il existe une **équation différentielle ordinaire déterministe** qui a exactement les mêmes marginales $p_t$ :

$$
\frac{dx}{dt} = f(x, t) - \frac12\, g(t)^2\, \nabla_x \log p_t(x).
$$

Conséquences :

- **Échantillonnage déterministe** : bruit → donnée devient une fonction. On peut utiliser des solveurs d'ODE d'ordre élevé (Heun, RK45, DPM-Solver) et générer en 10 à 50 évaluations. DDIM avec $\eta=0$ en est une discrétisation (note [03](03_ddim_echantillonnage.md)).
- **Encodage** : on peut remonter l'ODE de la donnée vers le bruit (inversion), utile pour l'édition d'images.
- **Vraisemblance exacte** via la formule du changement de variable instantané (comme pour les *neural ODE*) :
  $\log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla\cdot \tilde f(x_t, t)\,dt$, où $\tilde f$ est le champ de l'ODE.

## 6. Prédicteur-correcteur

Song et al. proposent de combiner :

- un **prédicteur** : un pas de solveur numérique de la SDE inverse (Euler–Maruyama ou « reverse diffusion »),
- un **correcteur** : quelques pas de Langevin au temps courant, qui ramènent l'échantillon vers $p_t$ et corrigent l'erreur de discrétisation.

Utile surtout pour les SDE de type VE.

## 7. Dictionnaire DDPM ↔ score

| DDPM (discret) | Score / SDE (continu) |
|---|---|
| $\epsilon_\theta(x_t, t)$ | $-\sqrt{1-\bar\alpha_t}\; s_\theta(x_t, t)$ |
| $\bar\alpha_t$ | $\exp\!\big(-\int_0^t \beta(s)\,ds\big)$ |
| `p_sample_loop` | Euler–Maruyama sur la SDE inverse (VP) |
| DDIM, $\eta = 0$ | Euler sur l'ODE de flot de probabilité (dans les bonnes variables) |
| $L_\text{simple}$ | DSM pondéré par $\lambda(t) = 1-\bar\alpha_t$ |

## 8. Exercices

1. Vérifier que la discrétisation d'Euler–Maruyama de la VP-SDE avec $\beta(t)\,\Delta t = \beta_t$ redonne, au premier ordre, $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$.
2. Pour $p_\text{data} = \mathcal{N}(\mu, s^2)$ en 1D, calculer $\nabla \log p_t$ sous la VP-SDE et résoudre l'ODE de flot à la main.
3. Implémenter un échantillonneur par Langevin recuit en réutilisant un modèle entraîné par `train_toy.py` (via $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$). C'est une bonne première contribution au dépôt : `samplers/langevin.py`.
