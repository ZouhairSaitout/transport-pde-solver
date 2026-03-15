# EDP de transfert - Schémas aux différences finies

Implémentation et analyse comparative de quatre schémas aux différences finies pour l'équation de transport 1D, avec un dashboard interactif pour explorer la stabilité et la convergence.

## C'est quoi ce projet

L'équation de transport advective

$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$

est l'un des cas tests fondamentaux en analyse numérique des EDP. Sa simplicité apparente cache des comportements riches : diffusion numérique, instabilités de Von Neumann, oscillations parasites. Ce projet implémente quatre schémas classiques et les compare honnêtement sur plusieurs profils initiaux.

Trois questions ont guidé l'analyse :
- Pourquoi le schéma upwind est-il *exact* à CFL = 1, mais plus on raffine le pas de temps, plus il devient diffusif ?
- Lax-Wendroff est d'ordre 2. Dans quels cas ça compte vraiment, et quand est-ce que ça se retourne contre lui ?
- Pourquoi le schéma centré est-il inconditionnellement instable, quelle que soit la valeur de CFL ?

## Structure

```
├── solvers.py       # 4 schémas + 4 conditions initiales (créneau, gaussienne, sinus, triangle)
├── analysis.py      # benchmark (L1/L2/L∞), graphes de convergence, analyse de stabilité
└── app.py           # dashboard Streamlit interactif
```

## Les quatre schémas

**Euler Upwind** est le schéma de référence. Ordre 1, stable si CFL ≤ 1, il introduit une diffusion numérique proportionnelle à `Δx(1 - CFL)`. La conséquence directe : CFL = 1 annule exactement cette diffusion et le schéma transporte la condition initiale sans déformation. Réduire le pas de temps dégrade la solution, ce qui peut sembler contre-intuitif mais reste mathématiquement cohérent.

**Lax-Friedrichs** est robuste mais très diffusif. Il remplace la valeur centrale par une moyenne des voisins, ce qui stabilise les discontinuités au prix d'un lissage important. C'est le bon choix quand la solution présente des chocs et que la précision n'est pas prioritaire.

**Lax-Wendroff** atteint l'ordre 2 par un développement de Taylor en temps. Sur des profils lisses (gaussienne, sinus), il est clairement supérieur. Mais à CFL = 1, des oscillations numériques apparaissent derrière les fronts raides : il s’agit d’un artefact de la précision d’ordre 2 qui surcorrige.

**Euler Centré** est inconditionnellement instable. L'analyse de Von Neumann le confirme : le facteur d'amplification vaut `|g|² = 1 + (CFL·sin(kΔx))²`, toujours ≥ 1. Aucune valeur de CFL ne peut stabiliser ce schéma. Il est inclus précisément pour illustrer ce point.

## Résultats clés

Le benchmark sur quatre profils (créneau, gaussienne, sinus, triangle) confirme les prédictions théoriques. Lax-Wendroff domine sur les profils lisses grâce à son ordre 2. Lax-Friedrichs reste le plus robuste sur les discontinuités malgré une diffusion plus forte. Upwind à CFL = 1 est compétitif sur tous les profils à coût très faible.

Le comportement d'Euler Centré n'est pas un bug : le résidu explose en quelques itérations, quelle que soit la configuration. C'est le résultat attendu.

## Dashboard

Le dashboard Streamlit propose quatre vues :

- **Explorateur CFL** : simulation en temps réel avec slider CFL, choix du schéma et du profil initial.
- **Benchmark** : tableau comparatif L1/L2/L∞/CPU avec mise en évidence automatique des meilleurs scores.
- **Convergence** : graphe log-log avec estimation empirique de l'ordre numérique qui permet de vérifier que Lax-Wendroff atteint effectivement l'ordre 2 sur maillage raffiné.
- **Stabilité** : heatmap amplitude × CFL, détection automatique des explosions numériques.

**[▶ Live Dashboard](link)**

## Limites

Les quatre schémas sont explicites et conditionnellement stables (sauf Euler Centré, inconditionnellement instable). Pour des problèmes raides ou des domaines multidimensionnels, on se tournerait vers des schémas implicites ou des méthodes de type Runge-Kutta d'ordre élevé avec limiteurs de flux.
