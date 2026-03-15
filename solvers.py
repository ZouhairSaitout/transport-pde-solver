"""
Quatre schémas aux différences finies pour l'équation de transport 1D :
    du/dt + c * du/dx = 0

Schémas disponibles
-------------------
- Euler Upwind      : ordre 1, stable si CFL <= 1, diffusif
- Euler Centré      : ordre 2, inconditionnellement INSTABLE
- Lax-Friedrichs    : ordre 1, stable si CFL <= 1, très diffusif
- Lax-Wendroff      : ordre 2, stable si CFL <= 1, oscillations près des chocs

Conditions initiales
--------------------
- gaussian(mu, sigma)       : profil gaussien lisse
- sine(k)                   : profil sinusoïdal
- square(x_left, x_right)   : fonction créneau discontinue
- triangle(x_peak, hw)      : chapeau linéaire par morceaux
"""

import numpy as np


# Conditions initiales


def gaussian(mu=0.3, sigma=0.05):
    """Profil gaussien centré en mu avec écart-type sigma."""
    def u0(x):
        return np.exp(-0.5 * ((np.asarray(x, dtype=float) - mu) / sigma) ** 2)
    u0.__name__ = "gaussian"
    return u0


def sine(k=2):
    """Profil sinusoïdal : sin(2*pi*k*x)."""
    def u0(x):
        return np.sin(2 * np.pi * k * np.asarray(x, dtype=float))
    u0.__name__ = "sine"
    return u0


def square(x_left=0.2, x_right=0.4):
    """Fonction créneau : 1 dans [x_left, x_right], 0 ailleurs."""
    def u0(x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= x_left) & (x <= x_right), 1.0, 0.0)
    u0.__name__ = "square"
    return u0


def triangle(x_peak=0.3, half_width=0.1):
    """Chapeau triangulaire : pic de 1 en x_peak, base de largeur 2*half_width."""
    def u0(x):
        x = np.asarray(x, dtype=float)
        return np.maximum(0.0, 1.0 - np.abs(x - x_peak) / half_width)
    u0.__name__ = "triangle"
    return u0


# Registre des profils (utilisé par analysis.py et app.py)
PROFILES = {
    "Gaussienne": gaussian(),
    "Sinus":      sine(),
    "Créneau":    square(),
    "Triangle":   triangle(),
}

# Noms affichés dans le dashboard
SCHEME_NAMES = {
    "upwind":         "Euler Upwind",
    "centered":       "Euler Centré",
    "lax_friedrichs": "Lax-Friedrichs",
    "lax_wendroff":   "Lax-Wendroff",
}

# Couleurs Plotly associées à chaque schéma
SCHEME_COLORS = {
    "upwind":         "#4C6EF5",
    "centered":       "#F03E3E",
    "lax_friedrichs": "#20C997",
    "lax_wendroff":   "#F59F00",
}


# Solveurs

def exact_solution(u0, x, T, c):
    """Solution exacte par la méthode des caractéristiques : u(x, T) = u0(x - c*T)."""
    return u0(x - c * T)


def solver_upwind(u0, L, T, c, nx, nt):
    """
    Schéma Euler Upwind (décentré amont).
        u_i^{n+1} = u_i^n - CFL * (u_i^n - u_{i-1}^n)

    Stable si CFL <= 1. Exact quand CFL = 1 (pas de mélange entre cellules).
    Diffusion numérique ~ (c*dx/2) * (1 - CFL) : augmente quand CFL diminue.
    """
    dx, dt = L / (nx - 1), T / nt
    cfl = c * dt / dx          # nombre de Courant-Friedrichs-Lewy
    x = np.linspace(0, L, nx)
    u = u0(x)
    unew = np.empty_like(u)
    for n in range(nt):
        # Points intérieurs et bord droit : différence amont
        unew[1:]  = u[1:] - cfl * (u[1:] - u[:-1])
        # Bord entrant (x=0) : solution exacte imposée
        unew[0]   = u0(-c * (n + 1) * dt)
        u[:] = unew
    return x, u


def solver_centered(u0, L, T, c, nx, nt):
    """
    Schéma Euler Centré.
        u_i^{n+1} = u_i^n - (CFL/2) * (u_{i+1}^n - u_{i-1}^n)

    INCONDITIONNELLEMENT INSTABLE pour tout CFL > 0.
    Inclus uniquement à titre de comparaison pédagogique.
    L'analyse de Von Neumann montre |g(xi)| > 1 pour tout nombre d'onde xi.
    """
    dx, dt = L / (nx - 1), T / nt
    cfl = c * dt / dx
    x = np.linspace(0, L, nx)
    u = u0(x)
    unew = np.empty_like(u)
    for n in range(nt):
        # Différence centrée : regarde les deux voisins symétriquement
        unew[1:-1] = u[1:-1] - 0.5 * cfl * (u[2:] - u[:-2])
        # Conditions aux bords : solution exacte imposée des deux côtés
        unew[0]    = u0(-c * (n + 1) * dt)
        unew[-1]   = u0(L - c * (n + 1) * dt)
        u[:] = unew
    return x, u


def solver_lax_friedrichs(u0, L, T, c, nx, nt):
    """
    Schéma de Lax-Friedrichs.
        u_i^{n+1} = (1/2)(u_{i+1}^n + u_{i-1}^n) - (CFL/2)(u_{i+1}^n - u_{i-1}^n)

    Stable si CFL <= 1. Le plus diffusif des quatre schémas.
    Le terme de moyenne (u_{i+1} + u_{i-1})/2 introduit une diffusion artificielle
    proportionnelle à dx²/dt très forte pour de petits pas de temps.
    """
    dx, dt = L / (nx - 1), T / nt
    cfl = c * dt / dx
    x = np.linspace(0, L, nx)
    u = u0(x)
    unew = np.empty_like(u)
    for n in range(nt):
        # Moyenne des voisins + terme de transport centré
        unew[1:-1] = 0.5 * (u[2:] + u[:-2]) - (cfl / 2) * (u[2:] - u[:-2])
        unew[0]    = u0(-c * (n + 1) * dt)
        unew[-1]   = u0(L - c * (n + 1) * dt)
        u[:] = unew
    return x, u


def solver_lax_wendroff(u0, L, T, c, nx, nt):
    """
    Schéma de Lax-Wendroff.
        u_i^{n+1} = u_i^n
                  - (CFL/2)(u_{i+1}^n - u_{i-1}^n)             <- terme centré ordre 1
                  + (CFL²/2)(u_{i+1}^n - 2u_i^n + u_{i-1}^n)  <- correction ordre 2

    Stable si CFL <= 1. Ordre 2 en espace et en temps.
    Le terme correctif en CFL² annule l'erreur de troncature du premier ordre.
    Produit des oscillations de Gibbs au voisinage des discontinuités.
    """
    dx, dt = L / (nx - 1), T / nt
    cfl = c * dt / dx
    x = np.linspace(0, L, nx)
    u = u0(x)
    unew = np.empty_like(u)
    for n in range(nt):
        unew[1:-1] = (
            u[1:-1]
            - (cfl / 2)    * (u[2:] - u[:-2])            # différence centrée
            + (cfl**2 / 2) * (u[2:] - 2*u[1:-1] + u[:-2])  # correction dispersive
        )
        unew[0]  = u0(-c * (n + 1) * dt)
        unew[-1] = u0(L - c * (n + 1) * dt)
        u[:] = unew
    return x, u


# Registre des solveurs (utilisé par analysis.py et app.py)
SOLVERS = {
    "upwind":         solver_upwind,
    "centered":       solver_centered,
    "lax_friedrichs": solver_lax_friedrichs,
    "lax_wendroff":   solver_lax_wendroff,
}
