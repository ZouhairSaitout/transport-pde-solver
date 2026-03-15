"""
Analyse quantitative des quatre schémas numériques.

Fonctions
---------
- compute_errors(u_num, u_exact)   : erreurs L1, L2, L-infini
- run_benchmark(...)               : DataFrame complet des métriques
- convergence_study(...)           : ordres de convergence empiriques
- stability_map(...)               : amplitude max en fonction du CFL
"""

import time
import numpy as np
import pandas as pd

from solvers import SOLVERS, PROFILES, exact_solution

# Seuil de stabilité partagé entre run_benchmark, stability_map et app.py.
# Un schéma est instable si son amplitude max dépasse STABILITY_THRESHOLD * init_amp.
STABILITY_THRESHOLD = 5.0


# Calcul des erreurs

def compute_errors(u_num: np.ndarray, u_exact: np.ndarray) -> dict:
    """
    Calcule les erreurs L1, L2 et L-infini entre solution numérique et exacte.

    L1   = erreur absolue moyenne        (biais global)
    L2   = erreur quadratique moyenne    (métrique standard en numérique)
    Linf = erreur absolue maximale       (pire cas ponctuel)
    """
    diff = np.abs(u_num - u_exact)
    n    = len(diff)
    return {
        "L1":   float(np.sum(diff) / n),
        "L2":   float(np.sqrt(np.sum(diff ** 2) / n)),
        "Linf": float(np.max(diff)),
    }



# Benchmark

def run_benchmark(
    L: float = 1.0,
    T: float = 0.5,
    c: float = 1.0,
    nx: int  = 200,
    cfl_values: list = None,
) -> pd.DataFrame:
    """
    Lance toutes les combinaisons (schéma, profil, CFL) et retourne un DataFrame de métriques.

    Pour chaque combinaison on mesure :
    - Les erreurs L1, L2, Linf par rapport à la solution exacte
    - Le temps de calcul CPU en millisecondes
    - La stabilité (détection d'explosion numérique)

    Colonnes du DataFrame : scheme, profile, cfl, L1, L2, Linf, cpu_ms, stable
    """
    if cfl_values is None:
        cfl_values = [0.2, 0.5, 0.8, 0.9, 1.0]

    dx   = L / (nx - 1)
    rows = []

    for scheme_name, solver in SOLVERS.items():
        for profile_name, u0 in PROFILES.items():
            for cfl in cfl_values:
                # Calcul du pas de temps à partir du CFL cible
                dt         = cfl * dx / c
                nt         = max(1, int(np.ceil(T / dt)))
                dt_actual  = T / nt
                cfl_actual = round(c * dt_actual / dx, 4)  # CFL effectif après arrondi de nt

                x = np.linspace(0, L, nx)

                # Mesure du temps CPU
                t0 = time.perf_counter()
                try:
                    _, u_num = solver(u0, L, T, c, nx, nt)
                    cpu_ms   = (time.perf_counter() - t0) * 1000
                    max_amp  = float(np.max(np.abs(u_num)))
                    # Instable si la solution a grandi de plus de STABILITY_THRESHOLD fois l'amplitude initiale
                    init_amp = float(np.max(np.abs(u0(x)))) + 1e-12
                    stable   = max_amp < STABILITY_THRESHOLD * init_amp
                except Exception:
                    u_num   = np.zeros(nx)
                    cpu_ms  = float("nan")
                    stable  = False
                    max_amp = float("inf")

                u_exact = exact_solution(u0, x, T, c)
                errors  = compute_errors(u_num, u_exact) if stable else {"L1": None, "L2": None, "Linf": None}

                rows.append({
                    "scheme":  scheme_name,
                    "profile": profile_name,
                    "cfl":     cfl_actual,
                    "L1":      errors["L1"],
                    "L2":      errors["L2"],
                    "Linf":    errors["Linf"],
                    "cpu_ms":  round(cpu_ms, 2),
                    "stable":  stable,
                })

    return pd.DataFrame(rows)


# Étude de convergence

def convergence_study(
    L: float = 1.0,
    T: float = 0.5,
    c: float = 1.0,
    cfl: float = 0.9,
    nx_values: list = None,
) -> pd.DataFrame:
    """
    Fait varier la résolution spatiale et estime l'ordre de convergence empirique.

    L'ordre p est estimé à partir de la pente du graphe log-log :
        erreur L2 ~ C * dx^p
        p = log(err_1 / err_2) / log(dx_1 / dx_2)

    Attendu : p=1 pour Upwind et Lax-Friedrichs, p=2 pour Lax-Wendroff.
    Euler Centré est exclu (inconditionnellement instable).
    """
    if nx_values is None:
        nx_values = [50, 100, 200, 400, 800]

    # On exclut Euler Centré de l'étude de convergence
    stable_solvers = {k: v for k, v in SOLVERS.items() if k != "centered"}
    u0   = PROFILES["Gaussienne"]  # profil lisse pour une convergence propre
    rows = []

    for scheme_name, solver in stable_solvers.items():
        prev_l2, prev_dx = None, None  # valeurs de la résolution précédente

        for nx in nx_values:
            dx       = L / (nx - 1)
            dt       = cfl * dx / c
            nt       = max(1, int(np.ceil(T / dt)))
            x        = np.linspace(0, L, nx)
            _, u_num = solver(u0, L, T, c, nx, nt)
            u_exact  = exact_solution(u0, x, T, c)
            l2       = compute_errors(u_num, u_exact)["L2"]

            # Ordre local entre deux résolutions consécutives
            order = (
                round(np.log(prev_l2 / l2) / np.log(prev_dx / dx), 3)
                if (prev_l2 and l2 and prev_l2 > 0 and l2 > 0)
                else None
            )

            rows.append({
                "scheme": scheme_name,
                "nx":     nx,
                "dx":     round(dx, 6),
                "L2":     l2,
                "order":  order,
            })
            prev_l2, prev_dx = l2, dx

    return pd.DataFrame(rows)


# Carte de stabilité

def stability_map(
    L: float = 1.0,
    T: float = 0.3,
    c: float = 1.0,
    nx: int  = 100,
    cfl_values: list = None,
) -> pd.DataFrame:
    """
    Calcule l'amplitude maximale pour chaque (schéma, CFL) afin de révéler le seuil de stabilité.

    Un schéma est considéré instable si max_amplitude > STABILITY_THRESHOLD * amplitude_initiale.
    La colonne 'stable' du DataFrame reflète ce même seuil que run_benchmark.

    Colonnes : scheme, cfl, max_amplitude, stable
    """
    if cfl_values is None:
        # On teste au-delà de CFL=1 pour capturer l'instabilité
        cfl_values = np.round(np.arange(0.1, 1.51, 0.05), 2).tolist()

    u0  = PROFILES["Gaussienne"]
    dx  = L / (nx - 1)
    rows = []

    # init_amp calculé sur la même grille nx que les simulations
    init_amp = float(np.max(np.abs(u0(np.linspace(0, L, nx))))) + 1e-12

    for scheme_name, solver in SOLVERS.items():
        for cfl in cfl_values:
            dt = cfl * dx / c
            nt = max(1, int(np.ceil(T / dt)))
            try:
                _, u_num = solver(u0, L, T, c, nx, nt)
                max_amp  = float(min(np.max(np.abs(u_num)), 1e6))
            except Exception:
                max_amp = 1e6

            rows.append({
                "scheme":        scheme_name,
                "cfl":           cfl,
                "max_amplitude": round(max_amp, 4),
                # Même seuil que run_benchmark pour cohérence
                "stable":        max_amp < STABILITY_THRESHOLD * init_amp,
            })

    return pd.DataFrame(rows)
