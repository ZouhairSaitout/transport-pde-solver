"""
Dashboard interactif

Lancer avec :
    streamlit run app.py

Onglets
-------
1. Explorateur CFL  - simulation en direct, déplacer le slider pour voir l'effet sur la solution
2. Benchmark        - tableau comparatif L1 / L2 / L∞ + graphique en barres
3. Convergence      - graphe log-log, estimation empirique de l'ordre numérique
4. Stabilité        - heatmap de l'amplitude maximale en fonction du CFL
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from solvers import SOLVERS, PROFILES, SCHEME_NAMES, SCHEME_COLORS, exact_solution
from analysis import compute_errors, run_benchmark, convergence_study, stability_map, STABILITY_THRESHOLD



# Configuration de la page

st.set_page_config(
    page_title="Transport PDE Solver",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EDP de transport - Analyse des schémas numériques")
st.caption(
    "Équation de transport 1D : **∂u/∂t + c · ∂u/∂x = 0**"
)


# Barre latérale - paramètres globaux

with st.sidebar:
    st.header("Paramètres globaux")
    L  = st.number_input("Longueur du domaine L", value=1.0, min_value=0.1, step=0.1)
    T  = st.number_input("Temps final T",          value=0.5, min_value=0.05, step=0.05)
    c  = st.number_input("Vitesse d'advection c",  value=1.0, min_value=0.1, step=0.1)
    nx = st.slider("Points spatiaux nx",            50, 400, 200, 50)

    st.divider()
    st.markdown("""
**Schémas**
- 🔵 Euler Upwind - ordre 1
- 🟢 Lax-Friedrichs - ordre 1
- 🟡 Lax-Wendroff - ordre 2
- 🔴 Euler Centré - instable

**Profils**
- Gaussienne · Sinus · Créneau · Triangle
""")


# Onglets

tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Explorateur CFL",
    "📊 Benchmark",
    "📈 Convergence",
    "🔬 Stabilité",
])



# Onglet 1 - Explorateur CFL

with tab1:
    st.subheader("Effet du nombre de Courant sur la solution")

    col1, col2, col3 = st.columns(3)
    with col1:
        scheme = st.selectbox(
            "Schéma",
            list(SCHEME_NAMES.keys()),
            format_func=lambda s: SCHEME_NAMES[s],
        )
    with col2:
        profile = st.selectbox("Profil initial", list(PROFILES.keys()))
    with col3:
        cfl_val = st.slider("CFL", 0.05, 1.3, 0.9, 0.05)

    # Calcul du pas de temps à partir du CFL choisi
    dx         = L / (nx - 1)
    dt         = cfl_val * dx / c
    nt         = max(1, int(np.ceil(T / dt)))
    cfl_actual = round(c * (T / nt) / dx, 4)   # CFL effectif après arrondi de nt
    x          = np.linspace(0, L, nx)
    u0         = PROFILES[profile]

    try:
        _, u_num = SOLVERS[scheme](u0, L, T, c, nx, nt)
        max_amp  = float(np.max(np.abs(u_num)))
        init_amp = float(np.max(np.abs(u0(x)))) + 1e-12
        # Instable si la solution a grandi de plus de STABILITY_THRESHOLD fois l'amplitude initiale
        stable   = max_amp < STABILITY_THRESHOLD * init_amp
    except Exception:
        u_num   = np.zeros(nx)
        stable  = False
        max_amp = float("inf")

    u_exact = exact_solution(u0, x, T, c)
    errors  = compute_errors(u_num, u_exact) if stable else {}

    # Ligne de métriques
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("CFL effectif", f"{cfl_actual:.3f}")
    m2.metric("L1",   f"{errors['L1']:.2e}"   if errors else "-")
    m3.metric("L2",   f"{errors['L2']:.2e}"   if errors else "-")
    m4.metric("L∞",   f"{errors['Linf']:.2e}" if errors else "-")
    m5.metric("Stabilité", "✅ Stable" if stable else "❌ Instable")

    if not stable:
        st.error(
            f"CFL = {cfl_actual} dépasse le seuil de stabilité. "
            f"Amplitude maximale = {max_amp:.1e}. Réduisez le CFL en dessous de 1."
        )

    # Graphique de la solution numérique vs exacte
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=u_exact, mode="lines", name="Solution exacte",
        line=dict(color="#000000", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=u_num, mode="lines",
        name=SCHEME_NAMES[scheme],
        line=dict(color=SCHEME_COLORS[scheme], width=2),
    ))
    fig.update_layout(dragmode="pan",
        xaxis_title="x",
        yaxis_title="u(x, T)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparaison de tous les schémas au même CFL
    with st.expander("Comparer tous les schémas à ce CFL"):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x, y=u_exact, mode="lines", name="Exacte",
            line=dict(color="#000000", width=2, dash="dash"),
        ))
        for s_name, s_fn in SOLVERS.items():
            try:
                _, u_s = s_fn(u0, L, T, c, nx, nt)
                # On n'affiche pas les schémas qui ont explosé
                if np.max(np.abs(u_s)) < STABILITY_THRESHOLD * init_amp:
                    fig2.add_trace(go.Scatter(
                        x=x, y=u_s, mode="lines",
                        name=SCHEME_NAMES[s_name],
                        line=dict(color=SCHEME_COLORS[s_name], width=2),
                    ))
            except Exception:
                pass
        fig2.update_layout(dragmode="pan",
            xaxis_title="x",
            yaxis_title="u(x, T)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=60),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)



# Onglet 2 - Benchmark

with tab2:
    st.subheader("Benchmark - L1 · L2 · L∞ · CPU")

    @st.cache_data
    def load_benchmark(L, T, c, nx):
        """Lance toutes les simulations et met le résultat en cache."""
        return run_benchmark(L=L, T=T, c=c, nx=nx)

    df = load_benchmark(L, T, c, nx)

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        profile_b = st.selectbox(
            "Profil", list(PROFILES.keys()), key="bench_profile"
        )
        metric_b = st.selectbox("Métrique", ["L1", "L2", "Linf"])
        cfl_options = sorted(df["cfl"].unique())
        cfl_default = min(cfl_options, key=lambda x: abs(x - 0.9))
        cfl_b = st.select_slider(
            "CFL", options=cfl_options, value=cfl_default
        )
        schemes_selected = st.multiselect(
            "Schémas à afficher",
            options=list(SCHEME_NAMES.keys()),
            default=["upwind", "lax_friedrichs", "lax_wendroff"],
            format_func=lambda s: SCHEME_NAMES[s],
        )

    if not schemes_selected:
        st.warning("Sélectionne au moins un schéma.")
        st.stop()

    # Tableau comparatif avec meilleure valeur surlignée en vert
    with col_b2:
        tol  = 0.05
        mask = (
            (df["profile"] == profile_b) &
            (df["cfl"].between(cfl_b - tol, cfl_b + tol)) &
            (df["scheme"].isin(schemes_selected)) &
            df["stable"]
        )
        sub = df[mask].copy()

        if not sub.empty:
            sub = sub.sort_values("cfl").groupby("scheme").first().reset_index()
            table = pd.DataFrame({
                "Schéma":   sub["scheme"].map(SCHEME_NAMES),
                "L1":       sub["L1"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "-"),
                "L2":       sub["L2"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "-"),
                "L∞":       sub["Linf"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "-"),
                "CPU (ms)": sub["cpu_ms"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "-"),
                "Stable":   sub["stable"].map(lambda v: "✅" if v else "❌"),
            }).set_index("Schéma")

            def highlight_best(col):
                """Surligne en vert la meilleure valeur (la plus basse) de chaque colonne."""
                styles = [""] * len(col)
                try:
                    vals = col.map(lambda x: float(x) if x not in ("-", "✅", "❌") else np.nan)
                    if vals.notna().any():
                        best = vals.idxmin()
                        styles[col.index.get_loc(best)] = (
                            "background-color: #d3f9d8; color: #1a7f3c; font-weight: 500"
                        )
                except Exception:
                    pass
                return styles

            styled = table.style
            for col in ["L1", "L2", "L∞", "CPU (ms)"]:
                styled = styled.apply(highlight_best, subset=[col])
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("Aucune donnée stable pour cette combinaison.")

    # Graphique en barres - erreurs par CFL
    df_plot = df[
        (df["profile"] == profile_b) &
        (df["scheme"].isin(schemes_selected)) &
        df["stable"]
    ].copy()
    df_plot["Schéma"] = df_plot["scheme"].map(SCHEME_NAMES)

    fig_bar = px.bar(
        df_plot,
        x="cfl", y=metric_b, color="Schéma", barmode="group",
        color_discrete_map={v: SCHEME_COLORS[k] for k, v in SCHEME_NAMES.items()},
        labels={metric_b: f"Erreur {metric_b}", "cfl": "CFL"},
    )
    fig_bar.update_layout(dragmode="pan", margin=dict(t=20), legend_title="Schéma")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Boîte à moustaches - temps de calcul CPU par schéma
    st.markdown("##### Temps de calcul par schéma")
    df_cpu = df[(df["profile"] == profile_b) & df["stable"]].copy()
    df_cpu["Schéma"] = df_cpu["scheme"].map(SCHEME_NAMES)
    fig_cpu = px.box(
        df_cpu, x="Schéma", y="cpu_ms", color="Schéma",
        color_discrete_map={v: SCHEME_COLORS[k] for k, v in SCHEME_NAMES.items()},
        labels={"cpu_ms": "CPU (ms)"},
        points="all",
    )
    fig_cpu.update_layout(dragmode="pan", showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig_cpu, use_container_width=True)



# Onglet 3 - Convergence

with tab3:
    st.subheader("Étude de convergence - ordre numérique empirique")
    st.markdown(
        "La pente du graphe log-log donne l'ordre de convergence. "
        "Attendu : **ordre 1** pour Upwind et Lax-Friedrichs, **ordre 2** pour Lax-Wendroff."
    )

    @st.cache_data
    def load_convergence(L, T, c):
        """Lance l'étude de convergence et met le résultat en cache."""
        return convergence_study(L=L, T=T, c=c)

    df_conv = load_convergence(L, T, c)

    fig_conv = go.Figure()
    for scheme in df_conv["scheme"].unique():
        sub = df_conv[df_conv["scheme"] == scheme].dropna(subset=["L2"])
        fig_conv.add_trace(go.Scatter(
            x=sub["dx"], y=sub["L2"],
            mode="lines+markers",
            name=SCHEME_NAMES.get(scheme, scheme),
            line=dict(color=SCHEME_COLORS.get(scheme, "#888"), width=2),
            marker=dict(size=7),
        ))

    # Droites de référence pour les ordres 1 et 2
    dx_ref = np.array([8e-4, 2.5e-2])
    fig_conv.add_trace(go.Scatter(
        x=dx_ref, y=dx_ref * 4,
        mode="lines", name="Ordre 1 (référence)",
        line=dict(color="#AAAAAA", dash="dot", width=1.5),
    ))
    fig_conv.add_trace(go.Scatter(
        x=dx_ref, y=dx_ref ** 2 * 150,
        mode="lines", name="Ordre 2 (référence)",
        line=dict(color="#555555", dash="dot", width=1.5),
    ))
    fig_conv.update_layout(dragmode="pan",
        xaxis_title="dx (pas spatial)",
        yaxis_title="Erreur L2",
        xaxis_type="log",
        yaxis_type="log",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20, b=60),
        hovermode="x unified",
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    with st.expander("Tableau des ordres de convergence"):
        df_display = df_conv.copy()
        df_display["scheme"] = df_display["scheme"].map(SCHEME_NAMES)
        df_display["L2"]     = df_display["L2"].map(lambda v: f"{v:.2e}")
        df_display["order"]  = df_display["order"].map(
            lambda v: str(round(v)) if pd.notna(v) else "-"
        )
        df_display = df_display.rename(columns={
            "scheme": "Schéma", "nx": "nx", "dx": "dx",
            "L2": "L2", "order": "Ordre estimé",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)



# Onglet 4 - Stabilité

with tab4:
    st.subheader("Carte de stabilité - amplitude maximale vs CFL")
    st.markdown(
        f"Un schéma est considéré **instable** si son amplitude dépasse "
        f"**{STABILITY_THRESHOLD}× l'amplitude initiale**. "
        "Une explosion numérique se traduit par une couleur rouge sur la heatmap."
    )

    @st.cache_data
    def load_stability(L, T, c, nx):
        """Calcule la carte de stabilité et met le résultat en cache."""
        return stability_map(L=L, T=T, c=c, nx=nx)

    df_stab = load_stability(L, T, c, nx)
    df_stab["log_amp"] = np.log10(np.clip(df_stab["max_amplitude"], 1e-3, 1e6))
    df_stab["Schéma"]  = df_stab["scheme"].map(SCHEME_NAMES)

    # Heatmap - un carré par (schéma, CFL)
    pivot = df_stab.pivot(index="Schéma", columns="cfl", values="log_amp")

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=0, zmax=3,
        colorbar=dict(
            title="log₁₀(amplitude)",
            tickvals=[0, 1, 2, 3],
            ticktext=["1", "10", "100", "1000"],
        ),
        hovertemplate="CFL: %{x}<br>Schéma: %{y}<br>Amplitude: 10^%{z:.2f}<extra></extra>",
    ))
    # Ligne verticale marquant le seuil CFL = 1
    fig_heat.add_vline(
        x=1.0, line_dash="dash", line_color="white", line_width=2,
        annotation_text="CFL = 1", annotation_position="top right",
    )
    fig_heat.update_layout(dragmode="pan",
        xaxis_title="Nombre de Courant (CFL)",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "Le schéma **Euler Centré** est inconditionnellement instable : "
        "son amplitude explose pour tout CFL > 0, confirmé par l'analyse de Von Neumann (|g(ξ)| > 1)."
    )

    # Courbes d'amplitude par schéma
    fig_line = px.line(
        df_stab,
        x="cfl", y="log_amp", color="Schéma",
        color_discrete_map={v: SCHEME_COLORS[k] for k, v in SCHEME_NAMES.items()},
        labels={"log_amp": "log₁₀(amplitude max)", "cfl": "CFL"},
    )
    fig_line.add_hline(
        y=0, line_dash="dash", line_color="#333",
        annotation_text="Seuil stable (amplitude = 1)",
    )
    fig_line.add_vline(x=1.0, line_dash="dot", line_color="#888")
    fig_line.update_layout(dragmode="pan", margin=dict(t=20))
    st.plotly_chart(fig_line, use_container_width=True)
