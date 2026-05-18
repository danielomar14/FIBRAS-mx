"""
Página: Algoritmo — Experimento 2

Sección A: ML Puro (9 modelos sobre feature matrix completa)
Sección B: Algoritmo Genético (búsqueda combinatoria, sin ML)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

st.set_page_config(page_title="Algoritmo — FIBRAs MX", layout="wide")
st.title("Experimento 2: Selección Inteligente de FIBRAs")
st.caption(
    "**2A — ML Puro**: 9 modelos aprenden qué señales predicen rendimiento. "
    "**2B — Algoritmo Genético**: búsqueda combinatoria sin ML en 40×10 = 400 features."
)

# ── Datos compartidos ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_prices_wide():
    prices_long = pd.read_parquet(ROOT / "data/processed/precios_diarios.parquet")
    pw = prices_long.pivot_table(index="date", columns="ticker", values="close")
    pw.index = pd.to_datetime(pw.index)
    return pw.ffill(limit=5)


# ── Layout: dos secciones ─────────────────────────────────────────────────────

sec_ml, sec_ga = st.tabs(["2A — ML Puro", "2B — Algoritmo Genético"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN A: ML PURO
# ═══════════════════════════════════════════════════════════════════════════════

with sec_ml:
    st.header("2A — ML Puro")
    st.markdown(
        "Entrena 9 modelos de ML sobre la **feature matrix completa** (hasta 400 variables). "
        "Walk-forward CV en train (2017-2023); evaluación en test (2024-2025)."
    )

    with st.expander("Configuración", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            from src.ml.models import ALL_MODELS
            selected_models = st.multiselect(
                "Modelos a entrenar", ALL_MODELS, default=ALL_MODELS,
                key="ml_models"
            )
        with col2:
            ml_topk = st.slider("Top-k FIBRAs", 3, 7, 5, key="ml_topk")
        with col3:
            ml_cache = st.checkbox("Usar cache (si existe)", value=True, key="ml_cache")

    run_ml = st.button("Entrenar todos los modelos", type="primary", key="run_ml")

    if run_ml or ("ml_results" in st.session_state):
        if run_ml:
            with st.spinner("Construyendo feature matrix y entrenando modelos..."):
                from src.ml.runner import run_all_models
                results = run_all_models(
                    model_names=selected_models,
                    top_k=ml_topk,
                    cache=ml_cache,
                )
            st.session_state["ml_results"] = results

        results = st.session_state.get("ml_results", {})
        if not results:
            st.info("Presiona 'Entrenar todos los modelos' para comenzar.")
        else:
            tab_table, tab_equity, tab_fi = st.tabs(
                ["Tab 1 — Comparativa", "Tab 2 — Equity curve", "Tab 3 — Feature importance"]
            )

            # ── Tab 1: Tabla comparativa ──────────────────────────────────
            with tab_table:
                rows = []
                for name, r in results.items():
                    rows.append({
                        "Modelo":       name,
                        "Sharpe Train": r.get("sharpe_train", np.nan),
                        "Sharpe Test":  r.get("sharpe_test",  np.nan),
                        "CAGR Test":    r.get("cagr_test",    np.nan),
                        "MaxDD Test":   r.get("max_dd_test",  np.nan),
                    })
                df_table = pd.DataFrame(rows).set_index("Modelo")

                def _color_sharpe(val):
                    if pd.isna(val): return ""
                    if val > 1:   return "background-color: #00cc66; color: black"
                    if val > 0.5: return "background-color: #99ffcc; color: black"
                    if val < 0:   return "background-color: #ff6666; color: white"
                    return ""

                st.dataframe(
                    df_table.style
                        .format({"Sharpe Train": "{:.2f}", "Sharpe Test": "{:.2f}",
                                 "CAGR Test": "{:.1%}", "MaxDD Test": "{:.1%}"})
                        .map(_color_sharpe, subset=["Sharpe Train", "Sharpe Test"]),
                    use_container_width=True,
                )

                best_name = max(results, key=lambda k: results[k].get("sharpe_test", -np.inf))
                st.success(f"Mejor modelo por Sharpe_test: **{best_name}**  "
                           f"(Sharpe={results[best_name].get('sharpe_test', np.nan):.2f})")

            # ── Tab 2: Equity curve ───────────────────────────────────────
            with tab_equity:
                fig = go.Figure()
                for name, r in results.items():
                    eq = r.get("equity_test")
                    if eq is not None and len(eq) > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(eq))), y=eq.values,
                            name=name, mode="lines",
                        ))
                fig.update_layout(
                    title="Equity curve — Test (2024-2025)",
                    xaxis_title="Trimestre", yaxis_title="Valor (base 1)",
                    height=450, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Tab 3: Feature importance ─────────────────────────────────
            with tab_fi:
                from src.features.registry import FEATURE_REGISTRY
                fi_models = [n for n in results if results[n].get("feature_importance")]
                if not fi_models:
                    st.info("Ningún modelo con feature importance disponible aún.")
                else:
                    sel_model = st.selectbox("Modelo", fi_models, key="fi_model")
                    fi = results[sel_model]["feature_importance"]
                    if fi:
                        fi_series = pd.Series(fi).sort_values(ascending=False).head(30)
                        labels = []
                        for fid in fi_series.index:
                            fd = FEATURE_REGISTRY.get(int(fid))
                            labels.append(f"{fd.var_name}[{fd.param_name}]" if fd else str(fid))
                        fig_fi = go.Figure(go.Bar(
                            x=fi_series.values[::-1],
                            y=labels[::-1],
                            orientation="h",
                        ))
                        fig_fi.update_layout(
                            title=f"Top 30 features — {sel_model}",
                            height=600,
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Presiona **Entrenar todos los modelos** para comenzar el experimento 2A.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN B: ALGORITMO GENÉTICO
# ═══════════════════════════════════════════════════════════════════════════════

with sec_ga:
    st.header("2B — Algoritmo Genético")
    st.markdown(
        "Busca en el espacio de **40 variables × 10 parametrizaciones = 400 genes** la combinación "
        "de 1 a 5 genes que maximiza el Sharpe del portfolio en train (2017-2023). "
        "**Sin ML** — el cromosoma define directamente la regla de scoring."
    )

    with st.expander("Configuración", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ga_gens = st.slider("Generaciones", 5, 100, 30, key="ga_gens")
        with col2:
            ga_pop  = st.slider("Población", 5, 50, 15, key="ga_pop")
        with col3:
            ga_seed = st.number_input("Semilla aleatoria", 0, 9999, 42, step=1, key="ga_seed")
        with col4:
            ga_cache = st.checkbox("Usar cache", value=True, key="ga_cache")

    run_ga_btn = st.button("Correr Algoritmo Genético", type="primary", key="run_ga")

    # Live progress
    ga_progress = st.empty()
    ga_status   = st.empty()

    # Warn if params changed since last run
    stored_params = st.session_state.get("ga_run_params", {})
    if stored_params and not run_ga_btn:
        if (stored_params.get("gens") != ga_gens
                or stored_params.get("pop") != ga_pop
                or stored_params.get("seed") != int(ga_seed)):
            st.warning(
                f"Parámetros cambiados desde el último run "
                f"(gen={stored_params['gens']}, pop={stored_params['pop']}, "
                f"seed={stored_params['seed']}). "
                "Presiona **Correr** para actualizar."
            )

    if run_ga_btn or ("ga_result" in st.session_state):
        if run_ga_btn:
            from src.genetic.ga import run_ga

            progress_bar = ga_progress.progress(0)
            status_text  = ga_status.empty()

            def _cb(gen, best_fit, mean_fit, best_ind):
                pct = int(gen / ga_gens * 100)
                progress_bar.progress(pct)
                status_text.text(
                    f"Gen {gen}/{ga_gens}  |  Best Sharpe: {best_fit:.3f}  |  "
                    f"Mean: {mean_fit:.3f}  |  {best_ind.summary()}"
                )

            with st.spinner("Ejecutando GA..."):
                result = run_ga(
                    n_generations=ga_gens,
                    population_size=ga_pop,
                    seed=int(ga_seed),
                    cache=False,          # siempre corre fresco al presionar el botón
                    progress_callback=_cb,
                )
            progress_bar.empty()
            status_text.empty()
            ga_progress.empty()
            ga_status.empty()
            st.session_state["ga_result"] = result
            st.session_state["ga_run_params"] = {
                "gens": ga_gens, "pop": ga_pop, "seed": int(ga_seed),
            }

        result = st.session_state.get("ga_result")
        if result is None:
            st.info("Presiona 'Correr Algoritmo Genético' para comenzar.")
        else:
            tab_anim, tab_conv, tab_chrom, tab_eq, tab_freq = st.tabs([
                "Tab 0 — Evolución animada",
                "Tab 1 — Convergencia",
                "Tab 2 — Cromosoma ganador",
                "Tab 3 — Equity curve",
                "Tab 4 — Frecuencia variables",
            ])

            # ── Tab 0: Evolución de población (slider interactivo) ────────
            with tab_anim:
                import math as _math
                history_populations = getattr(result, "history_populations", [])
                if not history_populations:
                    st.warning(
                        "No hay datos de evolución por generación en este resultado. "
                        "Reinicia Streamlit (`Ctrl+C` → `streamlit run app/main.py`) "
                        "y corre el GA de nuevo para ver esta gráfica."
                    )
                else:
                    n_gens = len(history_populations)

                    all_valid = [
                        f for pop in history_populations for f in pop
                        if not (_math.isinf(f) or _math.isnan(f))
                    ]
                    y_min = (min(all_valid) - 0.15) if all_valid else -1.0
                    y_max = (max(all_valid) + 0.15) if all_valid else  1.0

                    gen_sel = st.slider(
                        "Generación", min_value=1, max_value=n_gens, value=n_gens,
                        key="gen_sel_anim",
                        help="Arrastra para ver cómo evoluciona la población",
                    )

                    accum_gen:   list[int]   = []
                    accum_fit:   list[float] = []
                    max_by_gen:  list[float] = []
                    mean_by_gen: list[float] = []
                    gen_xs:      list[int]   = []

                    for g in range(gen_sel):
                        pop = history_populations[g]
                        valid = [f for f in pop
                                 if not (_math.isinf(f) or _math.isnan(f))]
                        accum_gen.extend([g + 1] * len(valid))
                        accum_fit.extend(valid)
                        max_by_gen.append(max(valid)             if valid else 0.0)
                        mean_by_gen.append(float(np.mean(valid)) if valid else 0.0)
                        gen_xs.append(g + 1)

                    fig_anim = go.Figure()
                    if accum_fit:
                        fig_anim.add_trace(go.Scatter(
                            x=accum_gen,
                            y=accum_fit,
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=accum_fit,
                                colorscale="RdYlGn",
                                cmin=y_min, cmax=y_max,
                                showscale=True,
                                colorbar=dict(title="Fitness", len=0.75, thickness=14),
                                opacity=0.85,
                                line=dict(width=0.5, color="white"),
                            ),
                            name="Individuos",
                            hovertemplate="Gen %{x}<br>Fitness: %{y:.3f}<extra></extra>",
                        ))
                    fig_anim.add_trace(go.Scatter(
                        x=gen_xs, y=max_by_gen,
                        mode="lines+markers",
                        line=dict(dash="dot", color="#00CC96", width=2.5),
                        marker=dict(size=7, symbol="diamond"),
                        name="Máximo",
                    ))
                    fig_anim.add_trace(go.Scatter(
                        x=gen_xs, y=mean_by_gen,
                        mode="lines+markers",
                        line=dict(dash="dot", color="#EF553B", width=2.5),
                        marker=dict(size=7, symbol="circle"),
                        name="Promedio",
                    ))
                    fig_anim.update_layout(
                        title=f"Evolución GA — Generaciones 1 a {gen_sel} de {n_gens}",
                        xaxis=dict(
                            title="Generación",
                            range=[0.5, n_gens + 0.5],
                            dtick=1, tickmode="linear",
                        ),
                        yaxis=dict(
                            title="Fitness (Sharpe ajustado)",
                            range=[y_min, y_max],
                        ),
                        height=480,
                        hovermode="closest",
                        margin=dict(t=60, b=30, l=60, r=80),
                        legend=dict(
                            orientation="h", y=1.07, x=0.5, xanchor="center",
                        ),
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)
                    st.caption(
                        "Arrastra el slider para ver la evolución generación a generación. "
                        "Verde = fitness alto, rojo = bajo. "
                        f"**{n_gens}** generaciones en total."
                    )

            # ── Tab 1: Convergencia ───────────────────────────────────────
            with tab_conv:
                if stored_params:
                    st.caption(
                        f"Resultado del run: **{stored_params.get('gens','?')} gen** · "
                        f"**{stored_params.get('pop','?')} individuos** · "
                        f"semilla {stored_params.get('seed','?')}. "
                        "Cambia los parámetros y presiona **Correr** para actualizar."
                    )
                fig_conv = go.Figure()
                gens = list(range(1, len(result.history_best) + 1))
                fig_conv.add_trace(go.Scatter(
                    x=gens, y=result.history_best,
                    name="Mejor fitness", mode="lines+markers",
                    line=dict(color="#00CC96", width=2),
                ))
                fig_conv.add_trace(go.Scatter(
                    x=gens, y=result.history_mean,
                    name="Fitness medio", mode="lines",
                    line=dict(color="#636EFA", dash="dash"),
                ))
                fig_conv.update_layout(
                    title="Convergencia del GA — Sharpe ajustado",
                    xaxis_title="Generación", yaxis_title="Fitness (Sharpe - parsimony)",
                    height=400, hovermode="x unified",
                )
                st.plotly_chart(fig_conv, use_container_width=True)
                st.metric("Mejor Sharpe train (ajustado)", f"{max(result.history_best):.3f}")

            # ── Tab 2: Cromosoma ganador ───────────────────────────────────
            with tab_chrom:
                best = result.best_individual
                st.subheader("Cromosoma ganador")
                st.metric("Fitness (Sharpe train ajustado)", f"{best.fitness:.3f}")
                st.metric("Top-k FIBRAs", best.top_k)
                st.metric("N genes", len(best.genes))

                from src.features.registry import FEATURE_REGISTRY
                from src.genetic.chromosome import GENE_ID_MAP
                gene_rows = []
                for g in best.genes:
                    fid = GENE_ID_MAP[g]
                    fd  = FEATURE_REGISTRY.get(fid)
                    gene_rows.append({
                        "Bloque":         fd.block if fd else "?",
                        "Variable":       fd.var_name if fd else f"var_{g.var_id}",
                        "Parametrización": fd.param_name if fd else f"param_{g.param_idx}",
                        "Gene ID (fid)":  fid,
                    })
                st.table(pd.DataFrame(gene_rows))

                st.subheader("Métricas por período")
                cols = st.columns(3)
                for col, (label, met) in zip(cols, [
                    ("Train (2017-2023)", result.train_metrics),
                    ("Test (2024-2025)",  result.test_metrics),
                    ("Validation (2026)", result.val_metrics),
                ]):
                    with col:
                        n_per = met.get("n_periods", "?")
                        st.markdown(f"**{label}** — {n_per} trimestres")
                        if isinstance(n_per, int) and n_per < 4:
                            st.warning(f"Solo {n_per} período(s) — métricas poco confiables.")
                        sharpe = met.get("sharpe", np.nan)
                        st.metric("Sharpe",
                                  f"{sharpe:.2f}" if (sharpe is not None and not np.isnan(sharpe)) else "N/A")
                        cagr = met.get("cagr", np.nan)
                        st.metric("CAGR",   f"{cagr:.1%}"  if (cagr is not None and not np.isnan(cagr))   else "N/A")
                        mdd = met.get("max_dd", np.nan)
                        st.metric("MaxDD",  f"{mdd:.1%}"   if (mdd is not None  and not np.isnan(mdd))    else "N/A")

            # ── Tab 3: Equity curve ────────────────────────────────────────
            with tab_eq:
                fig_eq = go.Figure()
                colors = {"Train": "#00CC96", "Test": "#636EFA", "Validation": "#EF553B"}
                for label, met in [
                    ("Train",      result.train_metrics),
                    ("Test",       result.test_metrics),
                    ("Validation", result.val_metrics),
                ]:
                    eq = met.get("equity")
                    if eq is not None and len(eq) > 0:
                        fig_eq.add_trace(go.Scatter(
                            x=list(range(len(eq))), y=eq.values,
                            name=label, mode="lines+markers",
                            line=dict(color=colors[label], width=2),
                        ))
                fig_eq.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Base")
                fig_eq.update_layout(
                    title="Equity curve — Cromosoma ganador",
                    xaxis_title="Trimestre", yaxis_title="Valor (base 1)",
                    height=450, hovermode="x unified",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            # ── Tab 4: Frecuencia variables ────────────────────────────────
            with tab_freq:
                var_freq = result.var_frequency
                if not var_freq:
                    st.info("No hay datos de frecuencia.")
                else:
                    from src.features.registry import FEATURE_REGISTRY

                    freq_rows = []
                    for var_id, count in sorted(var_freq.items(), key=lambda x: -x[1]):
                        fid = var_id * 10
                        fd  = FEATURE_REGISTRY.get(fid)
                        freq_rows.append({
                            "Variable":  fd.var_name if fd else f"var_{var_id}",
                            "Bloque":    fd.block if fd else "?",
                            "Frecuencia (top-10 individuos)": count,
                        })
                    df_freq = pd.DataFrame(freq_rows)

                    fig_freq = go.Figure(go.Bar(
                        x=df_freq["Frecuencia (top-10 individuos)"][::-1],
                        y=df_freq["Variable"][::-1],
                        orientation="h",
                        marker_color="#EF553B",
                    ))
                    fig_freq.update_layout(
                        title="Frecuencia de variables en el top-10 final",
                        height=max(300, len(df_freq) * 28),
                    )
                    st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Presiona **Correr Algoritmo Genético** para comenzar el experimento 2B.")
