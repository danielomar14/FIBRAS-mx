"""
Página 2 — Experimento 1: Comparativa de 11 estrategias de inversión.

Periodo: 2018-01-01 a 2025-12-31.
Capital inicial: $200,000 MXN.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.backtest.engine import run_backtest, _load_data, INITIAL_CAP
from src.backtest.strategies import STRATEGIES
from src.backtest.metrics import (
    summary, drawdown_series, annual_returns, cagr,
    annualized_volatility, max_drawdown, sharpe,
)
from src.data.benchmarks import (
    fetch_cetes, fetch_banxico_rate, cetes_daily_return,
)

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Estrategias FIBRAS", layout="wide")
st.title("Experimento 1 — Backtesting de 11 estrategias")
st.caption(
    "Capital inicial: $200,000 MXN · Rebalanceo trimestral · DRIP 100% · "
    "Comisión 0.25% + IVA · Slippage 0.10% · Periodo 2018–2025"
)

BACKTEST_START = "2018-01-01"
BACKTEST_END   = "2025-12-31"

# Colores por categoría
CAT_COLORS = {
    "Pasiva":      "#636EFA",
    "Tendencial":  "#EF553B",
    "Factor":      "#00CC96",
    "Fundamental": "#AB63FA",
    "Multi-factor":"#FFA15A",
    "Contraria":   "#19D3F3",
    "Macro":       "#FF6692",
}

# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Cargando datos de precios…")
def load_market_data():
    pw, div, met = _load_data()
    cetes   = fetch_cetes(BACKTEST_START, BACKTEST_END)
    banxico = fetch_banxico_rate(BACKTEST_START, BACKTEST_END)
    rf      = cetes_daily_return(cetes["cetes_pct"])
    return pw, div, met, rf, banxico


@st.cache_data(ttl=3600, show_spinner="Corriendo backtests…")
def run_all_backtests(_pw, _div, _met, _banxico):
    results = {}
    for code, s in STRATEGIES.items():
        df = run_backtest(
            s["fn"],
            start=BACKTEST_START,
            end=BACKTEST_END,
            prices_wide=_pw,
            dividends=_div,
            metrics=_met,
            banxico_rates=_banxico,
        )
        results[code] = df
    return results


pw, div, met, rf_daily, banxico = load_market_data()
all_results = run_all_backtests(pw, div, met, banxico)

# ── Sección 1: Descripción de estrategias ─────────────────────────────────────

with st.expander("¿Qué hace cada estrategia? (click para expandir)", expanded=False):
    st.markdown(
        "Las 11 estrategias cubren los principales enfoques cuantitativos de "
        "selección de activos. **E0 (Naive 1/N)** es la referencia interna: "
        "ninguna estrategia activa debería rendir consistentemente menos que "
        "asignar pesos iguales a todo el universo. "
        "**CETES** es la referencia de costo de oportunidad libre de riesgo."
    )

    # Agrupar por categoría
    cats_order = ["Pasiva","Tendencial","Factor","Fundamental","Multi-factor","Contraria","Macro"]
    by_cat: dict[str, list] = {c: [] for c in cats_order}
    for code, s in STRATEGIES.items():
        by_cat[s["categoria"]].append((code, s))

    for cat in cats_order:
        items = by_cat[cat]
        if not items:
            continue
        color = CAT_COLORS.get(cat, "#888")
        st.markdown(
            f"<span style='background:{color};color:white;"
            f"padding:2px 10px;border-radius:4px;font-size:0.85em'>"
            f"&nbsp;{cat}&nbsp;</span>",
            unsafe_allow_html=True,
        )
        for code, s in items:
            with st.container():
                col_badge, col_text = st.columns([1, 9])
                with col_badge:
                    st.markdown(
                        f"<div style='text-align:center;margin-top:8px;"
                        f"font-weight:bold;font-size:1.1em'>{code}</div>",
                        unsafe_allow_html=True,
                    )
                with col_text:
                    st.markdown(f"**{s['nombre']}**")
                    st.markdown(s["descripcion"])
                    st.caption(
                        f"Universo: {s['universo']} &nbsp;|&nbsp; "
                        f"Señal: {s['señal']}"
                    )
            st.divider()

# ── Sección 2: Tabla resumen ──────────────────────────────────────────────────

st.subheader("Resumen de métricas — todos los periodos (2018–2025)")

rows = []
for code, s in STRATEGIES.items():
    pv = all_results[code]["portfolio_value"]
    m  = summary(pv, rf_daily)
    rows.append({
        "Cód.":          code,
        "Estrategia":    s["nombre"],
        "Categoría":     s["categoria"],
        "CAGR %":        m["CAGR"],
        "Vol % (anual)": m["Volatilidad"],
        "Sharpe":        m["Sharpe"],
        "Calmar":        m["Calmar"],
        "MaxDD %":       m["MaxDD"],
        "Consist. %":    m["Consistencia"],
        "Valor final $": m["Valor final"],
    })

df_summary = pd.DataFrame(rows).set_index("Cód.")

def _c_cagr(v):
    if v >= 8:  return "background-color:#1a7a4a;color:white"
    if v >= 4:  return "background-color:#2ca05a;color:white"
    if v >= 0:  return "background-color:#d4edda"
    return "background-color:#f8d7da"

def _c_mdd(v):
    if v >= -20: return "background-color:#d4edda"
    if v >= -35: return "background-color:#fff3cd"
    return "background-color:#f8d7da"

def _c_sharpe(v):
    if v >= 0.3: return "background-color:#1a7a4a;color:white"
    if v >= 0.1: return "background-color:#d4edda"
    if v >= 0:   return ""
    return "background-color:#f8d7da"

styled = (
    df_summary.style
    .map(_c_cagr,   subset=["CAGR %"])
    .map(_c_mdd,    subset=["MaxDD %"])
    .map(_c_sharpe, subset=["Sharpe"])
    .format({
        "CAGR %":        "{:.2f}",
        "Vol % (anual)": "{:.2f}",
        "Sharpe":        "{:.3f}",
        "Calmar":        "{:.3f}",
        "MaxDD %":       "{:.2f}",
        "Consist. %":    "{:.1f}",
        "Valor final $": "{:,.0f}",
    })
)
st.dataframe(styled, use_container_width=True, height=450)

# ── Sección 3: Selector ────────────────────────────────────────────────────────

st.divider()
all_codes  = list(STRATEGIES.keys())
default_sel = ["E0", "E5", "E9", "E11", "E12"]
selected = st.multiselect(
    "Seleccionar estrategias a graficar",
    options=all_codes,
    default=default_sel,
    format_func=lambda c: f"{c} — {STRATEGIES[c]['nombre']} ({STRATEGIES[c]['categoria']})",
)
if not selected:
    st.info("Selecciona al menos una estrategia.")
    st.stop()

COLOR_PALETTE = px.colors.qualitative.Plotly

# CETES acumulado (referencia libre de riesgo)
cetes_full  = fetch_cetes(BACKTEST_START, BACKTEST_END)
rf_full     = cetes_daily_return(cetes_full["cetes_pct"])
trading     = all_results["E0"].index
rf_trading  = rf_full.reindex(trading).ffill().fillna(0)
cetes_val   = INITIAL_CAP * (1 + rf_trading).cumprod()
cetes_cagr_ = cagr(cetes_val) * 100

# ── Tabs de gráficas ──────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Curvas de valor",
    "📉 Drawdown",
    "📅 Retornos anuales",
    "⚖️ Riesgo vs retorno",
])

# ── Tab 1: Equity curves ──────────────────────────────────────────────────────

with tab1:
    st.markdown(
        "Evolución del valor del portafolio desde **$200,000 MXN**. "
        "La línea gris punteada representa invertir todo en CETES 28d "
        "(costo de oportunidad libre de riesgo)."
    )
    fig = go.Figure()

    for i, code in enumerate(selected):
        pv   = all_results[code]["portfolio_value"]
        cat  = STRATEGIES[code]["categoria"]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv,
            name=f"{code} — {STRATEGIES[code]['nombre']}",
            line=dict(color=color, width=1.8),
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f} MXN<extra>%{fullData.name}</extra>",
        ))

    fig.add_trace(go.Scatter(
        x=cetes_val.index, y=cetes_val,
        name=f"CETES 28d ({cetes_cagr_:.1f}% CAGR)",
        line=dict(color="gray", dash="dash", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f} MXN<extra>CETES</extra>",
    ))

    fig.update_layout(
        height=460,
        xaxis_title="", yaxis_title="Valor portafolio (MXN)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    cols_m = st.columns(len(selected))
    for i, code in enumerate(selected):
        pv = all_results[code]["portfolio_value"]
        m  = summary(pv, rf_daily)
        with cols_m[i]:
            st.metric(
                label=f"{code} · {STRATEGIES[code]['nombre']}",
                value=f"${m['Valor final']:,.0f}",
                delta=f"CAGR {m['CAGR']:.2f}% · Sharpe {m['Sharpe']:.2f}",
            )

# ── Tab 2: Drawdown ───────────────────────────────────────────────────────────

with tab2:
    st.markdown(
        "Caída relativa desde el máximo histórico de cada portafolio. "
        "Un drawdown de -35% significa que en ese punto el portafolio "
        "valía 35% menos que su pico anterior."
    )
    fig2 = go.Figure()
    for i, code in enumerate(selected):
        pv = all_results[code]["portfolio_value"]
        dd = drawdown_series(pv) * 100
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=f"{code} — {STRATEGIES[code]['nombre']}",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.07)") if color.startswith("rgb") else color,
            hovertemplate="%{x|%d %b %Y}<br>DD: %{y:.1f}%<extra>%{fullData.name}</extra>",
        ))
    fig2.add_hline(y=0, line_width=0.5, line_color="black")
    fig2.update_layout(
        height=420,
        xaxis_title="", yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    mdd_data = {
        code: f"{max_drawdown(all_results[code]['portfolio_value'])*100:.2f}%"
        for code in selected
    }
    st.dataframe(
        pd.DataFrame([mdd_data], index=["MaxDD"]),
        use_container_width=True,
    )

# ── Tab 3: Retornos anuales ───────────────────────────────────────────────────

with tab3:
    st.markdown(
        "Retorno anual por estrategia. "
        "Verde oscuro = muy positivo, rojo = pérdida."
    )
    ann_matrix = {}
    for code in selected:
        pv = all_results[code]["portfolio_value"]
        yr = annual_returns(pv) * 100
        ann_matrix[code] = yr

    df_ann = pd.DataFrame(ann_matrix)
    df_ann.index = df_ann.index.year

    fig3 = go.Figure(data=go.Heatmap(
        z=df_ann.values,
        x=[f"{c} — {STRATEGIES[c]['nombre']}" for c in df_ann.columns],
        y=df_ann.index.tolist(),
        colorscale=[[0,"#c0392b"],[0.5,"#f9e79f"],[1,"#1a7a4a"]],
        zmid=0,
        text=np.round(df_ann.values, 1),
        texttemplate="%{text:.1f}%",
        hovertemplate="Año %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Retorno %"),
    ))
    fig3.update_layout(
        height=380,
        xaxis_title="", yaxis_title="Año",
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig3, use_container_width=True)

    def _color_ret(v):
        if v > 15:  return "background-color:#1a7a4a;color:white"
        if v > 0:   return "background-color:#d4edda"
        if v > -10: return "background-color:#fff3cd"
        return "background-color:#f8d7da"

    st.dataframe(
        df_ann.style.map(_color_ret).format("{:.1f}%"),
        use_container_width=True,
    )

# ── Tab 4: Risk-return scatter ────────────────────────────────────────────────

with tab4:
    st.markdown(
        "Cada punto es una estrategia. El color indica el Sharpe ratio "
        "(verde = mejor). La línea punteada marca el CAGR de CETES: "
        "**cualquier estrategia por debajo de esa línea no compensó el riesgo tomado**."
    )
    rr_rows = []
    for code, s in STRATEGIES.items():
        pv = all_results[code]["portfolio_value"]
        rr_rows.append({
            "Código":     code,
            "Estrategia": s["nombre"],
            "Categoría":  s["categoria"],
            "CAGR %":     cagr(pv) * 100,
            "Vol %":      annualized_volatility(pv) * 100,
            "Sharpe":     sharpe(pv, rf_daily),
            "MaxDD %":    abs(max_drawdown(pv)) * 100,
            "Seleccionada": code in selected,
        })
    df_rr = pd.DataFrame(rr_rows)

    fig4 = go.Figure()
    for _, row in df_rr.iterrows():
        opacity = 1.0 if row["Seleccionada"] else 0.3
        size    = 16  if row["Seleccionada"] else 9
        fig4.add_trace(go.Scatter(
            x=[row["Vol %"]], y=[row["CAGR %"]],
            mode="markers+text",
            name=row["Código"],
            text=[row["Código"]],
            textposition="top center",
            marker=dict(
                size=size, opacity=opacity,
                color=row["Sharpe"],
                colorscale="RdYlGn",
                cmin=-0.1, cmax=0.5,
                showscale=(row["Código"] == df_rr.iloc[-1]["Código"]),
                colorbar=dict(title="Sharpe"),
            ),
            hovertemplate=(
                f"<b>{row['Código']} — {row['Estrategia']}</b><br>"
                f"Categoría: {row['Categoría']}<br>"
                f"CAGR: {row['CAGR %']:.2f}%<br>"
                f"Volatilidad: {row['Vol %']:.2f}%<br>"
                f"Sharpe: {row['Sharpe']:.3f}<br>"
                f"MaxDD: -{row['MaxDD %']:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig4.add_hline(
        y=cetes_cagr_,
        line_dash="dot", line_color="gray",
        annotation_text=f"CETES ({cetes_cagr_:.1f}%)",
        annotation_position="bottom right",
    )
    fig4.update_layout(
        height=480,
        xaxis_title="Volatilidad anualizada (%)",
        yaxis_title="CAGR (%)",
        xaxis_ticksuffix="%", yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(
        df_rr[["Código","Estrategia","Categoría","CAGR %","Vol %","Sharpe","MaxDD %"]]
        .set_index("Código")
        .sort_values("Sharpe", ascending=False)
        .style.format({
            "CAGR %":  "{:.2f}",
            "Vol %":   "{:.2f}",
            "Sharpe":  "{:.3f}",
            "MaxDD %": "{:.2f}",
        }),
        use_container_width=True,
    )
