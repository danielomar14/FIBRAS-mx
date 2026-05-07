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
from src.backtest.strategies import STRATEGIES, SECTOR_MAP
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

# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Cargando datos de precios…")
def load_market_data():
    pw, div, met = _load_data()
    cetes = fetch_cetes(BACKTEST_START, BACKTEST_END)
    banxico = fetch_banxico_rate(BACKTEST_START, BACKTEST_END)
    rf = cetes_daily_return(cetes["cetes_pct"])
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

# ── Tabla resumen ─────────────────────────────────────────────────────────────

st.subheader("Resumen de métricas")

rows = []
for code, s in STRATEGIES.items():
    pv = all_results[code]["portfolio_value"]
    m = summary(pv, rf_daily)
    rows.append({
        "Cód.":        code,
        "Estrategia":  s["nombre"],
        "CAGR %":      m["CAGR"],
        "Vol % (anual)": m["Volatilidad"],
        "Sharpe":      m["Sharpe"],
        "Calmar":      m["Calmar"],
        "MaxDD %":     m["MaxDD"],
        "Consistent. %": m["Consistencia"],
        "Valor final $": m["Valor final"],
    })

df_summary = pd.DataFrame(rows).set_index("Cód.")

def color_cagr(v):
    if v >= 8:   return "background-color: #1a7a4a; color: white"
    if v >= 4:   return "background-color: #2ca05a; color: white"
    if v >= 0:   return "background-color: #d4edda"
    return "background-color: #f8d7da"

def color_mdd(v):
    if v >= -20: return "background-color: #d4edda"
    if v >= -35: return "background-color: #fff3cd"
    return "background-color: #f8d7da"

def color_sharpe(v):
    if v >= 0.3: return "background-color: #1a7a4a; color: white"
    if v >= 0.1: return "background-color: #d4edda"
    if v >= 0:   return ""
    return "background-color: #f8d7da"

styled = (
    df_summary.style
    .map(color_cagr, subset=["CAGR %"])
    .map(color_mdd,  subset=["MaxDD %"])
    .map(color_sharpe, subset=["Sharpe"])
    .format({
        "CAGR %":       "{:.2f}",
        "Vol % (anual)":"{:.2f}",
        "Sharpe":       "{:.3f}",
        "Calmar":       "{:.3f}",
        "MaxDD %":      "{:.2f}",
        "Consistent. %":"{:.1f}",
        "Valor final $": "{:,.0f}",
    })
)
st.dataframe(styled, use_container_width=True, height=450)

# ── Selector de estrategias para gráficas ─────────────────────────────────────

st.divider()
all_codes = list(STRATEGIES.keys())
default_sel = ["E0", "E2", "E5", "E7", "E9"]
selected = st.multiselect(
    "Seleccionar estrategias a graficar",
    options=all_codes,
    default=default_sel,
    format_func=lambda c: f"{c} — {STRATEGIES[c]['nombre']}",
)
if not selected:
    st.info("Selecciona al menos una estrategia.")
    st.stop()

COLOR_PALETTE = px.colors.qualitative.Plotly

# ── Tab 1: Equity curves ─────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Curvas de valor",
    "📉 Drawdown",
    "📅 Retornos anuales",
    "⚖️ Risk-return",
])

with tab1:
    st.markdown("Evolución del valor del portafolio desde $200,000 MXN.")
    fig = go.Figure()
    for i, code in enumerate(selected):
        pv = all_results[code]["portfolio_value"]
        fig.add_trace(go.Scatter(
            x=pv.index, y=pv,
            name=f"{code} — {STRATEGIES[code]['nombre']}",
            line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=1.8),
            hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f} MXN<extra>%{fullData.name}</extra>",
        ))

    # Línea CETES acumulado como referencia
    cetes_full = fetch_cetes(BACKTEST_START, BACKTEST_END)
    rf = cetes_daily_return(cetes_full["cetes_pct"])
    trading = all_results["E0"].index
    rf_trading = rf.reindex(trading).ffill().fillna(0)
    cetes_val = INITIAL_CAP * (1 + rf_trading).cumprod()
    fig.add_trace(go.Scatter(
        x=cetes_val.index, y=cetes_val,
        name="CETES 28d (referencia)",
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

    # Métricas en columnas
    cols_m = st.columns(len(selected))
    for i, code in enumerate(selected):
        pv = all_results[code]["portfolio_value"]
        m = summary(pv, rf_daily)
        with cols_m[i]:
            st.metric(
                label=code,
                value=f"${m['Valor final']:,.0f}",
                delta=f"CAGR {m['CAGR']:.2f}%",
            )

# ── Tab 2: Drawdown ───────────────────────────────────────────────────────────

with tab2:
    st.markdown("Caída relativa desde el máximo histórico de cada portafolio.")
    fig2 = go.Figure()
    for i, code in enumerate(selected):
        pv = all_results[code]["portfolio_value"]
        dd = drawdown_series(pv) * 100
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=f"{code} — {STRATEGIES[code]['nombre']}",
            line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=1.5),
            fill="tozeroy",
            fillcolor=COLOR_PALETTE[i % len(COLOR_PALETTE)].replace("rgb", "rgba").replace(")", ",0.08)"),
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

    # Tabla de MaxDD por estrategia seleccionada
    mdd_data = {
        code: f"{max_drawdown(all_results[code]['portfolio_value'])*100:.2f}%"
        for code in selected
    }
    st.dataframe(pd.DataFrame([mdd_data], index=["MaxDD"]), use_container_width=True)

# ── Tab 3: Retornos anuales ───────────────────────────────────────────────────

with tab3:
    st.markdown("Retorno anual por estrategia (heatmap). Verde = positivo, rojo = negativo.")

    ann_matrix = {}
    for code in selected:
        pv = all_results[code]["portfolio_value"]
        yr = annual_returns(pv) * 100
        ann_matrix[f"{code}"] = yr

    df_ann = pd.DataFrame(ann_matrix)
    df_ann.index = df_ann.index.year

    fig3 = go.Figure(data=go.Heatmap(
        z=df_ann.values,
        x=df_ann.columns.tolist(),
        y=df_ann.index.tolist(),
        colorscale=[[0, "#c0392b"], [0.5, "#f9e79f"], [1, "#1a7a4a"]],
        zmid=0,
        text=np.round(df_ann.values, 1),
        texttemplate="%{text:.1f}%",
        hovertemplate="Año %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Retorno %"),
    ))
    fig3.update_layout(height=380, xaxis_title="", yaxis_title="Año")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Tabla detallada**")
    fmt = {c: "{:.1f}%" for c in df_ann.columns}

    def _color_ret(v):
        if v > 15:  return "background-color: #1a7a4a; color: white"
        if v > 0:   return "background-color: #d4edda"
        if v > -10: return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    st.dataframe(
        df_ann.style.map(_color_ret).format("{:.1f}%"),
        use_container_width=True,
    )

# ── Tab 4: Risk-return scatter ────────────────────────────────────────────────

with tab4:
    st.markdown(
        "Mapa riesgo-retorno. Cada punto es una estrategia. "
        "Ideal: esquina superior-izquierda (alto CAGR, baja volatilidad)."
    )

    rr_rows = []
    for code, s in STRATEGIES.items():
        pv = all_results[code]["portfolio_value"]
        rr_rows.append({
            "Código": code,
            "Estrategia": s["nombre"],
            "CAGR %": cagr(pv) * 100,
            "Volatilidad %": annualized_volatility(pv) * 100,
            "Sharpe": sharpe(pv, rf_daily),
            "MaxDD %": abs(max_drawdown(pv)) * 100,
            "Seleccionada": code in selected,
        })
    df_rr = pd.DataFrame(rr_rows)

    fig4 = go.Figure()
    for _, row in df_rr.iterrows():
        opacity = 1.0 if row["Seleccionada"] else 0.35
        size    = 18  if row["Seleccionada"] else 10
        fig4.add_trace(go.Scatter(
            x=[row["Volatilidad %"]], y=[row["CAGR %"]],
            mode="markers+text",
            name=row["Código"],
            text=[row["Código"]],
            textposition="top center",
            marker=dict(
                size=size,
                opacity=opacity,
                color=row["Sharpe"],
                colorscale="RdYlGn",
                cmin=-0.1, cmax=0.5,
                showscale=(row["Código"] == df_rr.iloc[-1]["Código"]),
                colorbar=dict(title="Sharpe"),
            ),
            hovertemplate=(
                f"<b>{row['Código']} — {row['Estrategia']}</b><br>"
                f"CAGR: {row['CAGR %']:.2f}%<br>"
                f"Vol: {row['Volatilidad %']:.2f}%<br>"
                f"Sharpe: {row['Sharpe']:.3f}<br>"
                f"MaxDD: -{row['MaxDD %']:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Línea de frontera eficiente indicativa (CETES)
    cetes_cagr = cagr(cetes_val) * 100
    fig4.add_hline(
        y=cetes_cagr,
        line_dash="dot", line_color="gray",
        annotation_text=f"CETES ({cetes_cagr:.1f}%)",
        annotation_position="bottom right",
    )

    fig4.update_layout(
        height=460,
        xaxis_title="Volatilidad anualizada (%)",
        yaxis_title="CAGR (%)",
        xaxis_ticksuffix="%", yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(
        df_rr[["Código","Estrategia","CAGR %","Volatilidad %","Sharpe","MaxDD %"]]
        .set_index("Código")
        .style.format({"CAGR %":"{:.2f}","Volatilidad %":"{:.2f}","Sharpe":"{:.3f}","MaxDD %":"{:.2f}"}),
        use_container_width=True,
    )
