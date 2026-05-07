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

from src.backtest.engine import run_backtest, _load_data
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
st.title("Experimento 1 — Backtesting de 14 estrategias")
st.caption(
    "Capital inicial: $450,000 MXN · Aportación mensual: $10,000 MXN · "
    "Rebalanceo trimestral · DRIP 100% · Comisión 0.25%+IVA · Slippage 0.10% · "
    "Periodo 2021–2025"
)

BACKTEST_START        = "2021-01-01"
BACKTEST_END          = "2026-05-06"
INITIAL_CAPITAL       = 450_000.0
MONTHLY_CONTRIBUTION  = 10_000.0

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
            initial_capital=INITIAL_CAPITAL,
            monthly_contribution=MONTHLY_CONTRIBUTION,
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
        "Las 14 estrategias (E0–E13) cubren los principales enfoques de "
        "selección de activos. **E0 (Naive 1/N)** es la referencia interna. "
        "**E13** implementa los filtros del video: ocupación ≥ 90% y "
        "precio de mercado < NAV (valor teórico de los inmuebles). "
        "**CETES 28d** es el costo de oportunidad libre de riesgo."
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

st.subheader("Resumen de métricas — 2021–2025")

# Capital total invertido (igual para todas las estrategias)
total_invested = all_results["E0"]["invested_capital"].iloc[-1]
st.caption(
    f"Capital total aportado: **${total_invested:,.0f} MXN** "
    f"(${INITIAL_CAPITAL:,.0f} inicial + ${MONTHLY_CONTRIBUTION:,.0f}/mes × "
    f"{int((total_invested - INITIAL_CAPITAL) / MONTHLY_CONTRIBUTION):.0f} meses)"
)

rows = []
for code, s in STRATEGIES.items():
    res = all_results[code]
    pv  = res["portfolio_value"]
    m   = summary(pv, rf_daily)
    ganancia = pv.iloc[-1] - total_invested
    rows.append({
        "Cód.":           code,
        "Estrategia":     s["nombre"],
        "Categoría":      s["categoria"],
        "CAGR %":         m["CAGR"],
        "Sharpe":         m["Sharpe"],
        "MaxDD %":        m["MaxDD"],
        "Valor final $":  m["Valor final"],
        "Ganancia $":     round(ganancia, 0),
        "Ret. s/invertido %": round((pv.iloc[-1] / total_invested - 1) * 100, 1),
    })

df_summary = pd.DataFrame(rows).set_index("Cód.")

def _c_cagr(v):
    if v >= 30: return "background-color:#1a7a4a;color:white"
    if v >= 25: return "background-color:#2ca05a;color:white"
    if v >= 0:  return "background-color:#d4edda"
    return "background-color:#f8d7da"

def _c_mdd(v):
    if v >= -8:  return "background-color:#d4edda"
    if v >= -15: return "background-color:#fff3cd"
    return "background-color:#f8d7da"

def _c_sharpe(v):
    if v >= 1.5: return "background-color:#1a7a4a;color:white"
    if v >= 1.0: return "background-color:#d4edda"
    if v >= 0:   return ""
    return "background-color:#f8d7da"

def _c_ganancia(v):
    if v > 900_000: return "background-color:#1a7a4a;color:white"
    if v > 500_000: return "background-color:#2ca05a;color:white"
    if v > 0:       return "background-color:#d4edda"
    return "background-color:#f8d7da"

styled = (
    df_summary.style
    .map(_c_cagr,    subset=["CAGR %"])
    .map(_c_mdd,     subset=["MaxDD %"])
    .map(_c_sharpe,  subset=["Sharpe"])
    .map(_c_ganancia,subset=["Ganancia $"])
    .format({
        "CAGR %":              "{:.2f}",
        "Sharpe":              "{:.3f}",
        "MaxDD %":             "{:.2f}",
        "Valor final $":       "${:,.0f}",
        "Ganancia $":          "${:,.0f}",
        "Ret. s/invertido %":  "{:.1f}%",
    })
)
st.dataframe(styled, use_container_width=True, height=500)

# ── Sección 3: Selector ────────────────────────────────────────────────────────

st.divider()
all_codes  = list(STRATEGIES.keys())
default_sel = ["E0", "E9", "E11", "E13"]
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

# CETES acumulado con aportaciones mensuales (mismatch justo vs FIBRAs)
cetes_full = fetch_cetes(BACKTEST_START, BACKTEST_END)
rf_full    = cetes_daily_return(cetes_full["cetes_pct"])
trading    = all_results["E0"].index
rf_trading = rf_full.reindex(trading).ffill().fillna(0)

# Simular CETES con las mismas aportaciones
cetes_val = pd.Series(index=trading, dtype=float)
_cetes_balance = INITIAL_CAPITAL
for i, day in enumerate(trading):
    contrib = all_results["E0"]["contribution"].iloc[i]
    _cetes_balance = (_cetes_balance + contrib) * (1 + rf_trading.iloc[i])
    cetes_val.iloc[i] = _cetes_balance

cetes_cagr_ = cagr(cetes_val) * 100

# ── Tabs de gráficas ──────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Curvas de valor",
    "📉 Drawdown",
    "📅 Retornos anuales",
    "⚖️ Riesgo vs retorno",
    "💰 Dividendos recibidos",
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

# ── Tab 5: Dividendos recibidos ───────────────────────────────────────────────

with tab5:
    st.markdown(
        "**Sí, todas las estrategias reinvierten dividendos (DRIP 100%).**  \n"
        "En el día ex-dividendo, el motor calcula el efectivo recibido "
        "(acciones en cartera × dividendo por CBFI) y lo convierte "
        "automáticamente en más CBFIs al precio de cierre de ese día.  \n"
        "La gráfica de abajo muestra cuánto efectivo se habría generado "
        "cada mes si **no** se hubiera reinvertido — es el 'ingreso pasivo' "
        "equivalente de la estrategia."
    )
    st.divider()

    # Selector de estrategia (una sola)
    div_code = st.selectbox(
        "Ver dividendos de la estrategia:",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index("E13"),
        format_func=lambda c: f"{c} — {STRATEGIES[c]['nombre']}",
        key="div_strategy_selector",
    )

    div_series = all_results[div_code]["dividends_received"].copy()
    div_series.index = pd.to_datetime(div_series.index)

    # Últimos 3 años: 2024, 2025, 2026
    years_to_show = [2024, 2025, 2026]
    div_filtered = div_series[div_series.index.year.isin(years_to_show)]

    if div_filtered.empty:
        st.info("Sin dividendos registrados en 2024-2026 para esta estrategia.")
    else:
        # Agrupar por mes
        div_monthly = (
            div_filtered
            .resample("ME")
            .sum()
            .reset_index()
        )
        div_monthly["year"]  = div_monthly["date"].dt.year
        div_monthly["month"] = div_monthly["date"].dt.month
        div_monthly["mes"]   = div_monthly["date"].dt.strftime("%b")

        YEAR_COLORS = {2024: "#636EFA", 2025: "#00CC96", 2026: "#EF553B"}

        # ── Gráfica de barras agrupadas por año ───────────────────────────
        fig5 = go.Figure()
        for yr in years_to_show:
            df_yr = div_monthly[div_monthly["year"] == yr]
            if df_yr.empty:
                continue
            fig5.add_trace(go.Bar(
                x=df_yr["date"],
                y=df_yr["dividends_received"],
                name=str(yr),
                marker_color=YEAR_COLORS.get(yr, "#888"),
                hovertemplate=(
                    "<b>%{x|%B %Y}</b><br>"
                    "Dividendos: $%{y:,.0f} MXN"
                    "<extra></extra>"
                ),
            ))

        # Línea de promedio mensual del período completo
        avg_monthly = div_monthly["dividends_received"].mean()
        fig5.add_hline(
            y=avg_monthly,
            line_dash="dot", line_color="gray",
            annotation_text=f"Promedio ${avg_monthly:,.0f}/mes",
            annotation_position="top left",
        )

        fig5.update_layout(
            height=420,
            barmode="group",
            xaxis_title="",
            yaxis_title="Dividendos recibidos (MXN)",
            yaxis_tickformat="$,.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig5, use_container_width=True)

        # ── Métricas resumen ──────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        yr_cols = {2024: col1, 2025: col2, 2026: col3}
        for yr, col in yr_cols.items():
            yr_data = div_monthly[div_monthly["year"] == yr]
            if yr_data.empty:
                with col:
                    st.metric(label=f"Total {yr}", value="—")
                continue
            total_yr = yr_data["dividends_received"].sum()
            avg_yr   = yr_data[yr_data["dividends_received"] > 0]["dividends_received"].mean()
            with col:
                st.metric(
                    label=f"Total {yr}",
                    value=f"${total_yr:,.0f}",
                    delta=f"~${avg_yr:,.0f}/mes",
                )

        with col4:
            total_3yr = div_monthly["dividends_received"].sum()
            pv_now    = all_results[div_code]["portfolio_value"].iloc[-1]
            last_yr   = div_monthly["year"].max()
            yld_annl  = div_monthly[div_monthly["year"] == last_yr]["dividends_received"].sum()
            curr_yld  = yld_annl / pv_now * 100 if pv_now > 0 else 0
            st.metric(
                label="Total 2024–2026",
                value=f"${total_3yr:,.0f}",
                delta=f"Yield ~{curr_yld:.1f}% (año más reciente)",
            )

        # ── Tabla detallada ───────────────────────────────────────────────
        st.markdown("**Detalle mensual**")
        pivot = div_monthly.pivot_table(
            index="mes", columns="year",
            values="dividends_received", aggfunc="sum", fill_value=0
        )
        # Ordenar meses
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        pivot["Total"] = pivot.sum(axis=1)

        st.dataframe(
            pivot.style.format("${:,.0f}").background_gradient(
                cmap="Greens", subset=[c for c in pivot.columns if c != "Total"]
            ),
            use_container_width=True,
        )
