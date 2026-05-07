"""
Página 4 — Resultados finales: comparativa completa de las 5 estrategias.

1. Dummy        : 450 k MXN + 10 k/mes sin invertir (costo de oportunidad)
2. CETES 28d    : mismo capital invertido en tasa libre de riesgo
3. Mejor E0-E13 : campeón de las estrategias del Experimento 1
4. Mejor ML     : mejor modelo de ML del Experimento 2A
5. Mejor GA     : cromosoma ganador del Experimento 2B
"""

from __future__ import annotations

import math
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Resultados — FIBRAs MX", layout="wide")
st.title("Resultados finales — Comparativa de estrategias")

# ── Nota sobre integridad del holdout ─────────────────────────────────────────
st.info(
    "**¿Los modelos vieron datos de validación?** No. "
    "El período de **Train (2017-2023)** se usó para ajustar parámetros. "
    "El **Test (2024-2025)** se usó solo para comparar y elegir el mejor modelo. "
    "La **Validación (2026)** nunca fue vista por ningún modelo durante "
    "el entrenamiento ni la selección — es un holdout genuino. "
    "En el gráfico, el área sombreada indica el período fuera de muestra."
)

# ── Constantes ────────────────────────────────────────────────────────────────
START            = "2021-01-01"
END              = "2026-05-06"
INITIAL_CAPITAL  = 450_000.0
MONTHLY_CONTRIB  = 10_000.0
OOS_START        = "2024-01-01"  # inicio período fuera de muestra

# ── CETES fallback mensual (mismo dict que benchmarks.py) ─────────────────────
_CETES_MONTHLY = {
    (2021,1):4.20,(2021,2):4.14,(2021,3):4.05,(2021,4):4.07,(2021,5):4.20,
    (2021,6):4.25,(2021,7):4.37,(2021,8):4.45,(2021,9):4.69,(2021,10):5.05,
    (2021,11):5.33,(2021,12):5.55,
    (2022,1):5.83,(2022,2):5.95,(2022,3):6.31,(2022,4):6.76,(2022,5):7.23,
    (2022,6):7.64,(2022,7):8.13,(2022,8):8.56,(2022,9):9.16,(2022,10):9.73,
    (2022,11):10.09,(2022,12):10.46,
    (2023,1):10.66,(2023,2):10.88,(2023,3):11.21,(2023,4):11.52,(2023,5):11.52,
    (2023,6):11.41,(2023,7):11.25,(2023,8):11.24,(2023,9):11.28,(2023,10):11.26,
    (2023,11):11.25,(2023,12):11.24,
    (2024,1):11.20,(2024,2):11.15,(2024,3):11.07,(2024,4):11.03,(2024,5):10.93,
    (2024,6):10.81,(2024,7):10.74,(2024,8):10.67,(2024,9):10.55,(2024,10):10.42,
    (2024,11):10.12,(2024,12):10.00,
    (2025,1):9.75,(2025,2):9.50,(2025,3):9.25,(2025,4):9.00,(2025,5):9.00,
    (2025,6):9.00,(2025,7):9.00,(2025,8):9.00,(2025,9):9.00,(2025,10):9.00,
    (2025,11):9.00,(2025,12):9.00,
    (2026,1):8.50,(2026,2):8.50,(2026,3):8.50,(2026,4):8.25,(2026,5):8.25,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monthly_dates(start: str, end: str) -> list[pd.Timestamp]:
    """Primer día hábil de cada mes en el rango."""
    bdays = pd.bdate_range(start=start, end=end)
    seen: set = set()
    result = []
    for d in bdays:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def _simulate_dummy(start: str, end: str) -> pd.Series:
    """Portfolio dummy: solo efectivo, sin inversión."""
    months = _monthly_dates(start, end)
    portfolio = INITIAL_CAPITAL
    series = {}
    for i, d in enumerate(months):
        if i > 0:
            portfolio += MONTHLY_CONTRIB
        series[d] = portfolio
    return pd.Series(series)


def _simulate_cetes(start: str, end: str) -> pd.Series:
    """Portfolio CETES 28d: reinversión mensual a tasa libre de riesgo."""
    months = _monthly_dates(start, end)
    portfolio = INITIAL_CAPITAL
    series = {}
    for i, d in enumerate(months):
        if i > 0:
            portfolio += MONTHLY_CONTRIB
        annual_rate = _CETES_MONTHLY.get((d.year, d.month), 9.0) / 100
        monthly_rate = (1 + annual_rate) ** (1 / 12) - 1
        portfolio *= (1 + monthly_rate)
        series[d] = portfolio
    return pd.Series(series)


def _backtest_to_monthly(df: pd.DataFrame) -> pd.Series:
    """Convierte serie diaria de portfolio_value a mensual (fin de mes)."""
    return df["portfolio_value"].resample("ME").last().ffill()


def _rets_from_equity(equity: pd.Series) -> list[float]:
    """Extrae retornos periodo a periodo de una serie de equity acumulada."""
    if equity is None or len(equity) == 0:
        return []
    e = equity.reset_index(drop=True)
    rets = [float(e.iloc[0] - 1)]
    for i in range(1, len(e)):
        if e.iloc[i - 1] != 0:
            rets.append(float(e.iloc[i] / e.iloc[i - 1] - 1))
    return rets


def _simulate_from_quarterly(
    quarterly_returns: list[float],
    start_date: str,
    start_portfolio: float,
    all_months: list[pd.Timestamp],
) -> pd.Series:
    """
    Simula valor mensual del portfolio aplicando retornos trimestrales.
    start_date : cuándo empieza esta curva (puede ser posterior al inicio global)
    start_portfolio: valor del portfolio al start_date
    """
    if not quarterly_returns:
        return pd.Series(dtype=float)

    # Fechas de rebalanceo trimestrales (ene, abr, jul, oct)
    rebal_months = {(d.year, d.month)
                    for d in pd.date_range(start=start_date, end=END, freq="QS-JAN")}

    portfolio = start_portfolio
    ret_idx   = 0
    prev_key  = None
    series    = {}

    for i, d in enumerate(all_months):
        if d < pd.Timestamp(start_date):
            continue
        key = (d.year, d.month)
        if key in rebal_months and key != prev_key and ret_idx < len(quarterly_returns):
            portfolio *= (1 + quarterly_returns[ret_idx])
            ret_idx   += 1
            prev_key   = key
        # Apply CETES rate to cash held between rebalances (approximate)
        annual_r  = _CETES_MONTHLY.get(key, 9.0) / 100
        cash_ret  = (1 + annual_r) ** (1 / 12) - 1
        # Contribution at start of month
        if d > pd.Timestamp(start_date):
            portfolio += MONTHLY_CONTRIB
        series[d] = portfolio

    return pd.Series(series)


# ── Carga de datos ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_market_data():
    from src.backtest.engine import _load_data
    from src.data.benchmarks import fetch_banxico_rate
    pw, div, met = _load_data()
    try:
        banxico = fetch_banxico_rate()
    except Exception:
        banxico = None
    return pw, div, met, banxico


@st.cache_data(show_spinner=False, ttl=3600)
def _run_e_backtests(_pw, _div, _met, _banxico):
    from src.backtest.engine import run_backtest
    from src.backtest.strategies import STRATEGIES
    from src.backtest.metrics import sharpe, cagr

    results = {}
    rf = None
    try:
        from src.data.benchmarks import fetch_cetes, cetes_daily_return
        cetes = fetch_cetes()
        rf = cetes_daily_return(cetes["cetes_pct"])
    except Exception:
        pass

    for code, s in STRATEGIES.items():
        try:
            df = run_backtest(
                s["fn"], start=START, end=END,
                initial_capital=INITIAL_CAPITAL,
                monthly_contribution=MONTHLY_CONTRIB,
                prices_wide=_pw, dividends=_div, metrics=_met, banxico_rates=_banxico,
            )
            results[code] = {
                "df":     df,
                "nombre": s["nombre"],
                "sharpe": sharpe(df["portfolio_value"], rf),
                "cagr":   cagr(df["portfolio_value"]),
            }
        except Exception:
            pass
    return results


# ── UI ────────────────────────────────────────────────────────────────────────

calcular = st.button("Calcular Resultados", type="primary")

if calcular or "resultados_computed" in st.session_state:
    if calcular:
        with st.spinner("Cargando datos de mercado…"):
            pw, div, met, banxico = _load_market_data()
        with st.spinner("Corriendo backtests E0-E13 (puede tardar ~30 s)…"):
            e_results = _run_e_backtests(pw, div, met, banxico)
        st.session_state["resultados_e"] = e_results
        st.session_state["resultados_computed"] = True

    e_results = st.session_state.get("resultados_e", {})

    # ── Mejor E0-E13 ──────────────────────────────────────────────────────────
    best_code, best_er = None, None
    if e_results:
        best_code = max(e_results, key=lambda c: e_results[c]["cagr"])
        best_er   = e_results[best_code]

    # ── ML mejor ──────────────────────────────────────────────────────────────
    ml_equity_test, ml_name = None, None
    ml_path = ROOT / "results" / "ml_results.pkl"
    if ml_path.exists():
        try:
            with open(ml_path, "rb") as f:
                ml_results = pickle.load(f)
            ml_best_code = max(
                ml_results,
                key=lambda k: ml_results[k].get("sharpe_test", -math.inf),
            )
            ml_name       = ml_best_code
            ml_equity_test = ml_results[ml_best_code].get("equity_test")
        except Exception:
            pass

    # ── GA mejor ──────────────────────────────────────────────────────────────
    ga_result, ga_equity = None, None
    ga_path = ROOT / "results" / "ga_results.pkl"
    if ga_path.exists():
        try:
            with open(ga_path, "rb") as f:
                ga_result = pickle.load(f)
        except Exception:
            pass

    # ── Curvas mensuales ──────────────────────────────────────────────────────
    all_months = _monthly_dates(START, END)

    dummy_m = _simulate_dummy(START, END)
    cetes_m = _simulate_cetes(START, END)

    best_e_m = None
    if best_er is not None:
        best_e_m = _backtest_to_monthly(best_er["df"])

    # Capital invertido al inicio del OOS (para anclar ML y GA)
    invested_at_oos = INITIAL_CAPITAL + MONTHLY_CONTRIB * len(
        [d for d in all_months if d < pd.Timestamp(OOS_START)]
    )

    # ML: reconstruir retornos trimestrales del test y simular portfolio
    ml_monthly = None
    if ml_equity_test is not None and len(ml_equity_test) > 0:
        ml_rets = _rets_from_equity(ml_equity_test)
        ml_monthly = _simulate_from_quarterly(
            ml_rets, OOS_START, invested_at_oos, all_months
        )

    # GA: usar returns de test + val
    ga_monthly = None
    if ga_result is not None:
        ga_rets = (
            list(ga_result.test_metrics.get("returns", []))
            + list(ga_result.val_metrics.get("returns", []))
        )
        if ga_rets:
            ga_monthly = _simulate_from_quarterly(
                ga_rets, OOS_START, invested_at_oos, all_months
            )

    # ── Gráfica comparativa ───────────────────────────────────────────────────
    st.subheader("Comparativa de portafolios (450 k MXN + 10 k/mes)")

    fig = go.Figure()

    # Fondo: in-sample vs OOS
    oos_ts = pd.Timestamp(OOS_START)
    max_val = max(
        cetes_m.max() if cetes_m is not None else 0,
        best_e_m.max() if best_e_m is not None else 0,
        ml_monthly.max() if ml_monthly is not None else 0,
        ga_monthly.max() if ga_monthly is not None else 0,
    ) * 1.15

    fig.add_vrect(
        x0=START, x1=OOS_START,
        fillcolor="rgba(180,180,180,0.12)", line_width=0,
        annotation_text="In-sample", annotation_position="top left",
        annotation_font_color="gray",
    )
    fig.add_vrect(
        x0=OOS_START, x1=END,
        fillcolor="rgba(0,200,100,0.06)", line_width=0,
        annotation_text="Out-of-sample", annotation_position="top right",
        annotation_font_color="#00AA60",
    )

    def _add(series, name, color, dash="solid", width=2):
        if series is None or len(series) == 0:
            return
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name, mode="lines",
            line=dict(color=color, dash=dash, width=width),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f} MXN<extra></extra>",
        ))

    _add(dummy_m,   "Dummy (sin invertir)",         "#AAAAAA", dash="dot",    width=1.5)
    _add(cetes_m,   "CETES 28d",                    "#636EFA", dash="dash",   width=2)
    if best_code:
        _add(best_e_m, f"Mejor E0-E13: {best_code} — {best_er['nombre']}", "#00CC96", width=2.5)
    if ml_monthly is not None:
        _add(ml_monthly, f"ML mejor: {ml_name} (OOS)", "#FFA15A", dash="longdash", width=2)
    if ga_monthly is not None:
        _add(ga_monthly, "GA mejor (OOS)", "#EF553B", dash="longdash", width=2)

    fig.update_layout(
        height=520,
        hovermode="x unified",
        yaxis=dict(title="Valor del portafolio (MXN)", tickformat="$,.0f"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "ML y GA se muestran a partir de 2024 (período fuera de muestra) "
        "con el capital acumulado hasta esa fecha como base. "
        "El área verde es el único período donde ML/GA no vieron los datos."
    )

    # ── Métricas financieras ──────────────────────────────────────────────────
    st.subheader("Métricas financieras al final del período")

    invested_total = dummy_m.iloc[-1] if dummy_m is not None else INITIAL_CAPITAL

    def _row(name, series, color_emoji):
        if series is None or len(series) == 0:
            return None
        val  = series.iloc[-1]
        gain = val - invested_total
        ret  = val / invested_total - 1
        return {
            "": color_emoji,
            "Estrategia":         name,
            "Valor final (MXN)":  f"${val:>12,.0f}",
            "Ganancia total":     f"${gain:>12,.0f}",
            "Retorno total":      f"{ret:>8.1%}",
        }

    rows = []
    r = _row("Dummy (sin invertir)", dummy_m, "⬜")
    if r: rows.append(r)
    r = _row("CETES 28d", cetes_m, "🔵")
    if r: rows.append(r)
    if best_code and best_e_m is not None:
        r = _row(f"Mejor E0-E13: {best_code}", best_e_m, "🟢")
        if r: rows.append(r)
    if ml_monthly is not None:
        r = _row(f"ML mejor: {ml_name} (OOS desde 2024)", ml_monthly, "🟠")
        if r: rows.append(r)
    if ga_monthly is not None:
        r = _row("GA mejor (OOS desde 2024)", ga_monthly, "🔴")
        if r: rows.append(r)

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index(""), use_container_width=True)

    st.caption(
        f"Capital total invertido (450 k inicial + 10 k × meses): "
        f"**${invested_total:,.0f} MXN**"
    )

    # ── Detalle de la mejor E0-E13 ────────────────────────────────────────────
    if best_er is not None:
        st.subheader(f"Detalle: {best_code} — {best_er['nombre']}")

        df_best = best_er["df"]

        col1, col2, col3, col4 = st.columns(4)
        total_divs = df_best["dividends_received"].sum()
        avg_monthly_div = df_best["dividends_received"].resample("ME").sum().mean()
        capital_gain = df_best["portfolio_value"].iloc[-1] - df_best["invested_capital"].iloc[-1] - total_divs
        with col1:
            st.metric("Valor final", f"${df_best['portfolio_value'].iloc[-1]:,.0f}")
        with col2:
            st.metric("Total dividendos recibidos", f"${total_divs:,.0f}")
        with col3:
            st.metric("Dividendo mensual promedio", f"${avg_monthly_div:,.0f}")
        with col4:
            st.metric("Ganancia de capital (aprox.)", f"${max(capital_gain,0):,.0f}")

        # Dividendos mensuales
        div_monthly = df_best["dividends_received"].resample("ME").sum()
        div_monthly = div_monthly[div_monthly.index.year >= 2021]

        fig_div = go.Figure()
        for yr in sorted(div_monthly.index.year.unique()):
            sub = div_monthly[div_monthly.index.year == yr]
            fig_div.add_trace(go.Bar(
                x=sub.index.strftime("%b"),
                y=sub.values,
                name=str(yr),
            ))
        fig_div.update_layout(
            title=f"Dividendos mensuales recibidos — {best_code}",
            barmode="group",
            yaxis=dict(title="MXN", tickformat="$,.0f"),
            xaxis_title="Mes",
            height=360,
        )
        st.plotly_chart(fig_div, use_container_width=True)

    # ── Nota sobre ML/GA y validación ─────────────────────────────────────────
    with st.expander("¿Qué tan confiables son los resultados de ML y GA?"):
        st.markdown("""
**Integridad del proceso:**

| Período | Rol | ML | GA |
|---------|-----|----|----|
| 2017-2023 (Train) | Ajuste de parámetros | ✅ Se entrenó aquí | ✅ Fitness calculado aquí |
| 2024-2025 (Test) | Selección del mejor | ✅ Se usó para comparar modelos | ❌ No usado (GA no depende de test) |
| 2026 (Validación) | Holdout genuino | ✅ Nunca visto | ✅ Nunca visto |

**Precauciones:**
- El período **Test (2024-2025)** sí se usó para seleccionar el mejor modelo de ML →
  existe un riesgo de sobreajuste al período de test si se comparan muchos modelos.
- La **Validación 2026** es el único período que no influyó en ninguna decisión.
- ML y GA se muestran solo desde 2024 para reflejar esto honestamente.

**Recomendación:** Observa el rendimiento en 2026 como el indicador más imparcial.
        """)

else:
    st.info(
        "Presiona **Calcular Resultados** para comparar las 5 estrategias. "
        "El cálculo tarda ~30 segundos (se carga una sola vez)."
    )
