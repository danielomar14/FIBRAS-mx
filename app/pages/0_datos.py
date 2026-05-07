"""
Página 0: Calidad y cobertura de datos históricos de FIBRAs.

Muestra:
- Tabla de cobertura por ticker (%, fuente, estado)
- Heatmap de huecos en la serie de precios
- Botón para re-descargar datos
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Añadir raíz al path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.fetcher import fetch_all
from src.data.validator import run_full_validation

PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="Datos — FIBRAs MX", layout="wide")
st.title("Paso 0 — Calidad de Datos")
st.caption("Cobertura histórica 2016–2025 para las 15 FIBRAs mexicanas activas.")

# ── Botones de acción ─────────────────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Descargar datos", type="primary"):
        with st.spinner("Descargando desde yfinance…"):
            fetch_all(force=True)
        st.success("Descarga completada.")
        st.rerun()
    if st.button("Re-validar"):
        with st.spinner("Validando…"):
            run_full_validation()
        st.success("Validación completada.")
        st.rerun()

# ── Cargar reporte ────────────────────────────────────────────────────────
report_path = PROCESSED / "data_quality_report.json"
if not report_path.exists():
    st.warning(
        "No hay datos descargados todavía. "
        "Presiona **Descargar datos** para iniciar el Paso 0."
    )
    st.stop()

with open(report_path) as f:
    report = json.load(f)

# ── Tabla de cobertura ────────────────────────────────────────────────────
st.subheader("Cobertura por FIBRA")

rows = []
for ticker, info in report.items():
    pct = info.get("coverage_pct", 0)
    rows.append({
        "Ticker":       ticker,
        "Sector":       info.get("sector", "—"),
        "IPO":          info.get("ipo", "—"),
        "Cobertura (%)": pct,
        "Días reales":  info.get("actual_rows", 0),
        "Días esperados": info.get("expected_trading_days", 0),
        "Dividendos":   info.get("dividend_count", 0),
        "Fuente precios": info.get("source_prices", "—"),
        "Estado":       info.get("status", "—"),
        "Alertas precio": "; ".join(info.get("price_issues", [])) or "—",
        "Alertas dividendos": "; ".join(info.get("div_issues", [])) or "—",
    })

df_cov = pd.DataFrame(rows)

def _color_status(val: str) -> str:
    if val == "OK":
        return "background-color: #d4edda; color: #155724"
    elif val == "ALERTA":
        return "background-color: #fff3cd; color: #856404"
    return ""

def _color_pct(val: float) -> str:
    if val >= 90:
        return "color: #155724; font-weight: bold"
    elif val >= 80:
        return "color: #856404"
    else:
        return "color: #721c24; font-weight: bold"

styled = (
    df_cov.style
    .map(_color_status, subset=["Estado"])
    .map(_color_pct, subset=["Cobertura (%)"])
    .format({"Cobertura (%)": "{:.1f}%"})
)
st.dataframe(styled, use_container_width=True, height=450)

# Métricas de resumen
ok_count    = sum(1 for r in report.values() if r.get("status") == "OK")
alert_count = len(report) - ok_count
avg_cov     = sum(r.get("coverage_pct", 0) for r in report.values()) / len(report)

m1, m2, m3 = st.columns(3)
m1.metric("FIBRAs OK",    f"{ok_count} / {len(report)}")
m2.metric("Alertas",      alert_count)
m3.metric("Cobertura media", f"{avg_cov:.1f}%")

# ── Heatmap de cobertura ──────────────────────────────────────────────────
st.subheader("Heatmap de cobertura (%)")

if (PROCESSED / "precios_diarios.parquet").exists():
    df_prices = pd.read_parquet(PROCESSED / "precios_diarios.parquet")
    if "ticker" in df_prices.columns:
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices["year"] = df_prices.index.year

        pivot = (
            df_prices.groupby(["ticker", "year"])
            .size()
            .unstack(fill_value=0)
        )
        # Normalize to % of ~252 trading days
        pivot_pct = (pivot / 252 * 100).clip(upper=100).round(1)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_pct.values,
                x=[str(y) for y in pivot_pct.columns],
                y=pivot_pct.index.tolist(),
                colorscale=[
                    [0.0, "#d62728"],
                    [0.5, "#ffbb78"],
                    [0.8, "#aec7e8"],
                    [1.0, "#1f77b4"],
                ],
                zmin=0, zmax=100,
                text=pivot_pct.values,
                texttemplate="%{text:.0f}%",
                hovertemplate="<b>%{y}</b> — %{x}<br>Cobertura: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            title="Cobertura por año y FIBRA (% de días hábiles)",
            xaxis_title="Año",
            yaxis_title="",
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Descarga los datos primero para ver el heatmap.")

# ── Distribución de dividendos ────────────────────────────────────────────
if (PROCESSED / "distribuciones.parquet").exists():
    st.subheader("Distribuciones históricas")
    df_divs = pd.read_parquet(PROCESSED / "distribuciones.parquet")
    if not df_divs.empty and "date" in df_divs.columns:
        df_divs["date"] = pd.to_datetime(df_divs["date"])
        df_divs["year"] = df_divs["date"].dt.year

        pivot_divs = (
            df_divs.groupby(["ticker", "year"])["dividend"]
            .sum()
            .unstack(fill_value=0)
            .round(4)
        )

        fig2 = go.Figure(
            data=go.Heatmap(
                z=pivot_divs.values,
                x=[str(y) for y in pivot_divs.columns],
                y=pivot_divs.index.tolist(),
                colorscale="Greens",
                text=pivot_divs.values,
                texttemplate="%{text:.2f}",
                hovertemplate="<b>%{y}</b> — %{x}<br>Dividendo total: $%{z:.4f} MXN/CBFI<extra></extra>",
            )
        )
        fig2.update_layout(
            title="Dividendo total anual por CBFI (MXN)",
            xaxis_title="Año",
            yaxis_title="",
            height=450,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)
