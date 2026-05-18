"""
Página 5 — Reportes de portafolio.

Muestra los CSVs, gráficas y comentarios generados por scripts/export_results.py.
Incluye botón para regenerar los archivos y descarga directa.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT    = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Reportes — FIBRAs MX", layout="wide")
st.title("Reportes de portafolio")
st.caption(
    "CSVs diarios (lunes-viernes) con valor total y % por FIBRA, "
    "gráficas anotadas y comentarios de eventos relevantes."
)

# ── Generar / actualizar ──────────────────────────────────────────────────────
with st.expander("⚙️ Generar / actualizar reportes", expanded=not REPORTS.exists() or not any(REPORTS.glob("*.csv"))):
    st.info(
        "Los reportes de ML y GA requieren que primero hayas corrido los modelos "
        "en la pestaña **Algoritmo** (Entrenar todos los modelos + Correr GA)."
    )
    if st.button("Generar reportes ahora", type="primary"):
        with st.spinner("Corriendo backtest E0-E13 y generando archivos… (~30 s)"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "export_results.py")],
                capture_output=True, text=True, cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.success("Reportes generados correctamente.")
            st.code(result.stdout, language=None)
        else:
            st.error("Error al generar reportes.")
            st.code(result.stderr or result.stdout, language=None)
        st.rerun()

if not REPORTS.exists() or not any(REPORTS.glob("*.csv")):
    st.warning("No hay reportes generados todavía. Usa el botón de arriba para crearlos.")
    st.stop()

# ── Estrategias disponibles ───────────────────────────────────────────────────
ESTRATEGIAS: list[tuple[str, str]] = []
for csv_path in sorted(REPORTS.glob("portafolio_*.csv")):
    key  = csv_path.stem.replace("portafolio_", "")
    name = {
        "dummy":  "Dummy (sin invertir)",
        "cetes":  "CETES 28d",
    }.get(key, key.replace("_", " ").upper())
    ESTRATEGIAS.append((key, name))

if not ESTRATEGIAS:
    st.info("No se encontraron CSVs de portafolio. Genera los reportes primero.")
    st.stop()

tabs = st.tabs([name for _, name in ESTRATEGIAS] + ["Consolidado"])

# ── Tabs por estrategia ───────────────────────────────────────────────────────
for tab, (key, name) in zip(tabs[:-1], ESTRATEGIAS):
    with tab:
        col_chart, col_data = st.columns([3, 2], gap="large")

        # Gráfica
        png_path = REPORTS / f"{key}.png"
        with col_chart:
            st.subheader(f"Gráfica — {name}")
            if png_path.exists():
                st.image(str(png_path), use_container_width=True)
            else:
                st.info("Gráfica no disponible aún.")

        # Comentarios + descarga CSV
        with col_data:
            # Comentarios
            comments_path = REPORTS / f"{key}_comentarios.csv"
            st.subheader("Eventos destacados (10 máx.)")
            if comments_path.exists():
                cdf = pd.read_csv(comments_path)
                st.dataframe(cdf, use_container_width=True, hide_index=True)
                with open(comments_path, "rb") as f:
                    st.download_button(
                        f"⬇ Descargar comentarios CSV",
                        data=f.read(),
                        file_name=comments_path.name,
                        mime="text/csv",
                        key=f"dl_comments_{key}",
                    )
            else:
                st.info("Comentarios no disponibles.")

            st.divider()

            # CSV principal
            csv_path = REPORTS / f"portafolio_{key}.csv"
            st.subheader("CSV diario")
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                # Vista previa: últimas 10 filas, columnas compactas
                preview_cols = ["portafolio_total_mxn"] + [
                    c for c in df.columns if c.endswith("_pct")
                ][:6]
                st.dataframe(
                    df[preview_cols].tail(10).style.format("{:.1f}"),
                    use_container_width=True,
                )
                st.caption(f"{len(df):,} filas · {len(df.columns)} columnas · vista previa últimas 10 filas")
                with open(csv_path, "rb") as f:
                    st.download_button(
                        f"⬇ Descargar {csv_path.name}",
                        data=f.read(),
                        file_name=csv_path.name,
                        mime="text/csv",
                        key=f"dl_csv_{key}",
                    )
            else:
                st.info("CSV no disponible.")

# ── Tab Consolidado ───────────────────────────────────────────────────────────
with tabs[-1]:
    cons_path = REPORTS / "consolidado.csv"
    st.subheader("Consolidado — todas las estrategias")
    if cons_path.exists():
        df_cons = pd.read_csv(cons_path, index_col=0, parse_dates=True)

        # Gráfica rápida con Streamlit nativo
        st.line_chart(df_cons, height=380)

        st.caption(f"{len(df_cons):,} filas × {len(df_cons.columns)} estrategias")
        st.dataframe(df_cons.tail(10).style.format("${:,.0f}"), use_container_width=True)

        with open(cons_path, "rb") as f:
            st.download_button(
                "⬇ Descargar consolidado.csv",
                data=f.read(),
                file_name="consolidado.csv",
                mime="text/csv",
            )
    else:
        st.info("Consolidado no disponible. Genera los reportes primero.")
