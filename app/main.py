"""
FIBRAS Mexicanas — Streamlit App Entry Point

Run with:
    conda activate fibras-mx
    streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="FIBRAs Mexicanas",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FIBRAs Mexicanas — Análisis Cuantitativo")
st.markdown(
    """
    Bienvenido al dashboard de análisis de las **15 FIBRAs activas** en México.

    Usa el menú de la izquierda para navegar:

    | Página | Contenido |
    |--------|-----------|
    | **0 — Datos** | Calidad y cobertura de datos históricos |
    | **1 — Estrategias** | Backtest comparativo de 11 estrategias (E0–E10) |
    | **2 — Algoritmo Genético** | Optimización GA, cromosoma ganador, validación |

    ---
    **Universo:** FUNO11, FIBRAPL14, FIBRAMQ12, DANHOS13, FMTY14, FIHO12, FINN13,
    FSHOP13, FIBRAUP18, FNOVA17, FPLUS16, STORAGE18, FSITES20, EDUCA18, NEXT25

    **Período:** 2016–2025 | **Capital inicial:** $200,000 MXN | **Rebalanceo:** Trimestral
    """
)
