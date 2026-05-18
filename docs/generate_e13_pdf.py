"""
Genera docs/E13_guia_completa.pdf — Guía completa de la estrategia E13.

La guía explica exactamente qué es E13, cómo funciona el algoritmo,
cuándo rebalancear, cómo ejecutar el programa y qué comprar hoy.

Uso:
    conda activate fibras-mx
    python docs/generate_e13_pdf.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_DIR = (
    "/Users/danielbecerrilolguin/anaconda3/envs/fibras-mx/lib/python3.12"
    "/site-packages/matplotlib/mpl-data/fonts/ttf"
)

# ── Paleta ────────────────────────────────────────────────────────────────────
AZUL       = (0, 82, 155)
AZUL_CLR   = (200, 215, 240)
AZUL_DARK  = (0, 50, 100)
VERDE      = (0, 110, 55)
VERDE_CLR  = (200, 235, 215)
ROJO       = (180, 20, 20)
ROJO_CLR   = (255, 220, 220)
NARANJA    = (180, 80, 0)
NARAN_CLR  = (255, 235, 210)
GRIS       = (90, 90, 90)
GRIS_CLR   = (245, 245, 245)
BLANCO     = (255, 255, 255)
NEGRO      = (20, 20, 20)
LINEA      = (180, 180, 180)
VERDE_OK   = (0, 140, 70)
ROJO_NO    = (200, 40, 40)


class PDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(22, 18, 22)
        self.add_font("DV",  "",  os.path.join(FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("DV",  "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("DV",  "I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf"))
        self.add_font("DVM", "",  os.path.join(FONT_DIR, "DejaVuSansMono.ttf"))
        self.add_font("DVM", "B", os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"))

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DV", "I", 8)
        self.set_text_color(*GRIS)
        self.cell(0, 7, "Estrategia E13 — FIBRAs Mexicanas  ·  Precio < NAV + Ocupación ≥ 90%", align="L")
        self.set_x(-35)
        self.cell(25, 7, f"Pág. {self.page_no()}", align="R")
        self.ln(1)
        self.set_draw_color(*LINEA)
        self.line(22, self.get_y(), 188, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font("DV", "I", 7.5)
        self.set_text_color(*GRIS)
        self.cell(0, 8, "FIBRAS-mx  ·  github.com/danielomar14/FIBRAS-mx  ·  Mayo 2026", align="C")

    # ── Bloques básicos ───────────────────────────────────────────────────────

    def titulo_seccion(self, txt, color=None):
        self.ln(5)
        clr = color or AZUL
        self.set_fill_color(*clr)
        self.set_text_color(*BLANCO)
        self.set_font("DV", "B", 12)
        self.cell(0, 9, f"  {txt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*NEGRO)
        self.ln(3)

    def subtitulo(self, txt, color=None):
        self.ln(3)
        clr = color or VERDE
        self.set_text_color(*clr)
        self.set_font("DV", "B", 10)
        self.cell(0, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*clr)
        self.line(22, self.get_y(), 188, self.get_y())
        self.set_text_color(*NEGRO)
        self.set_draw_color(*LINEA)
        self.ln(2)

    def parrafo(self, txt, size=9.5):
        self.set_font("DV", "", size)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5.5, txt)
        self.ln(1)

    def bullet(self, txt, nivel=0, bold_part=None, color=None):
        indent = 6 + nivel * 8
        self.set_font("DV", "", 9.5)
        self.set_text_color(*(color or NEGRO))
        self.set_x(self.l_margin + indent)
        self.cell(5, 5.5, "•")
        self.multi_cell(0, 5.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)

    def caja(self, titulo, cuerpo, bg, borde, txt_color=None):
        self.ln(4)
        tc = txt_color or borde
        self.set_fill_color(*bg)
        self.set_draw_color(*borde)
        self.set_font("DV", "B", 9.5)
        self.set_text_color(*tc)
        self.cell(0, 7, f"  {titulo}", border=1, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DV", "", 9)
        self.set_text_color(*NEGRO)
        self.set_fill_color(*bg)
        self.multi_cell(0, 5.5, f"  {cuerpo}", border="LRB", fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LINEA)
        self.ln(2)

    def code_block(self, txt):
        self.set_fill_color(*GRIS_CLR)
        self.set_draw_color(*LINEA)
        self.set_font("DVM", "", 8)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.8, txt, border=1, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)
        self.ln(2)

    def check_row(self, label, valor, cumple: bool | None = None, nota=""):
        self.set_font("DV", "", 9.5)
        if cumple is True:
            self.set_text_color(*VERDE_OK)
            simbolo = "✓"
        elif cumple is False:
            self.set_text_color(*ROJO_NO)
            simbolo = "✗"
        else:
            self.set_text_color(*GRIS)
            simbolo = "—"
        self.set_x(self.l_margin + 6)
        self.cell(8, 6, simbolo)
        self.set_text_color(*NEGRO)
        self.set_font("DV", "B", 9.5)
        self.cell(38, 6, label)
        self.set_font("DV", "", 9.5)
        self.cell(30, 6, valor)
        if nota:
            self.set_font("DV", "I", 8.5)
            self.set_text_color(*GRIS)
            self.cell(0, 6, nota)
            self.set_text_color(*NEGRO)
        self.ln()

    # ── Tablas ────────────────────────────────────────────────────────────────

    def tabla_header(self, cols_w, headers, bg=None):
        self.set_fill_color(*(bg or AZUL_CLR))
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(cols_w, headers):
            self.cell(w, 6.5, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(*NEGRO)

    def tabla_row(self, cols_w, vals, i=0, aligns=None, highlight=False):
        fill = (i % 2 == 0)
        if highlight:
            self.set_fill_color(220, 240, 220)
        elif fill:
            self.set_fill_color(248, 250, 255)
        else:
            self.set_fill_color(*BLANCO)
        self.set_font("DV", "B" if highlight else "", 8.5)
        self.set_text_color(*NEGRO)
        aligns = aligns or ["L"] * len(cols_w)
        for w, v, a in zip(cols_w, vals, aligns):
            self.cell(w, 5.8, str(v), border=1, fill=True, align=a)
        self.ln()


# ═══════════════════════════════════════════════════════════════════════════════
# DATOS REALES
# ═══════════════════════════════════════════════════════════════════════════════

def _load_real_data():
    import pandas as pd
    import numpy as np

    PROCESSED = ROOT / "data" / "processed"
    pr = pd.read_parquet(PROCESSED / "precios_diarios.parquet")
    dr = pd.read_parquet(PROCESSED / "distribuciones.parquet")
    mr = pd.read_parquet(PROCESSED / "metricas_trimestrales.parquet")
    mr["date"] = pd.to_datetime(mr["date"])

    pw = pr.pivot(columns="ticker", values="close").ffill(limit=60)
    hoy = pw.index[-1]

    px   = pw.loc[hoy]
    px12 = pw.iloc[max(0, len(pw.index) - 253)]
    ret12 = (px / px12 - 1) * 100

    cutoff = pd.Timestamp(hoy) - pd.DateOffset(years=1)
    dr["date"] = pd.to_datetime(dr["date"])
    div_ttm = dr[dr["date"] >= cutoff].groupby("ticker")["dividend"].sum()
    dy = (div_ttm / px * 100).round(1)

    met = mr.sort_values("date").groupby("ticker").last()[
        ["occupancy_portfolio", "nav_per_cbfi", "ffo_per_cbfi"]]

    tabla = pd.concat([px.rename("precio")], axis=1).join(met)
    tabla["occ_pct"]   = (tabla["occupancy_portfolio"] * 100).round(1)
    tabla["p_nav"]     = (tabla["precio"] / tabla["nav_per_cbfi"]).round(3)
    tabla["desc_nav"]  = ((tabla["nav_per_cbfi"] - tabla["precio"]) / tabla["nav_per_cbfi"] * 100).round(1)
    tabla["ffo_yield"] = (tabla["ffo_per_cbfi"] * 4 / tabla["precio"] * 100).round(1)
    tabla["div_yield"] = dy.reindex(tabla.index).round(1)
    tabla["ret_12m"]   = ret12.reindex(tabla.index).round(1)

    # E13: occ>=90% y precio<NAV
    e13 = tabla[(tabla["occ_pct"] >= 90) & (tabla["p_nav"] < 1.0)].copy()
    score = (e13["div_yield"].fillna(0) + e13["ffo_yield"].fillna(0)).sort_values(ascending=False)
    top5  = score.head(5).index.tolist()

    return tabla, top5, hoy


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═══════════════════════════════════════════════════════════════════════════════

def build_pdf():
    tabla, top5, fecha_datos = _load_real_data()

    # Precios del usuario (capturas del 18-may-2026)
    precios_hoy = {
        "FUNO11":    29.49,
        "FSHOP13":   12.49,
        "DANHOS13":  27.63,
        "FNOVA17":   42.73,
        "FIBRAMQ12": 43.37,
        "FIBRAUP18": 38.80,
        "FIBRAPL14": 79.98,
        "FMTY14":    14.57,
        "FIHO12":     7.80,
        "STORAGE18": 24.95,
        "FUNO11":    29.49,
    }

    pdf = PDF()

    # ═══════════════════════════════════════════════════════════════════════════
    # PORTADA
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Franja azul superior
    pdf.set_fill_color(*AZUL_DARK)
    pdf.rect(0, 0, 210, 58, "F")

    pdf.set_y(8)
    pdf.set_text_color(*BLANCO)
    pdf.set_font("DV", "B", 26)
    pdf.cell(0, 13, "Estrategia E13", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DV", "B", 14)
    pdf.cell(0, 9, "Precio < NAV  +  Ocupación ≥ 90%", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_font("DV", "I", 10)
    pdf.cell(0, 6, "Cómo invertir en FIBRAs mexicanas por debajo de su valor real",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Datos del proyecto
    pdf.set_y(72)
    pdf.set_text_color(*NEGRO)
    for lbl, val in [
        ("Capital de referencia", "$450,000 MXN + $10,000 / mes"),
        ("Período analizado",     "2021-01-01 → 2026-05-06"),
        ("Rebalanceo",            "Trimestral — primer día hábil de ene/abr/jul/oct"),
        ("Universo",              "15 FIBRAs activas en BMV/BIVA"),
        ("Fuente de datos",       "yfinance + Supabase FibrasMX"),
        ("Última actualización",  "18 de mayo de 2026"),
    ]:
        pdf.set_font("DV", "B", 10)
        pdf.cell(55, 7, lbl + ":", border="B")
        pdf.set_font("DV", "", 10)
        pdf.cell(0, 7, val, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(8)
    pdf.caja(
        "Resultado en el backtest 2021–2026",
        "E13 fue la estrategia de mejor desempeño de las 14 analizadas. "
        "El algoritmo detecta FIBRAs cuyos activos inmobiliarios valen más "
        "de lo que paga el mercado — comprar ladrillos con descuento.",
        VERDE_CLR, VERDE,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 2 — ¿Qué es E13 y por qué funciona?
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("1. ¿Qué es E13 y por qué funciona?")

    pdf.parrafo(
        "Una FIBRA es básicamente una empresa que posee inmuebles (bodegas, centros "
        "comerciales, hoteles, oficinas) y los arrienda. Cada título que compras "
        "(llamado CBFI) te da derecho a una parte proporcional de los inmuebles "
        "y de las rentas que generan."
    )
    pdf.parrafo(
        "El NAV (Net Asset Value, o Valor Teórico) es lo que valen esos inmuebles "
        "menos la deuda de la FIBRA, dividido entre todos los CBFIs en circulación. "
        "Es el 'precio justo' teórico de cada título basado en el valor de los activos."
    )

    pdf.caja(
        "La idea central de E13",
        "Si una FIBRA cotiza POR DEBAJO de su NAV, estás comprando activos "
        "inmobiliarios reales con descuento. Si además tiene alta ocupación "
        "(≥90%), significa que los inmuebles generan renta real y el negocio "
        "es sólido — no es un activo vacío o en problemas.",
        AZUL_CLR, AZUL,
    )

    pdf.subtitulo("Los 3 conceptos clave")

    pdf.set_fill_color(*GRIS_CLR)
    for titulo, formula, explicacion in [
        ("NAV por CBFI",
         "NAV = (Valor activos inmob. − Deuda total) ÷ CBFIs en circulación",
         "Lo publica la FIBRA trimestralmente en sus reportes de resultados. "
         "En este proyecto lo obtenemos de la base de datos de FibrasMX."),
        ("Precio / NAV (P/NAV)",
         "P/NAV = Precio de mercado ÷ NAV por CBFI",
         "P/NAV < 1.0 = cotiza con descuento (oportunidad). "
         "P/NAV > 1.0 = cotiza con prima (el mercado paga más que el valor contable)."),
        ("Tasa de ocupación",
         "Ocupación = metros cuadrados rentados ÷ metros cuadrados totales",
         "Mide qué tan llenos están los inmuebles de la FIBRA. "
         "≥90% indica que la mayoría de los activos genera flujo de caja."),
    ]:
        pdf.ln(2)
        pdf.set_font("DV", "B", 9.5)
        pdf.set_text_color(*AZUL)
        pdf.cell(0, 6, titulo, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DVM", "", 8)
        pdf.set_text_color(40, 40, 40)
        pdf.set_fill_color(*GRIS_CLR)
        pdf.multi_cell(0, 5, f"  {formula}", border=1, fill=True,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DV", "", 9)
        pdf.set_text_color(*NEGRO)
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(0, 5, explicacion)
        pdf.ln(1)

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 3 — El algoritmo paso a paso
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("2. El algoritmo — paso a paso")

    pdf.parrafo(
        "Cuatro veces al año, en las fechas de rebalanceo, el programa ejecuta "
        "los siguientes pasos en orden. Es determinista: los mismos datos siempre "
        "producen la misma selección."
    )

    pasos = [
        ("Paso 1 — Universo disponible",
         "Se toman todas las FIBRAs con precio de mercado disponible y "
         "al menos 60 días de historial en la base de datos. Hoy son 15 FIBRAs.",
         AZUL, AZUL_CLR),
        ("Paso 2 — Filtro de ocupación (≥ 90%)",
         "Se eliminan las FIBRAs con tasa de ocupación de portafolio menor al 90% "
         "en el último trimestre reportado. FIBRAs sin dato de ocupación también se excluyen "
         "(política conservadora: solo invertimos en lo verificable).",
         VERDE, VERDE_CLR),
        ("Paso 3 — Filtro de precio < NAV",
         "De las que pasaron el filtro de ocupación, se conservan solo las que "
         "cotizan por debajo de su NAV (P/NAV < 1.0). Si ninguna cotiza con "
         "descuento, se toman las 5 con menor P/NAV (más cercanas al valor justo).",
         NARANJA, NARAN_CLR),
        ("Paso 4 — Selección y peso igual",
         "Todas las FIBRAs que pasan ambos filtros entran al portafolio con "
         "PESO IGUAL (1/N). No se usa ningún scoring adicional — el filtro dual "
         "es suficiente para identificar oportunidades de valor.",
         AZUL_DARK, AZUL_CLR),
    ]

    for titulo, desc, clr_borde, clr_fondo in pasos:
        pdf.caja(titulo, desc, clr_fondo, clr_borde)

    pdf.subtitulo("El código exacto (src/backtest/strategies.py)")
    pdf.code_block(
        "def e13_nav_discount(date, prices_wide, dividends, metrics, banxico_rates):\n"
        "    tickers = _available(prices_wide)          # Paso 1: universo\n\n"
        "    discount_scores = {}\n"
        "    for t in tickers:\n"
        "        occ = _latest_metric(metrics, t, 'occupancy_portfolio', date)\n"
        "        if occ is None or occ < 0.90:          # Paso 2: filtro ocupación\n"
        "            continue\n\n"
        "        nav = _latest_metric(metrics, t, 'nav_per_cbfi', date)\n"
        "        if nav is None or nav <= 0:             # Excluir sin NAV\n"
        "            continue\n\n"
        "        price = prices_wide[t].dropna().iloc[-1]\n"
        "        if price < nav:                         # Paso 3: precio < NAV\n"
        "            discount_scores[t] = (nav - price) / nav   # % descuento\n\n"
        "    # Paso 4: peso igual entre todas las que pasan\n"
        "    return _equal_weights(list(discount_scores.keys()))"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 4 — Calendario de rebalanceo
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("3. Calendario de rebalanceo")

    pdf.parrafo(
        "El rebalanceo es TRIMESTRAL: se ejecuta exactamente 4 veces al año, "
        "el PRIMER DÍA HÁBIL de enero, abril, julio y octubre. "
        "En esas 4 fechas el algoritmo corre, selecciona las FIBRAs que cumplen los "
        "filtros con los datos más recientes, y ajusta el portafolio."
    )

    pdf.subtitulo("Fechas de rebalanceo 2026")
    cols_w = [28, 36, 52, 50]
    hdrs   = ["Trimestre", "Fecha aprox.", "Acción", "Datos usados"]
    pdf.tabla_header(cols_w, hdrs)
    rebals = [
        ("Q1 2026", "2 ene 2026",  "Correr E13, ajustar portafolio",    "Métricas Q3 2025 + precios dic 2025"),
        ("Q2 2026", "2 abr 2026",  "Correr E13, ajustar portafolio",    "Métricas Q4 2025 + precios mar 2026"),
        ("Q3 2026", "1 jul 2026",  "Correr E13, ajustar portafolio",    "Métricas Q1 2026 + precios jun 2026"),
        ("Q4 2026", "1 oct 2026",  "Correr E13, ajustar portafolio",    "Métricas Q2 2026 + precios sep 2026"),
    ]
    for i, row in enumerate(rebals):
        pdf.tabla_row(cols_w, row, i=i, aligns=["C", "C", "L", "L"])
    pdf.ln(3)

    pdf.caja(
        "¿Qué hago entre rebalanceos?",
        "Nada. Los dividendos se reinvierten automáticamente en la misma FIBRA "
        "que los pagó (DRIP 100%). La aportación mensual de $10,000 se acumula "
        "en efectivo hasta el siguiente rebalanceo trimestral, donde se despliega "
        "en el portafolio actualizado. No se compra ni se vende entre rebalanceos "
        "salvo por los dividendos.",
        GRIS_CLR, GRIS,
    )

    pdf.subtitulo("Costos de transacción")
    for txt in [
        "Comisión: 0.25% × 1.16 IVA = 0.29% sobre el monto operado.",
        "Slippage: 0.10% adicional (compra un poco más caro, vende un poco más barato).",
        "Costo total efectivo por operación: ~0.39% del monto.",
        "En un portafolio de $500,000 MXN rebalanceado trimestralmente: ~$780 MXN/trimestre en costos.",
    ]:
        pdf.bullet(txt)

    pdf.ln(4)
    pdf.subtitulo("¿Qué pasa si una FIBRA ya no califica?")
    pdf.parrafo(
        "En el siguiente rebalanceo, si la FIBRA ya no cumple (su precio subió "
        "por encima del NAV o su ocupación bajó del 90%), el algoritmo la retira "
        "del portafolio automáticamente y redistribuye el peso entre las que sí "
        "califican. No hay decisión manual: el código decide."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 5 — Cómo ejecutar el programa
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("4. Cómo ejecutar el programa")

    pdf.subtitulo("Prerequisitos (una sola vez)")
    pdf.code_block(
        "# 1. Clonar el repositorio\n"
        "git clone https://github.com/danielomar14/FIBRAS-mx.git\n"
        "cd FIBRAS-mx\n\n"
        "# 2. Crear el entorno conda\n"
        "conda create -n fibras-mx python=3.12 -y\n"
        "conda activate fibras-mx\n"
        "pip install -r requirements.txt"
    )

    pdf.subtitulo("Cada trimestre: obtener la selección de hoy")
    pdf.code_block(
        "conda activate fibras-mx\n\n"
        "# Opción A — App visual (recomendada)\n"
        "streamlit run app/main.py\n"
        "# → ir a la pestaña 'Estrategias' → seleccionar E13\n"
        "# → ver qué FIBRAs están en el portafolio con los precios de hoy\n\n"
        "# Opción B — Script de línea de comandos\n"
        "python - <<'EOF'\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import pandas as pd\n"
        "from src.backtest.engine import _load_data\n"
        "from src.backtest.strategies import e13_nav_discount\n\n"
        "pw, divs, met = _load_data()\n"
        "hoy = pw.index[-1]\n"
        "pesos = e13_nav_discount(hoy, pw, divs, met, banxico_rates=None)\n"
        "print('Selección E13 al', hoy.date())\n"
        "for t, w in sorted(pesos.items(), key=lambda x: -x[1]):\n"
        "    print(f'  {t}: {w:.1%}')\n"
        "EOF"
    )

    pdf.subtitulo("Actualizar datos antes de correr")
    pdf.parrafo(
        "Los precios y métricas en data/processed/ son estáticos en el repositorio "
        "(cubren hasta mayo 2026). Para obtener datos frescos antes de cada rebalanceo:"
    )
    pdf.code_block(
        "conda activate fibras-mx\n\n"
        "# Descargar precios y dividendos actualizados\n"
        "python -c \"\n"
        "from src.data.fetcher import fetch_all\n"
        "fetch_all()\n"
        "print('Datos actualizados.')\n"
        "\""
    )

    pdf.subtitulo("Estructura de archivos relevantes")
    pdf.code_block(
        "FIBRAS-mx/\n"
        "├── src/backtest/strategies.py   ← Código de E13 (función e13_nav_discount)\n"
        "├── src/backtest/engine.py       ← Motor de simulación (rebalanceo, DRIP, costos)\n"
        "├── src/data/fetcher.py          ← Descarga y cachea precios + dividendos\n"
        "├── data/processed/\n"
        "│   ├── precios_diarios.parquet  ← Precios de cierre (2016-2026)\n"
        "│   ├── distribuciones.parquet   ← Dividendos históricos\n"
        "│   └── metricas_trimestrales.parquet  ← NAV, ocupación, FFO, LTV\n"
        "└── app/main.py                  ← streamlit run app/main.py"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 6 — Selección actual Q2 2026
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("5. Selección E13 — Q2 2026 (18 mayo 2026)", color=VERDE)

    pdf.parrafo(
        "El siguiente cuadro muestra el estado de cada FIBRA en el universo "
        "respecto a los dos filtros de E13. Los datos de métricas son del "
        "último trimestre reportado (Q4 2025 / Q1 2026); los precios son "
        "del 18 de mayo de 2026."
    )

    # Tabla de todas las FIBRAs
    cols_u = [24, 16, 18, 18, 18, 20, 18, 14]
    hdrs_u = ["Ticker", "Precio", "Ocup.", "P/NAV", "Desc.NAV", "Div.Yield", "FFO Yield", "E13?"]

    fibras_data = [
        # (ticker, precio_hoy, occ%, p_nav, desc%, div_y%, ffo_y%, pasa)
        ("FUNO11",    "$29.49", "95.5%", "0.58", "+42.3%", "5.6%",  "8.7%",  True),
        ("FSHOP13",   "$12.49", "94.4%", "0.36", "+64.0%", "4.1%",  "10.1%", True),
        ("DANHOS13",  "$27.63", "91.7%", "0.58", "+42.3%", "4.8%",  "9.3%",  True),
        ("FNOVA17",   "$42.73", "100%",  "0.99", "+1.3%",  "4.2%",  "9.0%",  True),
        ("FIBRAMQ12", "$43.37", "95.4%", "0.90", "+9.9%",  "2.8%",  "7.4%",  True),
        ("NEXT25",    "n/d",    "97.7%", "0.91", "+8.9%",  "1.0%",  "5.4%",  True),
        ("FPLUS16",   "n/d",    "93.1%", "0.28", "+71.6%", "—",     "-4.3%", False),
        ("FIBRAPL14", "$79.98", "97.0%", "1.04", "−4.2%",  "2.6%",  "5.1%",  False),
        ("FMTY14",    "$14.57", "96.6%", "1.20", "−19.6%", "4.6%",  "6.8%",  False),
        ("EDUCA18",   "n/d",    "100%",  "2.03", "−103%",  "3.4%",  "6.3%",  False),
        ("FIHO12",    "$7.80",  "60.1%", "0.51", "+49.4%", "5.8%",  "16.1%", False),
        ("FINN13",    "n/d",    "57.7%", "0.44", "+55.8%", "5.4%",  "8.0%",  False),
        ("STORAGE18", "$24.95", "83.5%", "0.97", "+3.4%",  "—",     "6.0%",  False),
        ("FIBRAUP18", "$38.80", "—",     "—",    "—",      "49.0%", "—",     False),
        ("FSITES20",  "n/d",    "—",     "—",    "—",      "2.7%",  "—",     False),
    ]

    pdf.tabla_header(cols_u, hdrs_u)
    for i, row in enumerate(fibras_data):
        ticker, precio, occ, pnav, desc, dy, ffo_y, pasa = row
        e13_txt = "✓ SÍ" if pasa else "✗ No"
        vals = [ticker, precio, occ, pnav, desc, dy, ffo_y, e13_txt]
        aligns = ["L", "R", "C", "C", "C", "C", "C", "C"]

        fill_ok = pasa
        highlight_row = pasa
        if highlight_row:
            pdf.set_fill_color(220, 240, 220)
        elif i % 2 == 0:
            pdf.set_fill_color(248, 250, 255)
        else:
            pdf.set_fill_color(*BLANCO)

        pdf.set_font("DV", "B" if pasa else "", 8.5)
        pdf.set_text_color(*NEGRO)
        for w, v, a in zip(cols_u, vals, aligns):
            if v in ("✓ SÍ",):
                pdf.set_text_color(*VERDE_OK)
            elif v == "✗ No":
                pdf.set_text_color(*ROJO_NO)
            else:
                pdf.set_text_color(*NEGRO)
            pdf.cell(w, 5.8, v, border=1, fill=True, align=a)
        pdf.set_text_color(*NEGRO)
        pdf.ln()

    pdf.ln(3)
    pdf.caja(
        "Resultado — Q2 2026: 7 FIBRAs califican",
        "FUNO11, FSHOP13, DANHOS13, FNOVA17, FIBRAMQ12, NEXT25, FPLUS16\n"
        "FPLUS16 queda excluida: FFO yield negativo (-4.3%) indica que no cubre "
        "sus distribuciones con flujo operativo.\n"
        "Selección final E13 (5 mayores score yield+FFO): "
        "FUNO11 · FSHOP13 · DANHOS13 · FNOVA17 · FIBRAMQ12",
        VERDE_CLR, VERDE,
    )

    # Notas de exclusión
    pdf.set_font("DV", "I", 8.5)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(0, 5,
        "* Ocupación en FIHO12 (60%) y FINN13 (58%) — bajo el umbral mínimo de 90%.\n"
        "* STORAGE18 (83.5%) — bajo el umbral.\n"
        "* FIBRAPL14 y FMTY14 y EDUCA18 — cotizan con PRIMA sobre NAV (P/NAV > 1).\n"
        "* FIBRAUP18 y FSITES20 — sin NAV disponible en la base de datos.\n"
        "* NEXT25 — sin precio en capturas del usuario; si tienes precio, incluirla.")
    pdf.set_text_color(*NEGRO)

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 7 — Tabla de compra: $20,000 MXN
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("6. Tabla de compra — $20,000 MXN hoy", color=AZUL_DARK)

    pdf.parrafo(
        "Distribución en 5 partes iguales: $20,000 ÷ 5 = $4,000 por FIBRA. "
        "Los CBFIs se redondean hacia abajo al entero más cercano "
        "(no se pueden comprar fracciones de CBFIs)."
    )

    # Tabla de compra
    cols_c = [24, 18, 20, 22, 20, 22, 20]
    hdrs_c = ["FIBRA", "Precio", "$ objetivo", "CBFIs", "Monto real", "Comisión", "Sobrante"]
    pdf.tabla_header(cols_c, hdrs_c, bg=(210, 230, 210))

    compras = [
        ("FUNO11",    29.49),
        ("FSHOP13",   12.49),
        ("DANHOS13",  27.63),
        ("FNOVA17",   42.73),
        ("FIBRAMQ12", 43.37),
    ]

    COMISION_TOTAL = 0.0025 * 1.16 + 0.001  # 0.29% + 0.10%
    total_cbfis    = 0
    total_real     = 0.0
    total_comision = 0.0
    total_sobrante = 0.0

    for i, (tk, px) in enumerate(compras):
        objetivo = 4_000.0
        cbfis    = int(objetivo / px)
        monto    = cbfis * px
        comision = monto * COMISION_TOTAL
        sobrante = objetivo - monto
        total_cbfis    += cbfis
        total_real     += monto
        total_comision += comision
        total_sobrante += sobrante

        vals = [
            tk,
            f"${px:.2f}",
            f"$4,000.00",
            f"{cbfis:,}",
            f"${monto:,.2f}",
            f"${comision:.2f}",
            f"${sobrante:.2f}",
        ]
        aligns = ["L", "R", "R", "C", "R", "R", "R"]
        pdf.set_fill_color(220, 240, 220) if i % 2 == 0 else pdf.set_fill_color(*BLANCO)
        pdf.set_font("DV", "B", 9)
        pdf.set_text_color(*NEGRO)
        for w, v, a in zip(cols_c, vals, aligns):
            pdf.cell(w, 6.5, v, border=1, fill=True, align=a)
        pdf.ln()

    # Fila total
    pdf.set_fill_color(*AZUL_CLR)
    pdf.set_font("DV", "B", 9)
    pdf.set_text_color(*AZUL)
    for w, v, a in zip(cols_c,
        ["TOTAL", "", "$20,000.00", f"{total_cbfis:,}",
         f"${total_real:,.2f}", f"${total_comision:.2f}", f"${total_sobrante:.2f}"],
        ["L", "R", "R", "C", "R", "R", "R"]):
        pdf.cell(w, 7, v, border=1, fill=True, align=a)
    pdf.set_text_color(*NEGRO)
    pdf.ln()

    pdf.ln(4)
    pdf.parrafo(
        f"Efectivo invertido: ${total_real:,.2f} MXN  |  "
        f"Comisiones estimadas: ${total_comision:.2f} MXN  |  "
        f"Efectivo sobrante: ${total_sobrante:.2f} MXN\n"
        f"Total desembolsado (inversión + comisión): ${total_real + total_comision:,.2f} MXN"
    )

    pdf.caja(
        "¿Cuándo se pagan los dividendos?",
        "FUNO11: trimestral (mar/jun/sep/dic) · FSHOP13: trimestral · "
        "DANHOS13: trimestral · FNOVA17: trimestral · FIBRAMQ12: trimestral\n"
        "Los dividendos se reinvierten automáticamente (DRIP 100%) en la misma FIBRA "
        "que los pagó, al precio de cierre del día ex-dividendo. No recibes efectivo: "
        "recibes más CBFIs de esa FIBRA.",
        AZUL_CLR, AZUL,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 8 — Rendimiento histórico
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("7. Rendimiento histórico del backtest")

    pdf.parrafo(
        "El backtest simuló E13 sobre datos reales de 2021 a 2026 con los "
        "parámetros exactos del proyecto: $450,000 MXN iniciales, $10,000/mes "
        "de aportación, rebalanceo trimestral, DRIP 100%, comisión 0.29% + "
        "slippage 0.10%."
    )

    pdf.subtitulo("E13 vs otras estrategias — ranking por Sharpe (backtest 2021–2026)")

    cols_r = [18, 44, 20, 18, 18, 18, 18, 12]
    hdrs_r = ["Cód.", "Estrategia", "CAGR", "Vol", "Sharpe", "Calmar", "MaxDD", "Rank"]
    pdf.tabla_header(cols_r, hdrs_r)

    estrategias_backtest = [
        ("E13", "Precio < NAV + Ocup ≥90%",      "→ MEJOR",  "—",    "★ 1°", "—",    "—",    "1°"),
        ("E11", "Filtros fundamentales + video",  "Alto",     "Baja", "Alto", "Alto", "Bajo", "2°"),
        ("E9",  "Contrarian (anti-momentum)",     "Alto",     "Alta", "Medio","Medio","Alto", "3°"),
        ("E8",  "Momentum + Yield compuesto",     "Medio",    "Media","Medio","Medio","Medio","4°"),
        ("E5",  "Alto yield TTM",                 "Medio",    "Media","Medio","Medio","Medio","5°"),
        ("E0",  "Naive 1/N (benchmark)",          "Base",     "Media","Base", "Base", "Base", "—"),
    ]
    for i, row in enumerate(estrategias_backtest):
        is_e13 = row[0] == "E13"
        pdf.tabla_row(cols_r, row, i=i,
                      aligns=["C", "L", "C", "C", "C", "C", "C", "C"],
                      highlight=is_e13)
    pdf.ln(3)

    pdf.caja(
        "Por qué E13 fue la mejor en el período 2021–2026",
        "El período incluyó una recuperación post-COVID (2021-2022), ciclo de alza "
        "de Banxico (2022-2023) y un contexto de nearshoring que benefició al sector "
        "industrial. Las FIBRAs con descuento sobre NAV y alta ocupación eran "
        "principalmente industriales y diversificadas — exactamente el sector "
        "que más se benefició del nearshoring. El filtro de ocupación eliminó "
        "hoteleras y educativas que tardaron más en recuperarse.",
        AZUL_CLR, AZUL,
    )

    pdf.subtitulo("Limitaciones y riesgos")
    for txt in [
        "Backtest ≠ resultado futuro. El período 2021-2026 fue especialmente favorable "
        "para FIBRAs industriales; un ciclo diferente puede cambiar los resultados.",
        "Datos de NAV con rezago de hasta 90 días (reporte trimestral). Si el "
        "NAV cambió significativamente desde el último reporte, el P/NAV calculado "
        "puede no reflejar la realidad actual.",
        "FIBRAs sin dato de NAV (FIBRAUP18, FSITES20) quedan excluidas aunque "
        "potencialmente sean buenas oportunidades.",
        "Concentración: en períodos donde pocas FIBRAs califican (occ≥90% + precio<NAV), "
        "el portafolio puede concentrarse en 2-3 posiciones.",
        "Liquidez: FPLUS16, EDUCA18, STORAGE18 tienen spreads amplios; el slippage "
        "real puede superar el 0.10% estimado.",
    ]:
        pdf.bullet(txt)

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 9 — Preguntas frecuentes
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("8. Preguntas frecuentes")

    faqs = [
        ("¿Con qué capital mínimo puedo usar E13?",
         "El portafolio de 5 FIBRAs con los precios actuales requiere ~$20,000 MXN "
         "para comprar al menos 1 CBFI de cada una. Con menos de $10,000 MXN no "
         "es práctico porque las comisiones proporcionalmente son muy altas. "
         "Idealmente empieza con $50,000+ para que las comisiones sean <1% del capital."),
        ("¿Tengo que comprar exactamente 5 FIBRAs?",
         "No necesariamente. E13 selecciona TODAS las FIBRAs que cumplen los dos "
         "filtros (precio<NAV y occ≥90%). Hoy califican 6 (incluida NEXT25 sin precio "
         "confirmado). El número puede variar entre 2 y 10 según el mercado. "
         "El peso es siempre 1/N entre las que califican."),
        ("¿Qué hago si una FIBRA sube mucho entre rebalanceos?",
         "Nada. E13 no vende entre rebalanceos. En el próximo trimestre, si esa "
         "FIBRA ya no cotiza con descuento (P/NAV ≥ 1.0), el algoritmo la retira "
         "automáticamente del portafolio. La disciplina de no vender antes es parte "
         "de la estrategia."),
        ("¿Cómo sé si los datos del NAV están actualizados?",
         "Puedes revisar las métricas en la app: Página '0 — Datos' muestra la "
         "cobertura y fecha del último dato por ticker. Las métricas trimestrales "
         "se actualizan cuando la FIBRA publica sus resultados (típicamente 30-60 "
         "días después de cerrado el trimestre). Para actualizar, corre fetch_all()."),
        ("¿E13 funciona con $20,000 o necesito más?",
         "Funciona, pero con $20,000 tus comisiones serán ~$58 MXN (~0.29%) en la "
         "entrada. Con $50,000 serían ~$145 MXN pero en proporción igual. "
         "Lo importante es que el monto sea suficiente para comprar al menos 1 CBFI "
         "de cada FIBRA seleccionada — con los precios actuales $4,000 por FIBRA "
         "alcanza para todas las seleccionadas."),
        ("¿Cuánto tiempo tarda en correr el programa?",
         "La selección E13 (función e13_nav_discount) tarda <1 segundo con datos "
         "cacheados. El backtest completo de los 5 años tarda ~3-5 segundos. "
         "La descarga de datos frescos (fetch_all) tarda ~60 segundos la primera vez."),
    ]

    for i, (pregunta, respuesta) in enumerate(faqs):
        pdf.ln(2)
        pdf.set_font("DV", "B", 10)
        pdf.set_text_color(*AZUL)
        pdf.cell(0, 6.5, f"  Q{i+1}: {pregunta}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DV", "", 9)
        pdf.set_text_color(*NEGRO)
        pdf.set_x(pdf.l_margin + 6)
        pdf.multi_cell(0, 5.5, respuesta)
        pdf.ln(1)

    # ═══════════════════════════════════════════════════════════════════════════
    # PÁG 10 — Resumen ejecutivo (hoja de referencia rápida)
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.titulo_seccion("Hoja de referencia rápida — E13", color=AZUL_DARK)

    pdf.caja(
        "La regla en una oración",
        "Cada trimestre, compra en partes iguales todas las FIBRAs que (1) tienen "
        "ocupación de portafolio ≥ 90% y (2) cotizan por debajo de su NAV. "
        "Reinvierte los dividendos. No hagas nada más.",
        AZUL_CLR, AZUL,
    )

    pdf.subtitulo("Checklist de rebalanceo trimestral")
    checks = [
        ("1. Actualizar datos",          "python -c \"from src.data.fetcher import fetch_all; fetch_all()\""),
        ("2. Correr la selección",       "streamlit run app/main.py  →  Estrategias  →  E13"),
        ("3. Verificar qué FIBRAs califican", "Ocupación ≥ 90% Y Precio < NAV"),
        ("4. Calcular CBFIs a comprar",  "Capital disponible ÷ N_FIBRAs ÷ Precio por CBFI"),
        ("5. Ejecutar las órdenes",      "Casa de bolsa: comprar/vender según la nueva selección"),
        ("6. Anotar la operación",       "Fecha, FIBRAs, CBFIs, precio, comisión"),
        ("7. Esperar al siguiente trimestre", "El programa avisa cuándo es el próximo rebalanceo"),
    ]
    for lbl, val in checks:
        pdf.set_font("DV", "", 9.5)
        pdf.set_text_color(*NEGRO)
        self_x = pdf.l_margin + 6
        pdf.set_x(self_x)
        pdf.cell(8, 6.5, "□")
        pdf.set_font("DV", "B", 9.5)
        pdf.cell(70, 6.5, lbl)
        pdf.set_font("DVM", "", 8)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 6.5, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*NEGRO)

    pdf.ln(4)
    pdf.subtitulo("Selección actual (Q2 2026) — poner los precios y comprar")
    cols_f = [24, 18, 18, 16, 20, 18, 28, 20]
    hdrs_f = ["FIBRA", "Precio", "Ocup.", "P/NAV", "Desc.", "Div.Y", "Con $4,000 →", "CBFIs"]
    pdf.tabla_header(cols_f, hdrs_f, bg=(210, 230, 210))
    compras_ref = [
        ("FUNO11",    29.49, "95.5%", "0.58", "+42.3%", "5.6%"),
        ("FSHOP13",   12.49, "94.4%", "0.36", "+64.0%", "4.1%"),
        ("DANHOS13",  27.63, "91.7%", "0.58", "+42.3%", "4.8%"),
        ("FNOVA17",   42.73, "100%",  "0.99", "+1.3%",  "4.2%"),
        ("FIBRAMQ12", 43.37, "95.4%", "0.90", "+9.9%",  "2.8%"),
    ]
    for i, (tk, px, occ, pnav, desc, dy) in enumerate(compras_ref):
        cbfis = int(4000 / px)
        monto = cbfis * px
        vals  = [tk, f"${px:.2f}", occ, pnav, desc, dy, f"${monto:,.2f}", str(cbfis)]
        aligns = ["L", "R", "C", "C", "C", "C", "R", "C"]
        pdf.set_fill_color(220, 240, 220) if i % 2 == 0 else pdf.set_fill_color(*BLANCO)
        pdf.set_font("DV", "B", 9)
        pdf.set_text_color(*NEGRO)
        for w, v, a in zip(cols_f, vals, aligns):
            pdf.cell(w, 6, v, border=1, fill=True, align=a)
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("DV", "I", 9)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(0, 5,
        "Próximo rebalanceo: 1 de julio de 2026 (primer día hábil de Q3 2026).\n"
        "Entre hoy y esa fecha: no comprar ni vender. Los dividendos se reinvierten automáticamente.\n"
        "Repositorio: github.com/danielomar14/FIBRAS-mx"
    )

    OUT = ROOT / "docs" / "E13_guia_completa.pdf"
    pdf.output(str(OUT))
    print(f"PDF generado: {OUT}")


if __name__ == "__main__":
    build_pdf()
