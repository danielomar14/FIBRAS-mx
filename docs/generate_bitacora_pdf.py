"""
Genera docs/bitacora.pdf con fpdf2.
Refleja el contenido de docs/bitacora.tex (sesiones 1-6).
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_DIR = (
    "/Users/danielbecerrilolguin/anaconda3/envs/fibras-mx/lib/python3.12"
    "/site-packages/matplotlib/mpl-data/fonts/ttf"
)

AZUL      = (0, 82, 155)
AZUL_CLR  = (200, 215, 240)
VERDE     = (0, 100, 50)
VERDE_CLR = (200, 235, 215)
NARANJA   = (180, 80, 0)
GRIS      = (100, 100, 100)
BLANCO    = (255, 255, 255)
NEGRO     = (20, 20, 20)
GRIS_CLR  = (245, 245, 245)
LINEA     = (180, 180, 180)


class PDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(25, 20, 25)
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
        self.cell(0, 8, "Bitácora de Proyecto  ·  FIBRAs Mexicanas", align="L")
        self.set_x(-40)
        self.cell(30, 8, f"Pág. {self.page_no()}", align="R")
        self.ln(1)
        self.set_draw_color(*LINEA)
        self.line(25, self.get_y(), 185, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DV", "I", 8)
        self.set_text_color(*GRIS)
        self.cell(0, 10, "FIBRAS-mx  ·  Bitácora  ·  2026", align="C")

    def sesion(self, fecha, subtitulo):
        self.ln(4)
        self.set_fill_color(*AZUL)
        self.set_text_color(*BLANCO)
        self.set_font("DV", "B", 12)
        self.cell(0, 9, f"  Sesión del {fecha}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_fill_color(*AZUL_CLR)
        self.set_text_color(*AZUL)
        self.set_font("DV", "I", 9)
        self.cell(0, 6, f"  {subtitulo}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*NEGRO)
        self.ln(3)

    def subseccion(self, titulo):
        self.ln(4)
        self.set_text_color(*VERDE)
        self.set_font("DV", "B", 10)
        self.cell(0, 6, titulo, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*VERDE)
        self.line(25, self.get_y(), 185, self.get_y())
        self.set_text_color(*NEGRO)
        self.set_draw_color(*LINEA)
        self.ln(2)

    def parrafo(self, txt, size=9):
        self.set_font("DV", "", size)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5, txt)
        self.ln(1)

    def bullet(self, txt, nivel=0, bold=False, color=None):
        self.set_font("DV", "B" if bold else "", 9)
        self.set_text_color(*(color or NEGRO))
        indent = 6 + nivel * 8
        self.set_x(self.l_margin + indent)
        self.cell(5, 5, "•")
        self.multi_cell(0, 5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)

    def check(self, txt, done=True):
        self.set_font("DV", "", 9)
        self.set_text_color(*NEGRO)
        self.set_x(self.l_margin + 6)
        simbolo = "[✓]" if done else "[ ]"
        self.cell(10, 5, simbolo)
        self.multi_cell(0, 5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def code_block(self, txt):
        self.set_fill_color(*GRIS_CLR)
        self.set_draw_color(*LINEA)
        self.set_font("DVM", "", 7.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, txt, border=1, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)
        self.ln(2)

    def tabla_cobertura(self, filas):
        cols = [30, 28, 22, 24, 20, 22, 14]
        hdrs = ["Ticker", "Fuente precios", "Cobertura", "Dividendos", "Métricas", "Estado", ""]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 7.5)
        self.set_text_color(*AZUL)
        for w, h in zip(cols, hdrs):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        self.set_font("DV", "", 7.5)
        self.set_text_color(*NEGRO)
        for i, row in enumerate(filas):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            for w, val in zip(cols, row):
                self.cell(w, 5.5, val, border=1, fill=fill)
            self.ln()
        self.ln(3)

    def tabla_estrategias(self):
        cols = [16, 50, 94]
        hdrs = ["Cód.", "Nombre", "Señal principal"]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(cols, hdrs):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        rows = [
            ("E0",  "Naive 1/N (benchmark)",   "Igual peso en todas las FIBRAs disponibles"),
            ("E1",  "Large Cap",                "Peso proporcional a precio promedio 30 días"),
            ("E2",  "Momentum 12M",             "Top 5 por retorno trailing 252 días"),
            ("E3",  "Momentum 3M",              "Top 5 por retorno trailing 63 días"),
            ("E4",  "Baja volatilidad",         "Top 5 con menor vol realizada 252 días"),
            ("E5",  "Alto yield",               "Top 5 por dividendo TTM / precio"),
            ("E6",  "Calidad (ocupación)",      "Top 5 por tasa de ocupación trimestral"),
            ("E7",  "Valor (FFO yield)",        "Top 5 por FFO anualizado / precio"),
            ("E8",  "Momentum + Yield",         "Top 5 por score compuesto 50/50"),
            ("E9",  "Contrarian",               "Top 5 peores retornos 12M"),
            ("E10", "Rotación sectorial",       "Industrial/hotelero según dirección Banxico"),
            ("E11", "Filtros fundamentales",    "Ocupación ≥90%, LTV ≤40%, payout ≤80%; score yield+FFO"),
            ("E12", "Medias móviles",           "Golden Cross: precio > MA50 > MA200"),
            ("E13", "Precio < NAV",             "Ocupación ≥90% y precio < valor teórico"),
        ]
        self.set_font("DV", "", 7.5)
        self.set_text_color(*NEGRO)
        for i, (cod, nom, sen) in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            self.cell(cols[0], 5.5, cod, border=1, fill=fill)
            self.cell(cols[1], 5.5, nom, border=1, fill=fill)
            self.cell(cols[2], 5.5, sen, border=1, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def tabla_estado_final(self):
        cols = [38, 24, 20, 78]
        hdrs = ["Módulo", "Estado", "Tests", "Notas"]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(cols, hdrs):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        rows = [
            ("Pipeline de datos",   "Completo", "Manual", "Cobertura >80% en 10/15 FIBRAs"),
            ("Backtest E0-E13",      "Completo", "Visual", "Bug NaN corregido"),
            ("Feature registry",    "Completo", "Humo",   "400 features operativas"),
            ("ML Puro (2A)",         "Completo", "Visual", "Mejor: Logistic Regression"),
            ("GA (2B)",             "Completo", "Visual", "Converge a genes bloque A"),
            ("App Streamlit",       "Completo", "Manual", "6 páginas funcionales"),
            ("Reportes",            "Completo", "Visual", "CSV + PNG por estrategia"),
        ]
        self.set_font("DV", "", 8)
        self.set_text_color(*NEGRO)
        for i, row in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            for w, val in zip(cols, row):
                self.cell(w, 5.5, val, border=1, fill=fill)
            self.ln()
        self.ln(3)


def build_pdf():
    pdf = PDF()

    # ── PORTADA ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*AZUL)
    pdf.rect(0, 0, 210, 70, "F")
    pdf.set_y(12)
    pdf.set_text_color(*BLANCO)
    pdf.set_font("DV", "B", 28)
    pdf.cell(0, 14, "FIBRAs Mexicanas", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DV", "B", 14)
    pdf.cell(0, 9, "Backtest Comparativo y Optimización Genética", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_font("DV", "I", 10)
    pdf.cell(0, 6, "Bitácora de Proyecto", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(85)
    pdf.set_text_color(*NEGRO)
    for label, val in [
        ("Inicio",    "6 de mayo de 2026"),
        ("Capital",   "$450,000 MXN + $10,000/mes"),
        ("Universo",  "15 FIBRAs activas en BMV/BIVA"),
        ("Horizonte", "2021-2026"),
        ("Stack",     "Python 3.12 + Streamlit + DEAP + scikit-learn"),
        ("Repo",      "FIBRAS-mx/"),
    ]:
        pdf.set_font("DV", "B", 10)
        pdf.cell(42, 7, label + ":", border="B")
        pdf.set_font("DV", "", 10)
        pdf.cell(0, 7, val, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── OBJETIVOS GENERALES ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("DV", "B", 14)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 10, "Objetivos del Proyecto", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*AZUL)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(*NEGRO)

    objetivos = [
        ("Paso 0 — Datos",
         "Construir un pipeline robusto de adquisición y validación de datos históricos "
         "(precios diarios + distribuciones) para las 15 FIBRAs activas, con cobertura "
         "objetivo ≥80% por ticker."),
        ("Experimento 1",
         "Backtestear 14 estrategias de inversión (E0-E13) sobre el período 2021-2026 "
         "con capital inicial de $450,000 MXN, aportación mensual de $10,000, rebalanceo "
         "trimestral y reinversión DRIP al 100%."),
        ("Experimento 2A — ML Puro",
         "Entrenar 9 modelos de aprendizaje automático sobre un universo de hasta 400 "
         "features diarias; comparar en train (2017-2023), test (2024-2025) y holdout (2026)."),
        ("Experimento 2B — Algoritmo Genético",
         "Buscar en el espacio combinatorio de 40 variables × 10 parametrizaciones, "
         "eligiendo combinaciones de 1-5 genes, para descubrir qué regla de scoring "
         "maximiza el Sharpe del portafolio sin usar ningún modelo ML."),
        ("App Streamlit",
         "Exponer resultados y exploración interactiva en una aplicación multipágina "
         "accesible localmente."),
    ]
    for i, (titulo, texto) in enumerate(objetivos, 1):
        pdf.set_font("DV", "B", 10)
        pdf.set_text_color(*AZUL)
        pdf.cell(0, 7, f"  {i}. {titulo}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("DV", "", 9)
        pdf.set_text_color(*NEGRO)
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(0, 5, texto)
        pdf.ln(2)

    # ── UNIVERSO DE FIBRAs ────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_font("DV", "B", 12)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 8, "Universo de FIBRAs (15 activas, Q4 2025)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*AZUL)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(3)

    cols_f = [30, 42, 42, 32, 14]
    hdrs_f = ["Ticker", "Nombre", "Sector", "Ticker Yahoo", ""]
    fibras = [
        ("FUNO11",    "Fibra Uno",       "Diversificada",          "FUNO11.MX"),
        ("FIBRAPL14", "Fibra Prologis",  "Industrial/logística",   "FIBRAPL14.MX"),
        ("FIBRAMQ12", "Fibra Macquarie", "Industrial+comercial",   "FIBRAMQ12.MX"),
        ("DANHOS13",  "Fibra Danhos",    "Comercial+oficinas",     "DANHOS13.MX"),
        ("FMTY14",    "Fibra Monterrey", "Mixta",                  "FMTY14.MX"),
        ("FIHO12",    "Fibra Hotel",     "Hotelero",               "FIHO12.MX"),
        ("FINN13",    "Fibra Inn",       "Hotelero",               "FINN13.MX"),
        ("FSHOP13",   "Fibra Shop",      "Centros comerciales",    "FSHOP13.MX"),
        ("FIBRAUP18", "Fibra Upsite",    "Industrial PyMEs",       "FIBRAUP18.MX"),
        ("FNOVA17",   "Fibra Nova",      "Industrial+mixta",       "FNOVA17.MX"),
        ("FPLUS16",   "Fibra Plus",      "Diversificada pequeña",  "FPLUS16.MX"),
        ("STORAGE18", "Fibra Storage",   "Self-storage",           "STORAGE18.MX"),
        ("FSITES20",  "Fibra Sites",     "Infraestructura telco",  "FSITES20.MX"),
        ("EDUCA18",   "Fibra Educa",     "Educativo",              "EDUCA18.MX"),
        ("NEXT25",    "Fibra Next",      "Industrial (desde 2025)","NEXT25.MX"),
    ]
    pdf.set_fill_color(*AZUL_CLR)
    pdf.set_font("DV", "B", 8)
    pdf.set_text_color(*AZUL)
    for w, h in zip(cols_f[:-1], hdrs_f[:-1]):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("DV", "", 8)
    pdf.set_text_color(*NEGRO)
    for i, row in enumerate(fibras):
        fill = (i % 2 == 0)
        pdf.set_fill_color(248, 250, 255) if fill else pdf.set_fill_color(*BLANCO)
        for w, val in zip(cols_f[:-1], row):
            pdf.cell(w, 5.5, val, border=1, fill=fill)
        pdf.ln()

    # ── SESIÓN 1 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("6 de mayo de 2026", "Sesión inicial — Setup y estructura")
    pdf.subseccion("Objetivos")
    for t in ["Crear entorno conda fibras-mx (Python 3.12).",
              "Definir estructura de directorios del repositorio.",
              "Crear requirements.txt con todas las dependencias.",
              "Diseñar arquitectura del pipeline de datos (Paso 0)."]:
        pdf.bullet(t)

    pdf.subseccion("Acciones realizadas")
    pdf.parrafo("1. Entorno conda creado exitosamente:")
    pdf.code_block("conda create -n fibras-mx python=3.12 -y\nconda activate fibras-mx\npip install -r requirements.txt")
    pdf.parrafo("2. Estructura de directorios inicializada: data/{raw,processed,cache}, src/{data,backtest,genetic}, app/pages/, notebooks/, results/, docs/")
    pdf.parrafo("3. Fuentes de datos priorizadas (cascada): yfinance → Supabase FibrasMX → Banxico SIE/BMV → CSVs manuales.")
    pdf.parrafo("4. Arquitectura Streamlit diseñada: 3 páginas (datos, backtest, GA).")

    pdf.subseccion("Hallazgos")
    for t in ["NEXT25 solo existe desde 2025 — ajustar horizontes por ticker.",
              "Terrafina (TERRA13) absorbida por Fibra Prologis en 2024; excluida.",
              "Yahoo Finance: ajuste de dividendos inconsistente; se usará precio sin ajustar + dividendos separados.",
              "FIBRAs de baja liquidez (FPLUS16, EDUCA18, STORAGE18, FSITES20) pueden tener huecos."]:
        pdf.bullet(t)

    # ── SESIÓN 2 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("7 de mayo de 2026 (Sesión 2)", "Pipeline de datos completo + Página FAQ + AMEFIBRA")
    pdf.subseccion("Acciones realizadas")
    pdf.parrafo("Descubrimiento del cliente Supabase público de FibrasMX (clave sb_publishable_*). "
                "Tablas accesibles: fibra_dividends (537 registros desde 2011), "
                "fibra_premium_metrics (308 trimestres), fibras (16 FIBRAs).")
    pdf.parrafo("Pipeline multi-fuente: precios (yfinance → CSV manual), dividendos (Supabase → yfinance), "
                "métricas trimestrales (Supabase, 250 registros, 2013Q1-2025Q4).")
    pdf.parrafo("Benchmarks: IPC México (^MXX), CETES 28 días (tabla hardcoded 4.05%-11.52%), "
                "tasa Banxico 2018-2025, market caps desde API FibrasMX.")
    pdf.parrafo("Página FAQ (app/pages/1_faqs.py): 8 tabs analíticos (índice equiponderado, "
                "liquidez, FIBRAs vs CETES, Banxico, treemap sectorial, recortes de distribuciones, "
                "Amihud en estrés, COVID vs IPC).")
    pdf.parrafo("Scraper AMEFIBRA: 15 FIBRAs asociadas, 10 reportes PDF catalogados.")

    pdf.subseccion("Cobertura al cierre de la sesión")
    pdf.tabla_cobertura([
        ("FUNO11",    "yfinance", "96.4%", "42",  "Sí",   "OK",     ""),
        ("FIBRAPL14", "yfinance", "96.4%", "42",  "Sí",   "OK",     ""),
        ("FIBRAMQ12", "yfinance", "96.4%", "39",  "Sí",   "OK",     ""),
        ("DANHOS13",  "yfinance", "96.4%", "37",  "Sí",   "OK",     ""),
        ("FMTY14",    "yfinance", "96.4%", "73",  "Sí",   "OK",     "Mensual"),
        ("FIHO12",    "yfinance", "96.4%", "27",  "Sí",   "OK",     ""),
        ("FINN13",    "yfinance", "96.4%", "26",  "Sí",   "OK",     ""),
        ("FSHOP13",   "yfinance", "96.4%", "25",  "Sí",   "OK",     ""),
        ("FNOVA17",   "yfinance", "85.9%", "30",  "Sí",   "OK",     ""),
        ("EDUCA18",   "yfinance", "53.7%", "16",  "Parcial","Alerta",""),
        ("STORAGE18", "yfinance", "38.6%", "5",   "No",   "Alerta", ""),
        ("FSITES20",  "yfinance", "72.7%", "18",  "Parcial","Alerta",""),
        ("FIBRAUP18", "yfinance", "54.8%", "1",   "No",   "Alerta", ""),
        ("FPLUS16",   "yfinance", "96.4%", "1",   "No",   "Alerta", "Div."),
        ("NEXT25",    "yfinance", "33.7%", "1",   "No",   "Alerta", "Nueva"),
    ])

    # ── SESIÓN 3 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("7 de mayo de 2026 (Sesión 3)", "Motor de backtest, 14 estrategias E0-E13 y página Streamlit")
    pdf.subseccion("Motor de backtest (src/backtest/engine.py)")
    for t in ["Rebalanceo trimestral (primer día hábil de ene/abr/jul/oct).",
              "DRIP 100%: dividendos reinvertidos al precio de cierre del día ex-div.",
              "Comisión 0.25% × 1.16 IVA + slippage 0.10% sobre monto operado.",
              "Aportación mensual: primer día hábil de cada mes.",
              "pw_trade = pw.ffill(limit=5) para decisiones; pw_value = pw.ffill(limit=60) para valoración."]:
        pdf.bullet(t)

    pdf.subseccion("14 estrategias implementadas")
    pdf.tabla_estrategias()

    pdf.subseccion("Página Streamlit (app/pages/2_estrategias.py)")
    for t in ["Curvas de valor: equity curves + referencia CETES con aportaciones.",
              "Drawdown: serie de caída relativa, tabla MaxDD.",
              "Retornos anuales: heatmap + tabla coloreada.",
              "Riesgo vs retorno: scatter CAGR vs Vol, coloreado por Sharpe.",
              "Dividendos recibidos: barras mensuales 2024-2026."]:
        pdf.bullet(t)

    pdf.subseccion("Bug crítico corregido")
    pdf.parrafo("execute_rebalance liquidaba posiciones sin recuperar efectivo cuando el precio era NaN. "
                "Solución: parámetro prices_fallback con ffill(limit=60). "
                "El bug afectaba a todas las estrategias con STORAGE18; la corrección cambió "
                "el CAGR de E0 de 3.58% a 6.93%.")

    pdf.subseccion("Hallazgos")
    for t in ["E13 (precio < NAV) es la mejor estrategia del período 2021-2026.",
              "FMTY14 es la FIBRA con mayor frecuencia de distribución (mensual, 73 pagos desde IPO).",
              "E12 (Golden Cross) funcionó mal inicialmente por el bug del engine; corregido."]:
        pdf.bullet(t)

    # ── SESIÓN 4 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("7 de mayo de 2026 (Sesión 4)", "Diseño detallado del Experimento 2 — ML Puro y GA")
    pdf.subseccion("Aclaración arquitectural")
    pdf.parrafo("El Experimento 2 no es un sistema unificado; son dos enfoques independientes:")
    pdf.bullet("2A — ML Puro: alimentar 400 features a 9 modelos ML. "
               "Train 2017-2023, test 2024-2025, holdout 2026. Frecuencia diaria (~26,250 obs).", bold=True)
    pdf.bullet("2B — GA: buscar en el espacio de 400 genes posibles, eligiendo 1-5 genes. "
               "El cromosoma ES la estrategia (suma ponderada → top-k FIBRAs). Sin ML.", bold=True)

    pdf.subseccion("Encoding del cromosoma (Experimento 2B)")
    pdf.code_block(
        "Gene = namedtuple('Gene', ['var_id', 'param_idx'])\n"
        "# var_id   : 0-39  (40 variables)\n"
        "# param_idx: 0-9   (10 parametrizaciones)\n\n"
        "@dataclass\n"
        "class Individual:\n"
        "    genes:   list[Gene]  # 1-5 genes, sin repetir var_id\n"
        "    top_k:   int         # FIBRAs a seleccionar: {3,4,5,6,7}\n"
        "    fitness: float = -inf\n\n"
        "Fitness = Sharpe(train) - 0.02 * (n_genes - 1)"
    )
    pdf.parrafo("Parámetros del GA: 20 individuos, 50 generaciones, torneo k=3, elitismo top-2, "
                "early stopping 10 generaciones, re-diversificación si >80% de la población es idéntica.")
    pdf.parrafo("Plan detallado documentado en docs/plan_algoritmo.pdf (6 páginas).")

    # ── SESIÓN 5 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("8 de mayo de 2026 (Sesión 5)", "Implementación completa — Features, ML Puro y GA")
    pdf.subseccion("Feature registry (src/features/registry.py)")
    bloques = [
        ("A (15 vars)", "Retorno trailing, log-retorno, vol, precio/MA, Golden Cross, fuerza relativa, "
         "dividend yield TTM, P/NAV, ocupación, FFO yield, payout, LTV, cambio Banxico, precio 30d, sector."),
        ("B (5 vars)", "Fundamentales nuevas: NOI margin, deuda/activos, dilución CBFIs, apreciación propiedades, distribución/activos."),
        ("C (5 vars)", "Técnicas: RSI, Bollinger, z-score volumen, precio/52w-high, Amihud illiquidity."),
        ("D (5 vars)", "Tendencia: aceleración momentum, score multi-timeframe, consistencia, EMA/SMA, canal de precio."),
        ("E (5 vars)", "Mercado: fuerza relativa vs IPC, momentum sectorial, tendencia volumen, correlación IPC, tamaño relativo."),
        ("F (5 vars)", "Macro: tasa real (Banxico-inflación), spread yield-CETES, fase ciclo Banxico, USD/MXN, momentum IPC."),
    ]
    for bloque, desc in bloques:
        pdf.bullet(f"{bloque}: {desc}")

    pdf.subseccion("ML Puro (src/ml/)")
    for t in ["models.py: MODEL_REGISTRY con 9 modelos instanciables (Decision Tree, Ridge, "
              "Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, Extra Trees, ElasticNet).",
              "cross_val.py: walk_forward_cv() con ventana mín 252 días, paso 63 días; "
              "genera scores diarios → retornos de portafolio trimestrales.",
              "runner.py: run_all_models() corre los 9 modelos, cachea en results/ml_results.pkl."]:
        pdf.bullet(t)

    pdf.subseccion("Algoritmo Genético (src/genetic/)")
    for t in ["chromosome.py: Gene, Individual dataclasses.",
              "operators.py: tournament_select(), crossover() (unión de genes), mutate() "
              "con 4 sub-mutaciones independientes (p_m=0.3 cada una).",
              "fitness.py: evaluate_individual() → score por FIBRA → top-k → retorno trimestral → Sharpe.",
              "ga.py: loop principal con elitismo, early stopping, re-diversificación e immigration."]:
        pdf.bullet(t)

    pdf.subseccion("Hallazgos")
    for t in ["Logistic Regression resultó el mejor modelo ML en test; probable por universo pequeño (15 FIBRAs) y alta correlación entre features.",
              "GA converge consistentemente a genes del bloque A (retorno trailing + yield) con top_k=5.",
              "CatBoost genera catboost_info/ automáticamente; añadido a .gitignore.",
              "Primera corrida walk-forward CV tardaba ~8 min; optimizada con caché de feature matrix."]:
        pdf.bullet(t)

    # ── SESIÓN 6 ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.sesion("18 de mayo de 2026 (Sesión 6)", "Páginas de resultados y reportes + empaquetado")
    pdf.subseccion("Páginas Streamlit nuevas")
    for t in ["4_resultados.py: tabla comparativa unificada (E0-E13 + GA + mejor ML), animación de "
              "convergencia del GA por generación (Plotly interactivo).",
              "5_reportes.py: CSVs diarios de valor total y % por FIBRA, gráficas con anotaciones, "
              "botón de descarga y generación vía export_results.py."]:
        pdf.bullet(t)

    pdf.subseccion("Refinamientos")
    for t in ["engine.py: manejo mejorado de NaN en liquidaciones, compatibilidad con BACKTEST_END dinámico.",
              "fitness.py + ga.py: registro de historial por generación (mejor fitness, media, diversidad genómica) para la animación.",
              "cross_val.py + runner.py: corrección de solapamiento entre folds de walk-forward; soporte para feature_selection por bloque.",
              "README.md: documentación completa del repositorio.",
              "bitacora.pdf: generada con fpdf2."]:
        pdf.bullet(t)

    pdf.subseccion("Estado del proyecto al 18 de mayo de 2026")
    pdf.tabla_estado_final()

    pdf.subseccion("Pendientes futuros")
    for t, done in [
        ("Validación anti-overfitting GA: walk-forward de 5 ventanas.", False),
        ("Token Banxico SIE para CETES oficiales en tiempo real.", False),
        ("Parseo de PDFs AMEFIBRA con pdfplumber para métricas faltantes.", False),
        ("Deploy público en Streamlit Cloud (requiere credenciales Supabase).", False),
    ]:
        pdf.check(t, done)

    out = "/Users/danielbecerrilolguin/FIBRAS-mx/docs/bitacora.pdf"
    pdf.output(out)
    print(f"PDF generado: {out}")


if __name__ == "__main__":
    build_pdf()
