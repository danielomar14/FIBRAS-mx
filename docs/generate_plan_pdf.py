"""
Genera plan_algoritmo.pdf con fpdf2 + fuentes DejaVu (Unicode completo).
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_DIR = (
    "/Users/danielbecerrilolguin/anaconda3/envs/fibras-mx/lib/python3.12"
    "/site-packages/matplotlib/mpl-data/fonts/ttf"
)

AZUL     = (30, 80, 160)
AZUL_CLR = (200, 215, 240)
GRIS     = (80, 80, 80)
BLANCO   = (255, 255, 255)
NEGRO    = (20, 20, 20)
GRIS_CLR = (245, 245, 245)
LINEA    = (180, 180, 180)


class PDF(FPDF):
    def __init__(self):
        super().__init__(format="Letter")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 20, 20)
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
        self.cell(0, 8, "FIBRAS-mx  ·  Plan: Página Algoritmo — ML + GA", align="L")
        self.set_x(-50)
        self.cell(30, 8, f"Pág. {self.page_no()}", align="R")
        self.ln(2)
        self.set_draw_color(*LINEA)
        self.line(20, self.get_y(), 195, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DV", "I", 8)
        self.set_text_color(*GRIS)
        self.cell(0, 10, "FIBRAS-mx  ·  Experimento 2  ·  Mayo 2026", align="C")

    def titulo_seccion(self, txt):
        self.ln(6)
        self.set_fill_color(*AZUL)
        self.set_text_color(*BLANCO)
        self.set_font("DV", "B", 12)
        self.cell(0, 8, f"  {txt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*NEGRO)
        self.ln(3)

    def subtitulo(self, txt):
        self.ln(4)
        self.set_text_color(*AZUL)
        self.set_font("DV", "B", 10)
        self.cell(0, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)
        self.ln(1)

    def parrafo(self, txt, size=9):
        self.set_font("DV", "", size)
        self.set_text_color(*NEGRO)
        self.multi_cell(0, 5, txt)
        self.ln(1)

    def bullet(self, txt, indent=8):
        self.set_font("DV", "", 9)
        self.set_x(self.l_margin + indent)
        self.cell(5, 5, "•")
        self.multi_cell(0, 5, txt)

    def code_block(self, txt):
        self.set_fill_color(*GRIS_CLR)
        self.set_draw_color(*LINEA)
        self.set_font("DVM", "", 7.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, txt, border=1, fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*NEGRO)
        self.ln(2)

    def tabla_features(self, filas):
        col_w = [12, 50, 108]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 7.5)
        self.set_text_color(*AZUL)
        for w, h in zip(col_w, ["ID", "Variable", "10 Parametrizaciones"]):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        self.set_font("DV", "", 7)
        self.set_text_color(*NEGRO)
        for i, (fid, var, params) in enumerate(filas):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            y0 = self.get_y()
            self.cell(col_w[0], 5, fid, border="LTB", fill=fill)
            self.cell(col_w[1], 5, var, border="TB", fill=fill)
            self.multi_cell(col_w[2], 4.5, params, border="RTB", fill=fill,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if self.get_y() < y0 + 5:
                self.set_y(y0 + 5)
        self.ln(2)

    def tabla_modelos(self, filas):
        col_w = [8, 42, 120]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(col_w, ["#", "Modelo", "Hiperparámetros clave"]):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        self.set_font("DV", "", 8)
        self.set_text_color(*NEGRO)
        for i, row in enumerate(filas):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            self.cell(col_w[0], 5.5, row[0], border="LTB", fill=fill)
            self.cell(col_w[1], 5.5, row[1], border="TB", fill=fill)
            self.multi_cell(col_w[2], 5, row[2], border="RTB", fill=fill,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def tabla_splits(self):
        col_w = [28, 42, 28, 44, 28]
        headers = ["Split", "Período", "Días hábiles", "Obs. (días×15)", "Propósito"]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(col_w, headers):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        rows = [
            ("Train",      "2017-01 → 2023-12", "~1,750", "~26,250", "GA + ML"),
            ("Test",       "2024-01 → 2025-12", "~500",   "~7,500",  "Evaluacion GA"),
            ("Validation", "2026-01 → hoy",     "~90",    "~1,350",  "Holdout real"),
        ]
        self.set_font("DV", "", 8)
        self.set_text_color(*NEGRO)
        for i, row in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(*BLANCO)
            for w, cell in zip(col_w, row):
                self.cell(w, 5.5, cell, border=1, fill=fill)
            self.ln()
        self.ln(3)


def build_pdf():
    pdf = PDF()

    # ── PORTADA ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*AZUL)
    pdf.rect(0, 0, 216, 65, "F")
    pdf.set_y(12)
    pdf.set_text_color(*BLANCO)
    pdf.set_font("DV", "B", 24)
    pdf.cell(0, 12, "FIBRAS-mx", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DV", "B", 14)
    pdf.cell(0, 8, "Experimento 2: Algoritmo", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("DV", "", 11)
    pdf.cell(0, 7, "ML + Algoritmo Genético para Selección Óptima de FIBRAs",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(80)
    pdf.set_text_color(*NEGRO)
    info = [
        ("Autor",  "Daniel Becerril Olguín"),
        ("Fecha",  "Mayo 2026"),
        ("Stack",  "Python 3.12 · Streamlit · scikit-learn · XGBoost · CatBoost · LightGBM"),
        ("Datos",  "2017–2026 · 15 FIBRAs · Frecuencia diaria"),
        ("Páginas", "0_datos, 1_faqs, 2_estrategias, 3_algoritmo"),
    ]
    for label, val in info:
        pdf.set_font("DV", "B", 10)
        pdf.cell(50, 7, label + ":", border="B")
        pdf.set_font("DV", "", 10)
        pdf.cell(0, 7, val, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    # Resumen ejecutivo
    pdf.set_fill_color(*AZUL_CLR)
    pdf.set_font("DV", "B", 10)
    pdf.set_text_color(*AZUL)
    pdf.cell(0, 7, "  Resumen ejecutivo", border=1, fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*NEGRO)
    pdf.set_font("DV", "", 9)
    pdf.multi_cell(0, 5,
        "En lugar de codificar reglas de inversión manualmente (como E11 u E13), "
        "el Experimento 2 usa un Algoritmo Genético para descubrir qué combinación "
        "de features, modelo de ML e hiperparámetros maximiza el Sharpe del portafolio "
        "en 2017–2023, verificando que el resultado sea estadísticamente significativo "
        "en test (2024–2025) y holdout (2026).\n\n"
        "El universo de búsqueda son 400 features (40 variables × 10 parametrizaciones) "
        "en 6 bloques: derivadas de estrategias existentes (A), fundamentales (B), "
        "técnicas (C), tendencia (D), mercado (E) y macro (F). El cromosoma incluye "
        "1–5 genes, tipo de modelo, hiperparámetros, horizonte de predicción H y top-k.",
        border=1)

    # ── DATOS ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.titulo_seccion("1. Frecuencia de observaciones y partición de datos")

    pdf.subtitulo("Frecuencia: DIARIA — no solo trimestral")
    pdf.parrafo(
        "El dataset usa observaciones DIARIAS: ~26,250 filas en train "
        "(vs ~420 si usáramos solo rebalanceos trimestrales)."
    )
    for b in [
        "Precios OHLCV: diarios → features técnicas calculadas día a día.",
        "Fundamentales trimestrales: forward-fill diario entre reportes. "
          "La ocupación del Q2 vale hacia adelante hasta que llega el Q3.",
        "Label: retorno futuro a horizonte H días. "
          "H ∈ {1, 5, 10, 21, 42, 63, 126} — el GA descubre qué horizonte predice mejor.",
    ]:
        pdf.bullet(b)
    pdf.ln(4)

    pdf.subtitulo("Partición train / test / validation")
    pdf.tabla_splits()

    pdf.subtitulo("Walk-forward CV dentro del train")
    pdf.parrafo(
        "Ventana expansiva, mínimo 252 días de historia, paso de 63 días (trimestral). "
        "Entrena en días ≤ t, predice días t+1 a t+H."
    )

    pdf.subtitulo("Conexión con el rebalanceo trimestral")
    for b in [
        "Modelo entrenado con observaciones diarias.",
        "En cada fecha de rebalanceo Q, se predicen scores para los próximos H días.",
        "Se promedian los scores → ranking de FIBRAs → top-k elegidas.",
        "La ejecución sigue siendo trimestral (igual que E0–E13).",
    ]:
        pdf.bullet(b)

    # ── FEATURES BLOQUE A ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.titulo_seccion("2. Universo de features: 40 variables × 10 params = 400 features")

    pdf.subtitulo("Bloque A — Variables de estrategias existentes (E0–E13)")
    pdf.tabla_features([
        ("A01", "Retorno trailing",         "21d, 42d, 63d, 84d, 126d, 168d, 252d, 378d, 504d, 756d"),
        ("A02", "Log-retorno trailing",     "Mismos 10 horizontes que A01"),
        ("A03", "Volatilidad realizada",    "21d, 42d, 63d, 84d, 126d, 168d, 252d, 378d, 504d, ratio(21/252)"),
        ("A04", "Precio / MAN",             "MA20, MA30, MA50, MA60, MA100, MA120, MA150, MA200, MA250, MA300"),
        ("A05", "MAN > MAM (Golden Cross)", "(20>50), (20>100), (50>100), (50>200), (20>200), (100>200), (30>100), (30>200), (60>150), (10>50)"),
        ("A06", "Fuerza relativa vs MA",    "Distancia % a MA50, MA100, MA200, sus negativos/invertidos + log"),
        ("A07", "Dividend yield TTM",       "raw, rank, >3%, >4%, >5%, >6%, >7%, >8%, >9%, top25%"),
        ("A08", "Precio / NAV",             "raw, rank, <0.80, <0.90, <0.95, <1.0, <1.05, <1.10, descuento%, rank_desc"),
        ("A09", "Ocupación",                "raw, rank, ≥0.75, ≥0.80, ≥0.85, ≥0.87, ≥0.90, ≥0.92, ≥0.95, QoQ-change"),
        ("A10", "FFO yield",                "raw, rank, >4%, >5%, >6%, >7%, >8%, >9%, >10%, anualizado×4"),
        ("A11", "Payout ratio (dist/FFO)",  "raw, rank, <0.60, <0.70, <0.75, <0.80, <0.90, <1.0, QoQ-change, inverted"),
        ("A12", "LTV ratio",                "raw, rank, <0.30, <0.35, <0.40, <0.45, <0.50, QoQ-change, inverted, >0.50"),
        ("A13", "Cambio Banxico 6M",        "raw, >+50bps, >+100bps, <-50bps, <-100bps, |abs|>50bps, dirección, aceleración, rolling12M, zscore"),
        ("A14", "Precio 30d mean",          "raw, rank, rank_top3, rank_top5, rank_top7, top50%, percentil, z-score, log, norm"),
        ("A15", "Sector dummy",             "industrial=1, hoteles=1, comercial=1, industrial+comercial, diversificado=1, no_hoteles=1, ponderado, mixto=1, 4 dummies"),
    ])

    pdf.subtitulo("Bloque B — Variables Fundamentales (nuevas)")
    pdf.tabla_features([
        ("B01", "NOI margin",              "raw, rank, >30%, >40%, >50%, >60%, >65%, QoQ-change, YoY-change, trend4Q"),
        ("B02", "Deuda / Activos",         "raw, rank, <0.35, <0.40, <0.45, <0.50, <0.55, QoQ-change, inverted, trend4Q"),
        ("B03", "Dilución de CBFIs",       "%change YoY, positivo, negativo, abs change, <1%, <3%, <5%, <-1%, <-3%, z-score"),
        ("B04", "Apreciación propiedades", "%change YoY en appraised_value, rank, >0%, >2%, >5%, >8%, <0%, trend4Q, QoQ-change, zscore"),
        ("B05", "Distribución / Activos",  "dist×cbfis/total_assets, rank, >1%, >2%, >3%, >4%, >5%, trend4Q, QoQ-change, z-score"),
    ])

    pdf.subtitulo("Bloque C — Variables Técnicas (nuevas)")
    pdf.tabla_features([
        ("C01", "RSI",                   "RSI-7, RSI-9, RSI-14, RSI-21, RSI-28, RSI-42, RSI-63, RSI<30, RSI>70, distancia-de-50"),
        ("C02", "Posición Bollinger",    "BB(20,2), BB(20,1.5), BB(50,2), %B(0-1), %B<0.2, %B>0.8, BB-width, BB-width-z, squeeze, inv-%B"),
        ("C03", "Z-score de volumen",    "z(21d), z(63d), z(126d), z(252d), >+1σ, >+2σ, <-1σ, tendencia, ratio(21/63), ratio(5/21)"),
        ("C04", "Precio / 52-week high", "raw, >0.90, >0.95, >0.98, <0.80, <0.70, dist%, drawdown-52w, nuevo-máx(1/0), recuperación"),
        ("C05", "Amihud Illiquidity",    "ratio(21d), ratio(63d), ratio(252d), rank_liq, rank_iliq, thresh_high, thresh_low, trend4Q, zscore, vs-sector"),
    ])

    pdf.add_page()

    pdf.subtitulo("Bloque D — Variables de Tendencia (nuevas)")
    pdf.tabla_features([
        ("D01", "Aceleración momentum",  "ret63-ret126, ret21-ret63, ret126-ret252, ret63/ret126, señal_pos, zscore, rank, 2da-deriv, negativa, norm"),
        ("D02", "Score multi-timeframe", "0.25×ret21+0.25×ret63+0.25×ret126+0.25×ret252 + 8 combos de ponderaciones distintas + top25%"),
        ("D03", "Consistencia retornos", "%días pos 21d, 63d, 126d, 252d, >50%, >60%, >70%, tendencia-mejora, zscore, rank"),
        ("D04", "EMA vs SMA (suavidad)", "EMA21/SMA21, EMA50/SMA50, EMA vs SMA, >1(sube), <1(baja), distancia, ratio-cruzado, señal, z-score, rank"),
        ("D05", "Canal de precio",       "%channel(20d), %channel(63d), %channel(126d), >80%, >90%, <20%, <10%, breakout_up, breakout_dn, rango-norm"),
    ])

    pdf.subtitulo("Bloque E — Variables de Mercado (nuevas)")
    pdf.tabla_features([
        ("E01", "Fuerza relativa vs índice", "ret - ret_index (21d, 63d, 126d, 252d), rank, >0%, >5%, >10%, zscore, rolling_rank, tendencia"),
        ("E02", "Momentum sectorial",         "avg_ret_peers (21d, 63d, 126d), fibra_vs_sector, sector_rank, top2, bottom2, spread, zscore, rank, tendencia"),
        ("E03", "Tendencia de volumen",       "slope(21d), slope(63d), slope(126d), >0%, >+20%, >+50%, ratio(5/15d), zscore, señal-pos, rank"),
        ("E04", "Correlación con IPC",        "corr_63d, corr_126d, corr_252d, <0.3, >0.7, beta_63d, beta_126d, beta_252d, tracking-error, corr-inversa"),
        ("E05", "Tamaño relativo",            "rank(top5), rank(top3), decil, tercil, log_price30d, cambio_rank_YoY, stable_large, concentración, peso_eq, peso_prop"),
    ])

    pdf.subtitulo("Bloque F — Variables Macro (nuevas)")
    pdf.tabla_features([
        ("F01", "Tasa real (Banxico-inf)", "tasa-CPI_proxy, >5%, >6%, >7%, >8%, <4%, cambio_rate_real, zscore, rank, tendencia"),
        ("F02", "Spread yield-CETES",      "div_yield - CETES_28d, >0%, >1%, >2%, >3%, >4%, zscore, rank, tendencia-spread, spread_negativo"),
        ("F03", "Fase ciclo de tasas",     "subiendo(1/0), bajando(1/0), pausa(1/0), acelerando-suba, desacelerando-suba, cum_change_6M, velocidad, zscore, señal, dirección"),
        ("F04", "Tendencia USD/MXN",       "ret_usdmxn (21d, 63d, 126d, 252d), >0% peso-débil, <0% peso-fuerte, >+5%, <-5%, vol, corr-FIBRA, zscore"),
        ("F05", "Momentum IPC",            "ret_ipc (21d, 63d, 126d, 252d), >0%, >5%, >10%, <0%, zscore, rank_hist, tendencia, IPC>MA200(1/0), breadth_signal"),
    ])

    # ── MODELOS ML ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.titulo_seccion("3. Modelos de Machine Learning")
    pdf.tabla_modelos([
        ("1", "Decision Tree",       "max_depth [2,3,4,5,6], min_samples_split [2,5,10,20]"),
        ("2", "Ridge Regression",    "alpha [0.01, 0.1, 1, 10, 100]"),
        ("3", "Logistic Regression", "C [0.01, 0.1, 1, 10], solver [lbfgs, liblinear]"),
        ("4", "Random Forest",       "n_estimators [50,100,200], max_depth [2,3,4,5,None]"),
        ("5", "XGBoost",             "n_estimators [50,100,200], max_depth [2,3,4], lr [0.01,0.1,0.3]"),
        ("6", "CatBoost",            "iterations [50,100,200], depth [2,3,4], lr [0.01,0.05,0.1]"),
        ("7", "LightGBM",            "n_estimators [50,100,200], max_depth [2,3,4], lr [0.01,0.1]"),
        ("8", "Extra Trees",         "n_estimators [50,100,200], max_depth [2,3,4,None]"),
        ("9", "ElasticNet",          "alpha [0.01,0.1,1], l1_ratio [0.1,0.5,0.9]"),
    ])

    pdf.subtitulo("Tarea ML: ranking con horizonte configurable")
    for b in [
        "El modelo predice un score continuo para cada FIBRA en cada día.",
        "Label: retorno a H días hacia adelante, normalizado 0–1 con rank entre las 15 FIBRAs.",
        "H ∈ {1, 5, 10, 21, 42, 63, 126} — parte del cromosoma GA.",
        "En rebalanceo: promedio de scores últimos 63 días → top-k seleccionadas.",
        "top_k ∈ {3, 4, 5, 6, 7} — también optimizable por el GA.",
    ]:
        pdf.bullet(b)

    # ── CROMOSOMA ─────────────────────────────────────────────────────────────
    pdf.titulo_seccion("4. Encoding del cromosoma (GA)")
    pdf.code_block(
        "Gene = namedtuple('Gene', ['var_id', 'param_idx'])\n"
        "# var_id   : 0–39  (40 variables)\n"
        "# param_idx: 0–9   (10 parametrizaciones)\n"
        "# Total espacio de búsqueda: 400 genes posibles\n\n"
        "@dataclass\n"
        "class Individual:\n"
        "    genes:        list[Gene]   # 1–5 genes, sin repetir var_id\n"
        "    model_type:   str          # uno de los 9 modelos\n"
        "    model_params: dict         # hiperparámetros del modelo\n"
        "    horizon:      int          # H días: in {1,5,10,21,42,63,126}\n"
        "    top_k:        int          # FIBRAs a seleccionar: in {3,4,5,6,7}\n"
        "    fitness:      float = -inf"
    )

    # ── ALGORITMO GENETICO ────────────────────────────────────────────────────
    pdf.titulo_seccion("5. Algoritmo Genético")

    pdf.subtitulo("Inicialización (20 individuos)")
    pdf.code_block(
        "n_genes    = random.choice([1, 2, 3, 4, 5])\n"
        "genes      = random.sample(ALL_400_GENES, n_genes)  # sin repetir var_id\n"
        "model_type = random.choice(MODELOS)  # 9 opciones\n"
        "params     = random.choice(PARAM_GRID[model_type])\n"
        "horizon    = random.choice([1, 5, 10, 21, 42, 63, 126])\n"
        "top_k      = random.choice([3, 4, 5, 6, 7])"
    )

    pdf.subtitulo("Función de Fitness")
    pdf.code_block(
        "fitness(ind):\n"
        "  1. Construir feature matrix DIARIA desde genes del individuo:\n"
        "       - Features técnicas: día a día desde precios OHLCV\n"
        "       - Features fundamentales: forward-fill de datos trimestrales\n"
        "       - Label: retorno a ind.horizon días (rank 0–1 entre FIBRAs)\n\n"
        "  2. Walk-forward CV (ventana mín 252 días, paso 63 días):\n"
        "       Para cada fold t en [2017+252d … 2023-12-31]:\n"
        "         - Entrenar model(params) en días ≤ t\n"
        "         - Predecir scores diarios para [t+1 … t+ind.horizon]\n"
        "         - Agregar scores por FIBRA (promedio)\n"
        "         - Seleccionar top_k FIBRAs\n"
        "         - Calcular retorno portfolio (igual peso)\n\n"
        "  3. Calcular Sharpe de la serie de retornos de portfolio\n"
        "  4. Parsimony penalty: -0.02 × (n_genes - 1)\n"
        "  5. Retornar Sharpe ajustado"
    )

    pdf.subtitulo("Operadores evolutivos")
    pdf.parrafo("Selección: Torneo de tamaño 3. Elitismo: top-2 pasan directamente.")
    pdf.code_block(
        "Cruzamiento:\n"
        "  gene_pool    = union(parent1.genes, parent2.genes)\n"
        "  n_child      = random.choice([1…5])\n"
        "  child_genes  = random.sample(gene_pool, min(n_child, len(pool)))\n"
        "  child_model  = random.choice([p1.model_type, p2.model_type])\n"
        "  child_params = mezcla de parámetros de ambos padres (por key)\n"
        "  child_horizon = random.choice([p1.horizon, p2.horizon])\n"
        "  child_top_k   = random.choice([p1.top_k, p2.top_k])"
    )
    pdf.parrafo("Mutación (P=0.3 por tipo, independientes):")
    for b in [
        "mutate_add:    agregar un gene aleatorio (si n < 5)",
        "mutate_remove: eliminar un gene aleatorio (si n > 1)",
        "mutate_swap:   reemplazar un gene por uno nuevo (distinto var_id)",
        "mutate_param:  cambiar param_idx de un gene existente",
        "mutate_model:  cambiar model_type y/o model_params (P=0.2)",
        "mutate_horizon: cambiar horizonte H (P=0.2)",
    ]:
        pdf.bullet(b)

    pdf.subtitulo("Criterios de parada")
    for b in [
        "max_generations = 50  (configurable en UI, slider 10–100)",
        "Early stopping: sin mejora en el mejor fitness por 10 generaciones",
        "Re-diversificación: si >80% de la población es idéntica, reiniciar 40% (immigration)",
    ]:
        pdf.bullet(b)

    # ── ESTRUCTURA Y STREAMLIT ────────────────────────────────────────────────
    pdf.add_page()
    pdf.titulo_seccion("6. Estructura de archivos")
    pdf.code_block(
        "src/\n"
        "  features/\n"
        "    registry.py     # FEATURE_REGISTRY: dict[int, FeatureDef]\n"
        "    builder.py      # build_feature_matrix(genes, ...) -> DataFrame\n"
        "  genetic/\n"
        "    chromosome.py   # Gene, Individual dataclasses\n"
        "    operators.py    # crossover(), mutate(), tournament_select()\n"
        "    fitness.py      # evaluate_individual(ind, ...) -> float\n"
        "    ga.py           # run_ga(config) -> GAResult\n"
        "  ml/\n"
        "    models.py       # MODEL_REGISTRY: instanciar modelo por tipo+params\n"
        "    cross_val.py    # walk_forward_cv(...) -> List[float]\n"
        "app/pages/\n"
        "  3_algoritmo.py    # Streamlit página completa\n"
        "results/\n"
        "  ga_results.pkl    # Cache del resultado del GA"
    )

    pdf.titulo_seccion("7. Página Streamlit (3_algoritmo.py)")
    pdf.subtitulo("Sección 1 — Panel de configuración")
    for b in [
        "Generaciones: slider 10–100 (default 50)",
        "Tamaño de población: slider 10–50 (default 20)",
        "Modelos a incluir: multiselect (todos por default)",
        "k FIBRAs a seleccionar: slider 3–7 (default 5)",
        "Botón [▶ Correr GA] con spinner de progreso por generación",
        "Botón [Cargar último resultado] desde results/ga_results.pkl",
    ]:
        pdf.bullet(b)

    pdf.subtitulo("Sección 2 — Resultados del GA (4 tabs)")
    for tab_name, desc in [
        ("Tab 1 — Convergencia",
         "Gráfica fitness del mejor individuo y media poblacional vs. generación. "
         "Tabla top-5 individuos de la última generación."),
        ("Tab 2 — Mejor cromosoma",
         "Tabla de features seleccionadas: bloque, variable, parametrización. "
         "Modelo ML + hiperparámetros. Feature importance si disponible (RF, XGB, LGB)."),
        ("Tab 3 — Equity curve",
         "3 líneas: Train (2017–2023), Test (2024–2025), Validation (2026). "
         "Comparativa vs E0 y E13. Métricas: CAGR, Sharpe, MaxDD por período."),
        ("Tab 4 — Comparativa 9 modelos",
         "Para el mejor feature set, tabla de los 9 modelos: Sharpe_train, Sharpe_test, "
         "CAGR_train, CAGR_test, MaxDD_test. Celda ganadora resaltada en verde."),
    ]:
        pdf.set_font("DV", "B", 9)
        pdf.set_text_color(*AZUL)
        pdf.cell(0, 5, tab_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*NEGRO)
        pdf.set_font("DV", "", 9)
        pdf.set_x(pdf.l_margin + 8)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(1)

    pdf.subtitulo("Sección 3 — Análisis de variables")
    for b in [
        "Barras: frecuencia de cada variable en el top-10 de individuos finales.",
        "Scatter: param_idx elegido vs fitness (¿hay umbrales preferidos?).",
        "Tabla: variables nunca seleccionadas (candidatos a eliminar del registro).",
    ]:
        pdf.bullet(b)

    # ── IMPLEMENTACION ────────────────────────────────────────────────────────
    pdf.titulo_seccion("8. Orden de implementación")
    pasos = [
        "src/features/registry.py — definir las 400 features (funciones compute)",
        "src/features/builder.py — build_feature_matrix()",
        "src/ml/models.py + cross_val.py — MODEL_REGISTRY + walk_forward_cv()",
        "src/genetic/chromosome.py — Gene, Individual",
        "src/genetic/operators.py — crossover, mutate, select",
        "src/genetic/fitness.py — evaluate_individual()",
        "src/genetic/ga.py — run_ga()",
        "app/pages/3_algoritmo.py — página Streamlit completa",
        "Actualizar requirements.txt (xgboost, catboost, lightgbm)",
        "Smoke test: 3 generaciones x 5 individuos → sin errores",
    ]
    for i, paso in enumerate(pasos, 1):
        pdf.set_font("DV", "", 9)
        pdf.set_text_color(*NEGRO)
        pdf.multi_cell(0, 6, f"  {i}. {paso}",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.titulo_seccion("9. Verificación")
    pdf.code_block(
        "conda activate fibras-mx\n\n"
        "# Smoke test GA rápido (3 gen × 5 individuos)\n"
        "python -c \"\n"
        "from src.features.builder import build_feature_matrix\n"
        "from src.genetic.ga import run_ga\n"
        "result = run_ga(n_generations=3, population_size=5)\n"
        "print('Mejor Sharpe train:', result.best_individual.fitness)\n"
        "print('Genes:', result.best_individual.genes)\n"
        "print('Modelo:', result.best_individual.model_type)\n"
        "print('Horizonte H:', result.best_individual.horizon)\n"
        "\"\n\n"
        "# Verificar en Streamlit\n"
        "streamlit run app/main.py\n"
        "# -> Nueva pestaña 'Algoritmo' en sidebar\n"
        "# -> Botón 'Correr GA' -> spinner -> gráficas de convergencia"
    )

    out = "/Users/danielbecerrilolguin/FIBRAS-mx/docs/plan_algoritmo.pdf"
    pdf.output(out)
    print(f"PDF generado: {out}")


if __name__ == "__main__":
    build_pdf()
