"""
Genera plan_algoritmo.pdf con fpdf2 + fuentes DejaVu (Unicode completo).
Experimento 2 — dos sub-experimentos independientes:
  2A: ML Puro (9 modelos)
  2B: Algoritmo Genético (búsqueda combinatoria, sin ML)
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_DIR = (
    "/Users/danielbecerrilolguin/anaconda3/envs/fibras-mx/lib/python3.12"
    "/site-packages/matplotlib/mpl-data/fonts/ttf"
)

AZUL      = (30, 80, 160)
AZUL_CLR  = (200, 215, 240)
VERDE_OSC = (0, 100, 50)
VERDE_CLR = (200, 235, 215)
NARANJA   = (180, 80, 0)
NARAN_CLR = (255, 235, 210)
GRIS      = (80, 80, 80)
BLANCO    = (255, 255, 255)
NEGRO     = (20, 20, 20)
GRIS_CLR  = (245, 245, 245)
LINEA     = (180, 180, 180)


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
        self.cell(0, 8, "FIBRAS-mx  ·  Plan: Experimento 2 — ML Puro + Algoritmo Genético", align="L")
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

    def banner(self, txt, color_bg, color_txt=None):
        """Encabezado de sección coloreado."""
        self.ln(6)
        self.set_fill_color(*color_bg)
        self.set_text_color(*(color_txt or BLANCO))
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

    def bullet(self, txt, color=None):
        self.set_font("DV", "", 9)
        self.set_text_color(*(color or NEGRO))
        self.set_x(self.l_margin + 6)
        self.cell(5, 5, "•")
        self.multi_cell(0, 5, txt)
        self.set_text_color(*NEGRO)

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
        col_w = [28, 44, 28, 44, 26]
        headers = ["Split", "Período", "Días hábiles", "Obs. (días×15)", "Propósito"]
        self.set_fill_color(*AZUL_CLR)
        self.set_font("DV", "B", 8)
        self.set_text_color(*AZUL)
        for w, h in zip(col_w, headers):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        rows = [
            ("Train",      "2017-01 → 2023-12", "~1,750", "~26,250", "GA + ML"),
            ("Test",       "2024-01 → 2025-12", "~500",   "~7,500",  "Evaluación"),
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

    def caja_info(self, titulo, texto, color_borde, color_fondo):
        """Caja destacada para diferenciar los dos sub-experimentos."""
        self.set_fill_color(*color_fondo)
        self.set_draw_color(*color_borde)
        self.set_font("DV", "B", 9)
        self.set_text_color(*color_borde)
        self.cell(0, 7, f"  {titulo}", border=1, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DV", "", 8.5)
        self.set_text_color(*NEGRO)
        self.set_fill_color(*color_fondo)
        self.multi_cell(0, 5, f"  {texto}", border="LRB", fill=True,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LINEA)
        self.ln(3)


def build_pdf():
    pdf = PDF()

    # ── PORTADA ──────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*AZUL)
    pdf.rect(0, 0, 216, 65, "F")
    pdf.set_y(10)
    pdf.set_text_color(*BLANCO)
    pdf.set_font("DV", "B", 24)
    pdf.cell(0, 12, "FIBRAS-mx", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DV", "B", 15)
    pdf.cell(0, 9, "Experimento 2: Algoritmo", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("DV", "", 10)
    pdf.cell(0, 6, "Dos sub-experimentos independientes con el mismo split de datos",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(78)
    pdf.set_text_color(*NEGRO)
    for label, val in [
        ("Autor",    "Daniel Becerril Olguín"),
        ("Fecha",    "Mayo 2026"),
        ("Stack",    "Python 3.12 · scikit-learn · XGBoost · CatBoost · LightGBM · Streamlit"),
        ("Datos",    "2017–2026 · 15 FIBRAs · frecuencia diaria"),
    ]:
        pdf.set_font("DV", "B", 10)
        pdf.cell(45, 7, label + ":", border="B")
        pdf.set_font("DV", "", 10)
        pdf.cell(0, 7, val, border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)

    # Resumen: dos cajas de colores distintos
    pdf.caja_info(
        "Sub-experimento 2A — ML Puro",
        "Entrenar 9 modelos de Machine Learning (Decision Tree, Ridge, Logistic Regression, "
        "Random Forest, XGBoost, CatBoost, LightGBM, Extra Trees, ElasticNet) sobre el "
        "universo completo de hasta 400 features. El modelo aprende qué señales predicen "
        "mejor el retorno de cada FIBRA. No hay GA ni selección manual de features.",
        VERDE_OSC, VERDE_CLR,
    )
    pdf.caja_info(
        "Sub-experimento 2B — Algoritmo Genético",
        "Buscar en el espacio de 40 variables × 10 parametrizaciones = 400 genes posibles, "
        "eligiendo combinaciones de 1 a 5 genes. El cromosoma ES la estrategia: define una "
        "regla de scoring directa que selecciona las top-k FIBRAs cada trimestre. "
        "Ningún modelo ML interviene en la evaluación.",
        NARANJA, NARAN_CLR,
    )

    # ── DATOS COMPARTIDOS ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf.banner("0. Datos compartidos por ambos sub-experimentos", AZUL)

    pdf.subtitulo("Frecuencia: DIARIA — no solo trimestral")
    pdf.parrafo(
        "Ambos sub-experimentos comparten el mismo dataset. Se usa frecuencia DIARIA: "
        "~26,250 observaciones en train (vs ~420 si usáramos solo rebalanceos trimestrales)."
    )
    for b in [
        "Precios OHLCV: diarios → features técnicas calculadas día a día.",
        "Fundamentales trimestrales: forward-fill diario entre reportes.",
        "Rebalanceo: sigue siendo trimestral (ejecución igual que E0–E13).",
    ]:
        pdf.bullet(b)
    pdf.ln(4)

    pdf.subtitulo("Partición de datos")
    pdf.tabla_splits()

    pdf.subtitulo("Universo de features: 40 variables × 10 parametrizaciones = 400 features")
    pdf.set_fill_color(*AZUL_CLR)
    self_col = [16, 60, 22, 72]
    pdf.set_font("DV", "B", 8)
    pdf.set_text_color(*AZUL)
    for w, h in zip(self_col, ["Bloque", "Origen", "Vars", "Ejemplo de parametrización"]):
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln()
    bloques = [
        ("A (15 vars)", "Derivadas de E0–E13 (retorno, vol, MA, yield, NAV, occ…)", "×10",
         "ret trailing: 21d, 42d, 63d, 84d, 126d, 168d, 252d, 378d, 504d, 756d"),
        ("B (5 vars)",  "Fundamentales nuevas (NOI, deuda/activos, dilución…)", "×10",
         "NOI margin: raw, rank, >30%, >40%, >50%, >60%, >65%, QoQ, YoY, trend4Q"),
        ("C (5 vars)",  "Técnicas nuevas (RSI, Bollinger, z-vol, 52w-high, Amihud)", "×10",
         "RSI: 7, 9, 14, 21, 28, 42, 63, <30(oversold), >70(overbought), dist-50"),
        ("D (5 vars)",  "Tendencia nuevas (aceleración, multi-TF, consistencia…)", "×10",
         "Score multi-TF: 8 combos de ponderaciones distintas entre 1M/3M/6M/12M"),
        ("E (5 vars)",  "Mercado nuevas (fuerza relativa, sector, corr IPC…)", "×10",
         "Fuerza vs índice: ret-ret_index (21d, 63d, 126d, 252d), rank, >0%, >5%…"),
        ("F (5 vars)",  "Macro nuevas (tasa real, spread CETES, USD/MXN, IPC)", "×10",
         "Spread yield-CETES: >0%, >1%, >2%, >3%, >4%, zscore, rank, tendencia…"),
    ]
    pdf.set_font("DV", "", 7.5)
    pdf.set_text_color(*NEGRO)
    for i, (bl, orig, n, ej) in enumerate(bloques):
        fill = (i % 2 == 0)
        pdf.set_fill_color(248, 250, 255) if fill else pdf.set_fill_color(*BLANCO)
        pdf.cell(self_col[0], 5, bl, border="LTB", fill=fill)
        pdf.cell(self_col[1], 5, orig, border="TB", fill=fill)
        pdf.cell(self_col[2], 5, n, border="TB", fill=fill, align="C")
        pdf.multi_cell(self_col[3], 4.5, ej, border="RTB", fill=fill,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ── SUB-EXPERIMENTO 2A ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.banner("Sub-experimento 2A — ML Puro", VERDE_OSC)

    pdf.parrafo(
        "Se entrena cada uno de los 9 modelos sobre el conjunto completo de features "
        "(hasta 400 columnas). No hay selección previa de variables — el modelo aprende "
        "qué pesa. La tarea es ranking: predecir qué FIBRAs tendrán mejor retorno relativo "
        "en los próximos 63 días (1 trimestre)."
    )

    pdf.subtitulo("Pipeline")
    pdf.code_block(
        "1. Construir feature matrix diaria completa (hasta 400 columnas)\n"
        "   - Features técnicas: día a día desde precios OHLCV\n"
        "   - Features fundamentales: forward-fill de datos trimestrales\n\n"
        "2. Label: retorno de cada FIBRA a 63 días hacia adelante\n"
        "   - Normalizado 0–1 con rank entre las 15 FIBRAs\n"
        "   - Horizonte fijo: 63 días (1 trimestre = 1 período de rebalanceo)\n\n"
        "3. Walk-forward CV dentro del train (ventana mín 252 días, paso 63 días):\n"
        "   Para cada fold t:\n"
        "     - Entrenar modelo en días ≤ t\n"
        "     - Predecir scores diarios para [t+1 … t+63]\n"
        "     - Agregar scores por FIBRA (promedio diario)\n"
        "     - Seleccionar top-5 FIBRAs\n"
        "     - Calcular retorno portfolio (igual peso, 63 días)\n\n"
        "4. Evaluar en Test y Validation con modelo entrenado en todo el Train"
    )

    pdf.subtitulo("9 Modelos con hiperparámetros")
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

    pdf.subtitulo("Output de la Sección 2A en Streamlit")
    for b in [
        "Tabla comparativa: Sharpe_train, Sharpe_test, CAGR_train, CAGR_test, MaxDD_test",
        "Equity curve del mejor modelo vs E0 (Naive 1/N), E13 y CETES",
        "Feature importance de los modelos que la soportan (RF, XGB, LGB)",
    ]:
        pdf.bullet(b, VERDE_OSC)

    # ── SUB-EXPERIMENTO 2B ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.banner("Sub-experimento 2B — Algoritmo Genético", NARANJA)

    pdf.parrafo(
        "Búsqueda combinatoria sobre el espacio de 400 genes posibles. "
        "Un individuo elige entre 1 y 5 genes; cada gen es una (variable, parametrización). "
        "El cromosoma define directamente la regla de scoring: para cada FIBRA se suma "
        "el valor de sus genes seleccionados, se rankea, y se elige el top-k. "
        "No interviene ningún modelo de ML."
    )

    pdf.subtitulo("Encoding del cromosoma")
    pdf.code_block(
        "Gene = namedtuple('Gene', ['var_id', 'param_idx'])\n"
        "# var_id   : 0–39  (40 variables)\n"
        "# param_idx: 0–9   (10 parametrizaciones)\n"
        "# Espacio total: 400 genes posibles\n\n"
        "@dataclass\n"
        "class Individual:\n"
        "    genes:   list[Gene]   # 1–5 genes, sin repetir var_id\n"
        "    top_k:   int          # FIBRAs a seleccionar: in {3,4,5,6,7}\n"
        "    fitness: float = -inf"
    )

    pdf.subtitulo("Función de Fitness (sin ML)")
    pdf.code_block(
        "fitness(individual, train_data):\n"
        "  Para cada fecha de rebalanceo t en el train:\n"
        "    a. Para cada FIBRA: score = suma de gene_value[var_id][param_idx]\n"
        "    b. Normalizar scores (rank 0–1 entre FIBRAs)\n"
        "    c. Seleccionar top_k FIBRAs con mayor score\n"
        "    d. Retorno del portfolio: igual peso, hold 1 trimestre\n\n"
        "  Sharpe de la serie de retornos trimestrales\n"
        "  Parsimony penalty: -0.02 × (n_genes - 1)   ← menos genes = menos overfitting\n"
        "  Retornar Sharpe ajustado"
    )

    pdf.subtitulo("Parámetros del GA")
    for b in [
        "Población: 20 individuos",
        "Generaciones: 50 (configurable en UI, slider 10–100)",
        "Selección: torneo de tamaño 3, elitismo top-2",
        "Early stopping: sin mejora en 10 generaciones",
        "Re-diversificación: si >80% población idéntica, reiniciar 40% (immigration)",
    ]:
        pdf.bullet(b, NARANJA)
    pdf.ln(2)

    pdf.subtitulo("Operadores evolutivos")
    pdf.code_block(
        "Inicialización:\n"
        "  n_genes = random.choice([1,2,3,4,5])\n"
        "  genes   = random.sample(ALL_400_GENES, n_genes)   # sin repetir var_id\n"
        "  top_k   = random.choice([3,4,5,6,7])\n\n"
        "Cruzamiento:\n"
        "  gene_pool   = union(parent1.genes, parent2.genes)\n"
        "  n_child     = random.choice([1…5])\n"
        "  child_genes = random.sample(gene_pool, min(n_child, len(pool)))\n"
        "  child_top_k = random.choice([p1.top_k, p2.top_k])\n\n"
        "Mutación (P=0.3 por tipo, independientes):\n"
        "  mutate_add:    agregar gene aleatorio (si n < 5)\n"
        "  mutate_remove: eliminar gene aleatorio (si n > 1)\n"
        "  mutate_swap:   reemplazar gene por otro (distinto var_id)\n"
        "  mutate_param:  cambiar param_idx de un gene existente\n"
        "  mutate_topk:   cambiar top_k (P=0.2)"
    )

    pdf.subtitulo("Output de la Sección 2B en Streamlit")
    for b in [
        "Gráfica de convergencia: mejor fitness y media poblacional por generación",
        "Cromosoma ganador decodificado: variables, parametrización, top_k",
        "Equity curve en Train / Test / Validation vs E0 y E13",
        "Frecuencia de cada variable en el top-10 de individuos finales",
    ]:
        pdf.bullet(b, NARANJA)

    # ── FEATURES (bloques) ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.banner("1. Detalle de features — Bloque A (estrategias existentes E0–E13)", AZUL)
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

    pdf.banner("Bloques B–F (variables nuevas)", AZUL)
    pdf.tabla_features([
        ("B01", "NOI margin",              "raw, rank, >30%, >40%, >50%, >60%, >65%, QoQ-change, YoY-change, trend4Q"),
        ("B02", "Deuda / Activos",         "raw, rank, <0.35, <0.40, <0.45, <0.50, <0.55, QoQ-change, inverted, trend4Q"),
        ("B03", "Dilución de CBFIs",       "%change YoY, positivo, negativo, abs, <1%, <3%, <5%, <-1%, <-3%, z-score"),
        ("B04", "Apreciación propiedades", "%change YoY, rank, >0%, >2%, >5%, >8%, <0%, trend4Q, QoQ-change, zscore"),
        ("B05", "Distribución / Activos",  "dist×cbfis/total_assets, rank, >1%, >2%, >3%, >4%, >5%, trend4Q, QoQ, z-score"),
        ("C01", "RSI",                     "RSI-7, RSI-9, RSI-14, RSI-21, RSI-28, RSI-42, RSI-63, RSI<30, RSI>70, dist-de-50"),
        ("C02", "Posición Bollinger",      "BB(20,2), BB(20,1.5), BB(50,2), %B(0-1), %B<0.2, %B>0.8, BB-width, BB-width-z, squeeze, inv-%B"),
        ("C03", "Z-score volumen",         "z(21d), z(63d), z(126d), z(252d), >+1σ, >+2σ, <-1σ, tendencia, ratio(21/63), ratio(5/21)"),
        ("C04", "Precio / 52-week high",   "raw, >0.90, >0.95, >0.98, <0.80, <0.70, dist%, drawdown-52w, nuevo-máx, recuperación"),
        ("C05", "Amihud Illiquidity",      "ratio(21d), ratio(63d), ratio(252d), rank_liq, rank_iliq, thresh_high, thresh_low, trend4Q, zscore, vs-sector"),
        ("D01", "Aceleración momentum",    "ret63-ret126, ret21-ret63, ret126-ret252, ret63/ret126, señal_pos, zscore, rank, 2da-deriv, negativa, norm"),
        ("D02", "Score multi-timeframe",   "0.25×ret21+0.25×ret63+0.25×ret126+0.25×ret252, + 8 combos ponderaciones, top25%"),
        ("D03", "Consistencia retornos",   "%días pos 21d, 63d, 126d, 252d, >50%, >60%, >70%, tendencia-mejora, zscore, rank"),
        ("D04", "EMA vs SMA",              "EMA21/SMA21, EMA50/SMA50, EMA vs SMA, >1, <1, distancia, ratio-cruzado, señal, z-score, rank"),
        ("D05", "Canal de precio",         "%channel(20d), %channel(63d), %channel(126d), >80%, >90%, <20%, <10%, breakout_up, breakout_dn, norm"),
        ("E01", "Fuerza relativa vs índice","ret - ret_index (21d, 63d, 126d, 252d), rank, >0%, >5%, >10%, zscore, rolling_rank, tendencia"),
        ("E02", "Momentum sectorial",      "avg_ret_peers (21d, 63d, 126d), vs_sector, sector_rank, top2, bottom2, spread, zscore, rank, tendencia"),
        ("E03", "Tendencia de volumen",    "slope(21d), slope(63d), slope(126d), >0%, >+20%, >+50%, ratio(5/15d), zscore, señal-pos, rank"),
        ("E04", "Correlación con IPC",     "corr_63d, corr_126d, corr_252d, <0.3, >0.7, beta_63d, beta_126d, beta_252d, tracking-error, corr-inversa"),
        ("E05", "Tamaño relativo",         "rank(top5), rank(top3), decil, tercil, log_price30d, cambio_rank_YoY, stable_large, concentración, peso_eq, peso_prop"),
        ("F01", "Tasa real (Banxico-inf)", "tasa-CPI_proxy, >5%, >6%, >7%, >8%, <4%, cambio_rate_real, zscore, rank, tendencia"),
        ("F02", "Spread yield-CETES",      "div_yield - CETES_28d, >0%, >1%, >2%, >3%, >4%, zscore, rank, tendencia-spread, negativo"),
        ("F03", "Fase ciclo de tasas",     "subiendo(1/0), bajando(1/0), pausa(1/0), acelerando, desacelerando, cum_change_6M, velocidad, zscore, señal, dirección"),
        ("F04", "Tendencia USD/MXN",       "ret_usdmxn (21d, 63d, 126d, 252d), >0% peso-débil, <0% peso-fuerte, >+5%, <-5%, vol, corr-FIBRA, zscore"),
        ("F05", "Momentum IPC",            "ret_ipc (21d, 63d, 126d, 252d), >0%, >5%, >10%, <0%, zscore, rank_hist, tendencia, IPC>MA200(1/0), breadth_signal"),
    ])

    # ── ESTRUCTURA + IMPLEMENTACION ───────────────────────────────────────────
    pdf.add_page()
    pdf.banner("2. Estructura de archivos e implementación", AZUL)
    pdf.code_block(
        "src/\n"
        "  features/\n"
        "    registry.py     # FEATURE_REGISTRY: dict[int, FeatureDef] (400 features)\n"
        "    builder.py      # build_feature_matrix(gene_ids, ...) -> DataFrame\n"
        "  ml/                           # Sub-experimento 2A\n"
        "    models.py       # MODEL_REGISTRY: instanciar modelo por tipo+params\n"
        "    cross_val.py    # walk_forward_cv(...) -> List[float]\n"
        "    runner.py       # run_all_models(feature_matrix, labels) -> dict\n"
        "  genetic/                      # Sub-experimento 2B\n"
        "    chromosome.py   # Gene, Individual dataclasses\n"
        "    operators.py    # crossover(), mutate(), tournament_select()\n"
        "    fitness.py      # evaluate_individual(ind, rebalance_data) -> float\n"
        "    ga.py           # run_ga(config) -> GAResult\n"
        "app/pages/\n"
        "  3_algoritmo.py    # Sección 2A (ML Puro) + Sección 2B (GA)\n"
        "results/\n"
        "  ml_results.pkl    # Cache resultados 2A\n"
        "  ga_results.pkl    # Cache resultado 2B"
    )

    pdf.subtitulo("Orden de implementación")
    for i, paso in enumerate([
        "src/features/registry.py — 400 features con funciones compute",
        "src/features/builder.py — build_feature_matrix()",
        "src/ml/models.py + cross_val.py + runner.py — 9 modelos (2A)",
        "src/genetic/chromosome.py — Gene, Individual",
        "src/genetic/operators.py — crossover, mutate, select",
        "src/genetic/fitness.py — evaluate_individual() sin ML (2B)",
        "src/genetic/ga.py — run_ga()",
        "app/pages/3_algoritmo.py — Sección 2A + Sección 2B",
        "Actualizar requirements.txt (xgboost, catboost, lightgbm)",
        "Smoke test: 3 gen × 5 individuos + 1 modelo ML → sin errores",
    ], 1):
        pdf.set_font("DV", "", 9)
        pdf.set_text_color(*NEGRO)
        pdf.multi_cell(0, 6, f"  {i}. {paso}",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.subtitulo("Verificación")
    pdf.code_block(
        "conda activate fibras-mx\n\n"
        "# 2A: ML Puro\n"
        "python -c \"\n"
        "from src.ml.runner import run_all_models\n"
        "results = run_all_models()\n"
        "print('Mejor modelo:', max(results, key=lambda k: results[k]['sharpe_test']))\n"
        "\"\n\n"
        "# 2B: GA (smoke test rápido)\n"
        "python -c \"\n"
        "from src.genetic.ga import run_ga\n"
        "result = run_ga(n_generations=3, population_size=5)\n"
        "print('Mejor Sharpe train:', result.best_individual.fitness)\n"
        "print('Genes:', result.best_individual.genes)\n"
        "print('Top-k:', result.best_individual.top_k)\n"
        "\"\n\n"
        "# Streamlit\n"
        "streamlit run app/main.py\n"
        "# -> Pestaña 'Algoritmo': Sección 2A (ML Puro) | Sección 2B (GA)"
    )

    out = "/Users/danielbecerrilolguin/FIBRAS-mx/docs/plan_algoritmo.pdf"
    pdf.output(out)
    print(f"PDF generado: {out}")


if __name__ == "__main__":
    build_pdf()
