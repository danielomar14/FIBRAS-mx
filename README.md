# FIBRAS-mx

Análisis cuantitativo de las **15 FIBRAs mexicanas** activas en BMV/BIVA. El proyecto cubre tres capas de investigación sobre el mismo universo de datos:

1. **Experimento 1** — Backtest comparativo de 14 estrategias deterministas (E0–E13) sobre 2021–2026 con capital inicial de $450,000 MXN y aportación mensual de $10,000.
2. **Experimento 2A** — ML Puro: 9 modelos de aprendizaje automático (Decision Tree, Ridge, Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, Extra Trees, ElasticNet) sobre 400 features diarias.
3. **Experimento 2B** — Algoritmo Genético: búsqueda combinatoria en el espacio de 40 variables × 10 parametrizaciones para descubrir la regla de scoring que maximiza el Sharpe sin usar ML.

Todo el análisis es accesible a través de una app Streamlit multipágina que corre localmente.

---

## Universo

| Ticker | Nombre | Sector |
|--------|--------|--------|
| FUNO11 | Fibra Uno | Diversificada |
| FIBRAPL14 | Fibra Prologis | Industrial/logística |
| FIBRAMQ12 | Fibra Macquarie | Industrial + comercial |
| DANHOS13 | Fibra Danhos | Comercial + oficinas |
| FMTY14 | Fibra Monterrey | Mixta (mensual) |
| FIHO12 | Fibra Hotel | Hotelero |
| FINN13 | Fibra Inn | Hotelero |
| FSHOP13 | Fibra Shop | Centros comerciales |
| FIBRAUP18 | Fibra Upsite | Industrial PyMEs |
| FNOVA17 | Fibra Nova | Industrial + mixta |
| FPLUS16 | Fibra Plus | Diversificada pequeña |
| STORAGE18 | Fibra Storage | Self-storage |
| FSITES20 | Fibra Sites | Infraestructura telco |
| EDUCA18 | Fibra Educa | Educativo |
| NEXT25 | Fibra Next | Industrial (desde 2025) |

---

## Quickstart

```bash
conda create -n fibras-mx python=3.12 -y
conda activate fibras-mx
pip install -r requirements.txt
streamlit run app/main.py
```

La primera vez que se cargue la app, el pipeline descarga y cachea precios (yfinance) y distribuciones (Supabase FibrasMX). Puede tardar ~1 minuto.

---

## Estructura del repositorio

```
FIBRAS-mx/
├── app/
│   ├── main.py                  # Entrada Streamlit (menú lateral)
│   └── pages/
│       ├── 0_datos.py           # Dashboard de calidad de datos
│       ├── 1_faqs.py            # 8 tabs de análisis de mercado 2021-2026
│       ├── 2_estrategias.py     # Backtest E0-E13, curvas, drawdown, heatmap
│       ├── 3_algoritmo.py       # 2A ML Puro + 2B Algoritmo Genético
│       ├── 4_resultados.py      # Tabla comparativa + animación convergencia GA
│       └── 5_reportes.py        # Reportes de portafolio (CSV + gráficas)
├── src/
│   ├── data/
│   │   ├── fetcher.py           # Pipeline multi-fuente (yfinance → Supabase → CSV)
│   │   ├── fibrasmx.py          # Cliente Supabase FibrasMX (dividendos, métricas)
│   │   ├── amefibra.py          # Scraper AMEFIBRA
│   │   ├── benchmarks.py        # IPC, CETES 28d, tipo Banxico
│   │   └── validator.py         # Quality checks y reporte de cobertura
│   ├── backtest/
│   │   ├── engine.py            # Motor de simulación diaria (DRIP, comisión, rebalanceo)
│   │   ├── strategies.py        # 14 estrategias E0–E13
│   │   └── metrics.py           # CAGR, vol, Sharpe, Calmar, MaxDD, consistencia
│   ├── features/
│   │   ├── registry.py          # 400 features (40 vars × 10 params) con funciones compute
│   │   └── builder.py           # build_feature_matrix(gene_ids) → DataFrame
│   ├── ml/
│   │   ├── models.py            # MODEL_REGISTRY: 9 modelos con hiperparámetros
│   │   ├── cross_val.py         # Walk-forward CV (ventana mín 252 días, paso 63 días)
│   │   └── runner.py            # run_all_models() → dict de resultados
│   └── genetic/
│       ├── chromosome.py        # Gene, Individual dataclasses
│       ├── operators.py         # crossover(), mutate(), tournament_select()
│       ├── fitness.py           # evaluate_individual() → Sharpe ajustado
│       └── ga.py                # run_ga(config) → GAResult
├── docs/
│   ├── bitacora.pdf             # Bitácora completa del proyecto (sesiones)
│   ├── bitacora.tex             # Fuente LaTeX de la bitácora
│   ├── plan_algoritmo.pdf       # Plan detallado del Experimento 2 (6 pp)
│   └── fibras_mexico (1).pdf    # Referencia sobre FIBRAs mexicanas
├── reports/                     # Salidas de export_results.py
│   ├── consolidado.csv          # Tabla comparativa de todas las estrategias
│   ├── portafolio_*.csv         # Evolución diaria por estrategia
│   ├── *.png                    # Equity curves con anotaciones
│   └── *_comentarios.csv        # Eventos relevantes por estrategia
├── results/
│   ├── ml_results.pkl           # Cache de resultados 2A (ML Puro)
│   └── ga_results.pkl           # Cache de resultados 2B (GA)
├── data/
│   └── processed/               # Parquets generados (excluidos del repo)
├── requirements.txt
└── experimento 1.txt / 2.txt    # Especificaciones originales de los experimentos
```

---

## Experimento 1 — Backtest comparativo

**Parámetros:** $450,000 MXN iniciales · $10,000/mes · 2021-01-01 a 2026-05-06 · rebalanceo trimestral · DRIP 100% · comisión 0.25% + IVA · slippage 0.10%.

| Código | Estrategia | Señal principal |
|--------|-----------|-----------------|
| E0 | Naive 1/N (benchmark) | Igual peso en todas las FIBRAs disponibles |
| E1 | Large Cap | Peso proporcional a precio promedio 30d |
| E2 | Momentum 12M | Top 5 por retorno trailing 252 días |
| E3 | Momentum 3M | Top 5 por retorno trailing 63 días |
| E4 | Baja volatilidad | Top 5 con menor vol realizada 252 días |
| E5 | Alto yield | Top 5 por dividendo TTM / precio |
| E6 | Calidad (ocupación) | Top 5 por tasa de ocupación trimestral |
| E7 | Valor (FFO yield) | Top 5 por FFO anualizado / precio |
| E8 | Momentum + Yield | Top 5 por score compuesto 50/50 |
| E9 | Contrarian | Top 5 peores retornos 12M |
| E10 | Rotación sectorial | Industrial/hotelero según dirección Banxico |
| E11 | Filtros fundamentales | Ocupación ≥90%, LTV ≤40%, payout ≤80%; score yield+FFO |
| E12 | Medias móviles | Golden Cross: precio > MA50 > MA200 |
| E13 | Precio < NAV | Ocupación ≥90% y precio < valor teórico |

---

## Experimento 2A — ML Puro

9 modelos entrenados sobre el universo de **hasta 400 features diarias**. La tarea es ranking: predecir qué FIBRAs tendrán mejor retorno relativo a 63 días (1 trimestre).

- **Train:** 2017-01 → 2023-12 (~26,250 observaciones diarias × 15 FIBRAs)
- **Test:** 2024-01 → 2025-12
- **Holdout (validation):** 2026-01 → hoy

Evaluación con walk-forward CV dentro del train (ventana mín 252 días, paso 63 días).

---

## Experimento 2B — Algoritmo Genético

Búsqueda combinatoria sobre 400 genes posibles (40 variables × 10 parametrizaciones). Un individuo elige 1–5 genes; el cromosoma **es** la estrategia (suma ponderada de features → top-k FIBRAs por trimestre). No interviene ningún modelo ML.

**Fitness:** Sharpe del portafolio en train − 0.02 × (n_genes − 1) (parsimony penalty).

**Parámetros:** 20 individuos · 50 generaciones · torneo k=3 · elitismo top-2 · early stopping 10 generaciones.

---

## Fuentes de datos

| Dato | Fuente primaria | Fallback |
|------|----------------|---------|
| Precios diarios | yfinance (.MX) | CSV manual |
| Distribuciones | Supabase FibrasMX | yfinance |
| Métricas trimestrales (ocupación, NOI, FFO, LTV) | Supabase FibrasMX | — |
| IPC México | yfinance (^MXX) | — |
| CETES 28d | Tabla mensual hardcoded (Banxico SIE) | 7.5% default |
| Tipo Banxico | Serie histórica 2018–2025 | — |

> **Nota:** Yahoo Finance tiene ajuste de dividendos inconsistente en FIBRAs. Se usa precio **sin ajustar** + dividendos por separado para calcular retorno total manualmente.

---

## Documentación

| Archivo | Contenido |
|---------|-----------|
| `docs/bitacora.pdf` | Bitácora de sesiones (setup, pipeline, backtest, Exp. 2) |
| `docs/plan_algoritmo.pdf` | Plan técnico detallado del Experimento 2 (ML + GA) |
| `experimento 1.txt` | Especificación original del Experimento 1 |
| `experimento 2.txt` | Especificación original del Experimento 2 |

---

## Requisitos

```
Python 3.12
yfinance, pandas, numpy, pyarrow
streamlit, plotly, matplotlib
scikit-learn, xgboost, catboost, lightgbm
deap
requests, beautifulsoup4, lxml
```

Instalación completa: `pip install -r requirements.txt`

---

## Notas importantes

- **NEXT25** solo existe desde 2025; se excluye de backtests anteriores a esa fecha.
- **Terrafina (TERRA13)** fue absorbida por Fibra Prologis en 2024; no se incluye.
- FIBRAs con baja liquidez (STORAGE18, EDUCA18, FSITES20, FIBRAUP18) tienen huecos en sus series. El engine usa `ffill(limit=60)` para valoración.
- El bug crítico del engine (pérdida de capital al liquidar con precio NaN) está corregido en `src/backtest/engine.py` desde la sesión 3.

---

*Proyecto personal de análisis cuantitativo. Capital de referencia: $450,000 MXN. Inicio: mayo 2026.*
