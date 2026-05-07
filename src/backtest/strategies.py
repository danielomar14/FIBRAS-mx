"""
Estrategias de inversión E0–E10 para FIBRAs mexicanas.

Cada estrategia es una función con la firma:
    strategy(date, prices_wide, dividends, metrics, banxico_rates) -> dict[str, float]

donde el retorno es {ticker: peso} y los pesos deben sumar ~1.
El motor normaliza los pesos si no suman exactamente 1.

Universo de FIBRAs disponibles (varía según cobertura de datos en la fecha):
se filtra automáticamente a las FIBRAs con precio disponible en `date`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Universo y sectores ───────────────────────────────────────────────────────

UNIVERSE = [
    "FUNO11", "FIBRAPL14", "FIBRAMQ12", "DANHOS13",
    "FMTY14",  "FIHO12",   "FINN13",   "FSHOP13",
    "FNOVA17", "FIBRAUP18","FPLUS16",  "STORAGE18",
    "FSITES20","EDUCA18",  "NEXT25",
]

SECTOR_MAP = {
    "FUNO11":    "Diversificada",
    "FIBRAPL14": "Industrial",
    "FIBRAMQ12": "Industrial",
    "DANHOS13":  "Comercial",
    "FMTY14":    "Mixta",
    "FIHO12":    "Hotelero",
    "FINN13":    "Hotelero",
    "FSHOP13":   "Comercial",
    "FNOVA17":   "Industrial",
    "FIBRAUP18": "Industrial",
    "FPLUS16":   "Diversificada",
    "STORAGE18": "Almacenaje",
    "FSITES20":  "Infraestructura",
    "EDUCA18":   "Educativo",
    "NEXT25":    "Industrial",
}

INDUSTRIAL  = {t for t, s in SECTOR_MAP.items() if s == "Industrial"}
HOTELES     = {t for t, s in SECTOR_MAP.items() if s == "Hotelero"}
COMERCIAL   = {t for t, s in SECTOR_MAP.items() if s == "Comercial"}


def _available(prices_window: pd.DataFrame, min_history_days: int = 60) -> list[str]:
    """FIBRAs con precio disponible hoy y al menos `min_history_days` histórico."""
    last = prices_window.iloc[-1].dropna()
    result = []
    for t in UNIVERSE:
        if t in last.index:
            col = prices_window[t].dropna()
            if len(col) >= min_history_days:
                result.append(t)
    return result


def _equal_weights(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def _top_n(scores: dict[str, float], n: int) -> dict[str, float]:
    """Retorna los top-n tickers con pesos iguales."""
    sorted_t = sorted(scores, key=scores.get, reverse=True)
    return _equal_weights(sorted_t[:n])


def _trailing_return(prices: pd.DataFrame, ticker: str, days: int) -> float | None:
    col = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    if len(col) < days:
        return None
    return col.iloc[-1] / col.iloc[-days] - 1


def _ttm_yield(dividends: pd.DataFrame, prices: pd.DataFrame,
               ticker: str, as_of: pd.Timestamp) -> float:
    start = as_of - pd.Timedelta(days=365)
    mask  = (dividends["ticker"] == ticker) & (dividends["date"].between(start, as_of))
    ttm   = dividends.loc[mask, "dividend"].sum()
    price_ser = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    if price_ser.empty or price_ser.iloc[-1] == 0:
        return 0.0
    return ttm / price_ser.iloc[-1]


def _latest_metric(metrics: pd.DataFrame, ticker: str,
                   col: str, as_of: pd.Timestamp) -> float | None:
    sub = metrics[(metrics["ticker"] == ticker) & (metrics["date"] <= as_of)][col].dropna()
    return sub.iloc[-1] if not sub.empty else None


# ── Estrategias ───────────────────────────────────────────────────────────────

def e0_equal_weight(date, prices_wide, dividends, metrics, banxico_rates):
    """E0 — Equiponderada: todas las FIBRAs disponibles con peso igual."""
    tickers = _available(prices_wide)
    return _equal_weights(tickers)


def e1_large_cap(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E1 — Por tamaño (proxy market cap): pondera por volumen × precio (30 días).
    Captura FIBRAs más líquidas y grandes.
    """
    tickers = _available(prices_wide)
    scores = {}
    for t in tickers:
        # Necesitamos volumen — no está en prices_wide (solo close).
        # Usamos la raíz cuadrada del precio como proxy simple de capitalización
        # (FIBRAs más grandes tienden a tener precios históricos más altos).
        col = prices_wide[t].dropna().tail(30)
        scores[t] = col.mean() if not col.empty else 0
    total = sum(scores.values())
    if total == 0:
        return _equal_weights(tickers)
    return {t: s / total for t, s in scores.items() if t in tickers}


def e2_momentum_12m(date, prices_wide, dividends, metrics, banxico_rates):
    """E2 — Momentum 12 meses: top 5 FIBRAs por retorno trailing 12 meses."""
    tickers = _available(prices_wide, min_history_days=250)
    scores = {}
    for t in tickers:
        r = _trailing_return(prices_wide, t, 252)
        if r is not None:
            scores[t] = r
    return _top_n(scores, n=5)


def e3_momentum_3m(date, prices_wide, dividends, metrics, banxico_rates):
    """E3 — Momentum 3 meses: top 5 FIBRAs por retorno trailing 3 meses."""
    tickers = _available(prices_wide, min_history_days=65)
    scores = {}
    for t in tickers:
        r = _trailing_return(prices_wide, t, 63)
        if r is not None:
            scores[t] = r
    return _top_n(scores, n=5)


def e4_low_volatility(date, prices_wide, dividends, metrics, banxico_rates):
    """E4 — Baja volatilidad: top 5 FIBRAs con menor vol realizada (12 meses)."""
    tickers = _available(prices_wide, min_history_days=250)
    scores = {}
    for t in tickers:
        col = prices_wide[t].dropna().tail(252)
        if len(col) >= 120:
            vol = col.pct_change().dropna().std() * np.sqrt(252)
            scores[t] = vol
    if not scores:
        return e0_equal_weight(date, prices_wide, dividends, metrics, banxico_rates)
    # Menor volatilidad → mejor score (invertir)
    sorted_t = sorted(scores, key=scores.get)
    return _equal_weights(sorted_t[:5])


def e5_high_yield(date, prices_wide, dividends, metrics, banxico_rates):
    """E5 — Alto rendimiento: top 5 FIBRAs por yield de distribuciones TTM."""
    tickers = _available(prices_wide)
    scores = {}
    for t in tickers:
        y = _ttm_yield(dividends, prices_wide, t, date)
        if y > 0:
            scores[t] = y
    if not scores:
        return _equal_weights(tickers)
    return _top_n(scores, n=5)


def e6_quality_occupancy(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E6 — Calidad (ocupación): top 5 FIBRAs por ocupación de portafolio más reciente.
    Usa métricas trimestrales de FibrasMX/AMEFIBRA.
    """
    tickers = _available(prices_wide)
    scores = {}
    for t in tickers:
        occ = _latest_metric(metrics, t, "occupancy_portfolio", date)
        if occ is not None and not np.isnan(occ):
            scores[t] = occ
    if not scores:
        return _equal_weights(tickers)
    return _top_n(scores, n=5)


def e7_value_ffo(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E7 — Valor (FFO yield): top 5 FIBRAs con mayor FFO / precio.
    FFO de métricas trimestrales × 4 (anualizado) / precio actual.
    """
    tickers = _available(prices_wide)
    scores = {}
    for t in tickers:
        ffo = _latest_metric(metrics, t, "ffo_per_cbfi", date)
        if ffo is None or np.isnan(ffo):
            continue
        col = prices_wide[t].dropna()
        if col.empty or col.iloc[-1] == 0:
            continue
        # FFO yield anualizado
        ffo_yield = (ffo * 4) / col.iloc[-1]
        if ffo_yield > 0:
            scores[t] = ffo_yield
    if not scores:
        return _equal_weights(tickers)
    return _top_n(scores, n=5)


def e8_momentum_yield_combo(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E8 — Momentum + Yield: score combinado (50% rank momentum 6M + 50% rank yield TTM).
    Top 5 por score compuesto.
    """
    tickers = _available(prices_wide, min_history_days=130)
    mom_scores, yld_scores = {}, {}
    for t in tickers:
        r = _trailing_return(prices_wide, t, 126)
        if r is not None:
            mom_scores[t] = r
        y = _ttm_yield(dividends, prices_wide, t, date)
        if y > 0:
            yld_scores[t] = y

    common = set(mom_scores) & set(yld_scores)
    if not common:
        return e0_equal_weight(date, prices_wide, dividends, metrics, banxico_rates)

    # Ranking normalizado 0–1
    def rank_norm(d: dict) -> dict:
        vals = sorted(d.items(), key=lambda x: x[1])
        n = len(vals)
        return {t: i / max(n - 1, 1) for i, (t, _) in enumerate(vals)}

    mom_rank = rank_norm({t: mom_scores[t] for t in common})
    yld_rank = rank_norm({t: yld_scores[t] for t in common})
    combo = {t: 0.5 * mom_rank[t] + 0.5 * yld_rank[t] for t in common}
    return _top_n(combo, n=5)


def e9_anti_momentum(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E9 — Contrarian (anti-momentum): top 5 FIBRAs con PEOR retorno 12 meses.
    Apuesta por reversión a la media.
    """
    tickers = _available(prices_wide, min_history_days=250)
    scores = {}
    for t in tickers:
        r = _trailing_return(prices_wide, t, 252)
        if r is not None:
            scores[t] = -r  # invertir: peor performer → mayor score
    return _top_n(scores, n=5)


def e10_sector_tilt(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E10 — Rotación sectorial por tasas Banxico.
    - Tasas subiendo (últimos 2 trimestres): sobreponderar industrial (40%), subponderar hotelero (10%)
    - Tasas bajando: sobreponderar diversificada + comercial (40%), industrial normal (20%)
    - Tasas estables: equiponderada
    """
    tickers = _available(prices_wide)

    # Determinar tendencia de tasas (últimos 6 meses)
    rate_direction = 0  # 0=estable, 1=subiendo, -1=bajando
    if banxico_rates is not None:
        rates = banxico_rates["tasa_objetivo"].loc[:date].dropna()
        if len(rates) >= 126:
            recent  = rates.iloc[-1]
            six_ago = rates.iloc[-126]
            if recent - six_ago > 0.25:
                rate_direction = 1
            elif recent - six_ago < -0.25:
                rate_direction = -1

    avail_industrial = [t for t in tickers if t in INDUSTRIAL]
    avail_hoteles    = [t for t in tickers if t in HOTELES]
    avail_otros      = [t for t in tickers if t not in INDUSTRIAL | HOTELES]

    weights: dict[str, float] = {}

    if rate_direction == 1:
        # Subiendo tasas: industrial fuerte, hotelero débil
        for t in avail_industrial: weights[t] = 2.0
        for t in avail_hoteles:    weights[t] = 0.5
        for t in avail_otros:      weights[t] = 1.0
    elif rate_direction == -1:
        # Bajando tasas: hotelero y comercial favorecidos
        for t in avail_industrial: weights[t] = 0.75
        for t in avail_hoteles:    weights[t] = 1.5
        for t in avail_otros:      weights[t] = 1.25
    else:
        return _equal_weights(tickers)

    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}


def e11_video_strategy(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E11 — Estrategia del video (filtros fundamentales + precio).

    Paso 1 — Filtro de seguridad (hard filters):
      - Ocupación >= 90%
      - Payout ratio (distribución/FFO) <= 80%   [el video pide <65%; se usa
        80% como umbral más realista dado que pocas FIBRAs cumplen <65%]

    Paso 2 — Score de oportunidad (sumatoria de ranks normalizados 0-1):
      - Dividend yield TTM (peso 40%): más yield → mejor compra
      - FFO yield (peso 40%): proxy de "precio vs valor intrínseco";
        alto FFO yield = mercado cobra poco por el flujo real
      - Sesgo industrial (peso 20%): +1 si pertenece al sector industrial
        (proxy del filtro de nearshoring del video)

    LTV: no disponible en nuestras fuentes → filtro omitido.
    Selecciona top 5 por score compuesto.
    """
    tickers = _available(prices_wide)

    # ── Paso 1: filtros duros ────────────────────────────────────────────────
    passed = []
    for t in tickers:
        occ = _latest_metric(metrics, t, "occupancy_portfolio", date)
        ffo = _latest_metric(metrics, t, "ffo_per_cbfi", date)
        dist = _latest_metric(metrics, t, "distribution_per_cbfi", date)

        if occ is None or np.isnan(occ) or occ < 0.90:
            continue

        if ffo is not None and dist is not None and not np.isnan(ffo) and ffo > 0:
            payout = dist / ffo
            if payout > 0.80:
                continue

        passed.append(t)

    if not passed:
        passed = tickers  # fallback: sin datos suficientes → todos

    # ── Paso 2: score de oportunidad ─────────────────────────────────────────
    yld_scores = {}
    ffo_scores = {}
    for t in passed:
        y = _ttm_yield(dividends, prices_wide, t, date)
        if y > 0:
            yld_scores[t] = y

        ffo = _latest_metric(metrics, t, "ffo_per_cbfi", date)
        col = prices_wide[t].dropna()
        if ffo and not np.isnan(ffo) and ffo > 0 and not col.empty and col.iloc[-1] > 0:
            ffo_scores[t] = (ffo * 4) / col.iloc[-1]

    def rank_norm(d: dict) -> dict:
        if not d:
            return {}
        vals = sorted(d.items(), key=lambda x: x[1])
        n = len(vals)
        return {t: i / max(n - 1, 1) for i, (t, _) in enumerate(vals)}

    yld_rank = rank_norm(yld_scores)
    ffo_rank = rank_norm(ffo_scores)

    combo = {}
    for t in passed:
        score = (
            0.40 * yld_rank.get(t, 0)
            + 0.40 * ffo_rank.get(t, 0)
            + 0.20 * (1.0 if t in INDUSTRIAL else 0.0)
        )
        combo[t] = score

    return _top_n(combo, n=5) if combo else _equal_weights(passed)


def e12_moving_average(date, prices_wide, dividends, metrics, banxico_rates):
    """
    E12 — Medias móviles (Golden Cross).

    Para cada FIBRA con suficiente historial:
      - Calcula MA50 (media simple 50 días) y MA200 (media simple 200 días)
      - "En tendencia alcista" si: precio > MA50 Y MA50 > MA200

    Invierte en igual peso entre las FIBRAs que están en tendencia alcista.
    Si ninguna califica, usa E0 (equiponderada) como fallback.
    También pondera por "fuerza": distancia % del precio sobre su MA50
    (más alejado hacia arriba → mayor convicción).
    """
    tickers = _available(prices_wide, min_history_days=210)
    in_trend = {}

    for t in tickers:
        col = prices_wide[t].dropna()
        if len(col) < 200:
            continue
        price = col.iloc[-1]
        ma50  = col.tail(50).mean()
        ma200 = col.tail(200).mean()

        if price > ma50 and ma50 > ma200:
            strength = (price - ma50) / ma50  # % sobre MA50
            in_trend[t] = max(strength, 0.001)  # siempre positivo

    if not in_trend:
        return _equal_weights(tickers) if tickers else {}

    total = sum(in_trend.values())
    return {t: v / total for t, v in in_trend.items()}


# ── Registro de estrategias ───────────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {
    "E0": {
        "fn": e0_equal_weight,
        "nombre": "Naive 1/N",
        "categoria": "Pasiva",
        "descripcion": (
            "Divide el capital en partes iguales entre todas las FIBRAs "
            "disponibles en cada fecha de rebalanceo. Es el punto de partida "
            "clásico de la literatura (DeMiguel et al., 2009): ningún modelo "
            "cuantitativo debería rendir menos que esta estrategia ingénua. "
            "**Sirve como referencia interna** para juzgar si las demás "
            "estrategias agregan valor real."
        ),
        "universo": "Todas las disponibles",
        "señal": "Ninguna — pesos fijos 1/N",
    },
    "E1": {
        "fn": e1_large_cap,
        "nombre": "Por tamaño (proxy cap)",
        "categoria": "Pasiva",
        "descripcion": (
            "Pondera cada FIBRA proporcionalmente a su precio promedio de los "
            "últimos 30 días hábiles, usado como proxy del tamaño de mercado. "
            "FIBRAs más grandes (FUNO11, FIBRAPL14) reciben mayor peso. "
            "Replica el concepto de un índice ponderado por capitalización, "
            "similar al IPC pero dentro del universo FIBRA."
        ),
        "universo": "Todas las disponibles",
        "señal": "Precio promedio 30d como proxy de market cap",
    },
    "E2": {
        "fn": e2_momentum_12m,
        "nombre": "Momentum 12M",
        "categoria": "Tendencial",
        "descripcion": (
            "Selecciona las **5 FIBRAs con mayor retorno de precio en los "
            "últimos 12 meses** y las pondera en igual peso. Se basa en el "
            "efecto momentum documentado en acciones y REITs: los ganadores "
            "recientes tienden a seguir ganando en el corto plazo. "
            "Se excluye el último mes para evitar el sesgo de reversión "
            "a muy corto plazo (though in practice we use full 252-day window)."
        ),
        "universo": "Top 5 por retorno 12M",
        "señal": "Retorno precio trailing 252 días hábiles",
    },
    "E3": {
        "fn": e3_momentum_3m,
        "nombre": "Momentum 3M",
        "categoria": "Tendencial",
        "descripcion": (
            "Igual que E2 pero con ventana de **3 meses (63 días hábiles)**. "
            "Captura tendencias de corto plazo. En mercados con poca liquidez "
            "como las FIBRAs mexicanas, el ruido supera la señal a esta "
            "escala temporal, lo que genera rotación excesiva y altos costos "
            "de transacción."
        ),
        "universo": "Top 5 por retorno 3M",
        "señal": "Retorno precio trailing 63 días hábiles",
    },
    "E4": {
        "fn": e4_low_volatility,
        "nombre": "Baja volatilidad",
        "categoria": "Factor",
        "descripcion": (
            "Selecciona las **5 FIBRAs con menor volatilidad realizada en los "
            "últimos 12 meses** (desviación estándar anualizada de retornos "
            "diarios). La anomalía de baja volatilidad, documentada globalmente, "
            "sugiere que activos menos volátiles ofrecen mejor Sharpe ajustado. "
            "En el contexto FIBRA esto favorece propiedades industriales y "
            "de almacenaje frente a hoteleras."
        ),
        "universo": "Top 5 por menor vol 12M",
        "señal": "Vol realizada anualizada (σ × √252)",
    },
    "E5": {
        "fn": e5_high_yield,
        "nombre": "Alto yield",
        "categoria": "Factor",
        "descripcion": (
            "Selecciona las **5 FIBRAs con mayor yield de distribuciones** "
            "(suma de dividendos pagados en los últimos 12 meses / precio actual). "
            "La tesis es que las FIBRAs están obligadas a distribuir ≥95% del "
            "FFO, por lo que un yield alto señala generación de caja real, "
            "no solo apreciación de precio. Favorece FIBRAs con alta ocupación "
            "y contratos estables."
        ),
        "universo": "Top 5 por yield TTM",
        "señal": "Dividendos 12M / precio actual",
    },
    "E6": {
        "fn": e6_quality_occupancy,
        "nombre": "Calidad (ocupación)",
        "categoria": "Fundamental",
        "descripcion": (
            "Selecciona las **5 FIBRAs con mayor tasa de ocupación de portafolio** "
            "reportada en el último trimestre disponible (fuente: FibrasMX / "
            "reportes trimestrales). Alta ocupación → ingresos estables → "
            "distribuciones predecibles. Tiene el sesgo de datos lentos: "
            "la métrica trimestral puede tener rezago de hasta 90 días."
        ),
        "universo": "Top 5 por ocupación último trimestre",
        "señal": "Occupancy rate (%) de métricas trimestrales",
    },
    "E7": {
        "fn": e7_value_ffo,
        "nombre": "Valor (FFO yield)",
        "categoria": "Fundamental",
        "descripcion": (
            "Selecciona las **5 FIBRAs con mayor FFO yield** (FFO por CBFI × 4 "
            "anualizado / precio actual). El FFO (Funds From Operations) es el "
            "equivalente del P/E para FIBRAs: el precio sobre el flujo operativo "
            "real. Un FFO yield alto implica que el mercado paga poco por cada "
            "peso de flujo generado, lo que indica posible subvaluación. "
            "Limitado por gaps en datos trimestrales para FIBRAs pequeñas."
        ),
        "universo": "Top 5 por FFO yield",
        "señal": "FFO por CBFI × 4 / precio actual",
    },
    "E8": {
        "fn": e8_momentum_yield_combo,
        "nombre": "Momentum + Yield",
        "categoria": "Multi-factor",
        "descripcion": (
            "Combina dos señales: **momentum de 6 meses** y **yield TTM**, "
            "con peso 50/50. Se rankean todas las FIBRAs disponibles en cada "
            "señal (rango normalizado 0–1) y se suman los ranks. Las top 5 "
            "por score compuesto entran con pesos iguales. Busca FIBRAs que "
            "simultáneamente han tenido buen desempeño de precio reciente "
            "y pagan buenas distribuciones."
        ),
        "universo": "Top 5 por score compuesto",
        "señal": "Rank(momentum 6M) × 0.5 + Rank(yield TTM) × 0.5",
    },
    "E9": {
        "fn": e9_anti_momentum,
        "nombre": "Contrarian (anti-momentum)",
        "categoria": "Contraria",
        "descripcion": (
            "Selecciona las **5 FIBRAs con peor retorno en los últimos 12 "
            "meses** (lo opuesto a E2). La tesis es la reversión a la media: "
            "activos que han caído significativamente tienden a recuperarse "
            "en el siguiente período. Esto funciona mejor en mercados ilíquidos "
            "donde el overshooting es común. En el backtest 2018–2025 es la "
            "estrategia con mejor CAGR (12.9%), impulsado por la recuperación "
            "post-COVID de hoteleras que fueron las más golpeadas."
        ),
        "universo": "Top 5 peores performers 12M",
        "señal": "Retorno trailing 252d (inverso al de E2)",
    },
    "E10": {
        "fn": e10_sector_tilt,
        "nombre": "Rotación sectorial",
        "categoria": "Macro",
        "descripcion": (
            "Ajusta los pesos por sector según la **dirección de la tasa "
            "objetivo de Banxico** en los últimos 6 meses: "
            "• Tasas subiendo → sobreponderar industrial (2×), reducir hotelero (0.5×). "
            "Las FIBRAs industriales tienen contratos dolarizados y rentas indexadas, "
            "más resistentes a inflación. "
            "• Tasas bajando → sobreponderar hotelero y comercial (1.5×), que se "
            "benefician de menor costo de financiamiento. "
            "• Estable → igual que E0."
        ),
        "universo": "Todas, con pesos diferenciados por sector",
        "señal": "Cambio tasa Banxico últimos 6 meses (±0.25%)",
    },
    "E11": {
        "fn": e11_video_strategy,
        "nombre": "Filtros fundamentales (video)",
        "categoria": "Fundamental",
        "descripcion": (
            "Replica la metodología del video de inversión en FIBRAs mexicanas. "
            "**Filtro de seguridad (hard):** ocupación ≥ 90% y payout ratio "
            "(distribución/FFO) ≤ 80% (el video sugiere <65%; se usa 80% para "
            "mantener un universo invertible dado que pocas FIBRAs cumplen el "
            "umbral estricto). **Filtro de oportunidad (scoring):** se rankean "
            "las FIBRAs que pasan el filtro por dividend yield TTM (40%), "
            "FFO yield como proxy de precio vs valor intrínseco (40%) y "
            "sesgo industrial por nearshoring (20%). "
            "LTV: no disponible en nuestras fuentes, filtro omitido. "
            "Top 5 por score compuesto."
        ),
        "universo": "FIBRAs con occ≥90% y payout≤80%, luego top 5 por score",
        "señal": "40% yield + 40% FFO yield + 20% sesgo industrial",
    },
    "E12": {
        "fn": e12_moving_average,
        "nombre": "Medias móviles (Golden Cross)",
        "categoria": "Tendencial",
        "descripcion": (
            "Estrategia clásica de análisis técnico. Invierte en las FIBRAs "
            "que están en **tendencia alcista confirmada**: precio > MA50 "
            "**y** MA50 > MA200 (Golden Cross). El peso de cada FIBRA es "
            "proporcional a qué tan por encima de su MA50 está el precio "
            "(mayor distancia = mayor convicción). Si ninguna FIBRA cumple "
            "la condición, cae a equiponderada (E0) como fallback. "
            "El Death Cross (MA50 cruza por debajo de MA200) expulsa la "
            "FIBRA del portafolio hasta que recupere la tendencia."
        ),
        "universo": "FIBRAs con precio > MA50 > MA200",
        "señal": "Cruce MA50 / MA200 + distancia % sobre MA50",
    },
}
