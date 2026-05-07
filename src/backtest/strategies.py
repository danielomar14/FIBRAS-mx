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


# ── Registro de estrategias ───────────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {
    "E0":  {"fn": e0_equal_weight,      "nombre": "Equiponderada (benchmark)"},
    "E1":  {"fn": e1_large_cap,         "nombre": "Grandes (proxy cap)"},
    "E2":  {"fn": e2_momentum_12m,      "nombre": "Momentum 12M"},
    "E3":  {"fn": e3_momentum_3m,       "nombre": "Momentum 3M"},
    "E4":  {"fn": e4_low_volatility,    "nombre": "Baja volatilidad"},
    "E5":  {"fn": e5_high_yield,        "nombre": "Alto yield"},
    "E6":  {"fn": e6_quality_occupancy, "nombre": "Calidad (ocupación)"},
    "E7":  {"fn": e7_value_ffo,         "nombre": "Valor (FFO yield)"},
    "E8":  {"fn": e8_momentum_yield_combo,"nombre": "Momentum + Yield"},
    "E9":  {"fn": e9_anti_momentum,     "nombre": "Contrarian"},
    "E10": {"fn": e10_sector_tilt,      "nombre": "Rotación sectorial"},
}
