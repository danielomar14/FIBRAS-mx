"""
Función de fitness para el GA (sin ML).

Por cada fecha de rebalanceo trimestral en el train:
  score_fibra = sum de feature_value[gene] para cada gene del individuo
  top_k FIBRAs con mayor score → retorno igual peso, 63 días
Sharpe de la serie de retornos, menos parsimony penalty.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.genetic.chromosome import Gene, Individual, GENE_ID_MAP

log = logging.getLogger(__name__)

PARSIMONY = 0.02   # penalización por gene adicional
H         = 63     # horizonte hold


def _rebalance_dates(price_index: pd.Index, start: str, end: str) -> list[pd.Timestamp]:
    days = price_index[(price_index >= start) & (price_index <= end)]
    quarter_starts = pd.date_range(start=start, end=end, freq="QS-JAN")
    result = []
    for qs in quarter_starts:
        after = days[days >= qs]
        if len(after):
            result.append(after[0])
    return result


def evaluate_individual(
    ind: Individual,
    feature_matrix: pd.DataFrame,
    prices_wide: pd.DataFrame,
    start: str = "2017-01-01",
    end: str   = "2023-12-31",
) -> float:
    """
    Evalúa el fitness de un individuo en el período [start, end].

    Parameters
    ----------
    feature_matrix : MultiIndex (date, ticker) × feature_id
    prices_wide    : (date × ticker) precios close
    """
    fids = [GENE_ID_MAP[g] for g in ind.genes]
    # Solo features que existen en la matrix
    available_fids = [f for f in fids if f in feature_matrix.columns]
    if not available_fids:
        return -np.inf

    price_idx = prices_wide.index
    rebalance_dates = _rebalance_dates(price_idx, start, end)
    if not rebalance_dates:
        return -np.inf

    quarterly_returns: list[float] = []

    for entry_date in rebalance_dates:
        if entry_date not in feature_matrix.index.get_level_values("date"):
            continue

        day_features = feature_matrix.xs(entry_date, level="date")[available_fids]
        if day_features.empty:
            continue

        # Score = suma de features (ya están en escala relativa por ticker)
        score = day_features.fillna(0).sum(axis=1)

        # Normalizar: rank 0-1 entre las FIBRAs disponibles
        if score.nunique() <= 1:
            continue
        score_rank = score.rank(pct=True)

        # Seleccionar top_k
        top_tickers = score_rank.nlargest(ind.top_k).index.tolist()
        if not top_tickers:
            continue

        # Retorno del portfolio en los siguientes H días
        try:
            after = prices_wide.index[prices_wide.index >= entry_date]
            if len(after) < 2:
                continue
            exit_idx = min(H, len(after) - 1)
            entry_p  = prices_wide.loc[after[0], top_tickers].dropna()
            exit_p   = prices_wide.loc[after[exit_idx], top_tickers].dropna()
            common   = entry_p.index.intersection(exit_p.index)
            if len(common) == 0:
                continue
            ret = float((exit_p.loc[common] / entry_p.loc[common] - 1).mean())
            quarterly_returns.append(ret)
        except Exception:
            continue

    if len(quarterly_returns) < 3:
        return -np.inf

    arr = np.array(quarterly_returns)
    sharpe = float(arr.mean() / arr.std(ddof=1) * np.sqrt(4)) if arr.std(ddof=1) > 0 else 0.0
    penalty = PARSIMONY * (len(ind.genes) - 1)
    return sharpe - penalty


def evaluate_period(
    ind: Individual,
    feature_matrix: pd.DataFrame,
    prices_wide: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """Evalúa el individuo en un período y retorna métricas detalladas."""
    fids = [GENE_ID_MAP[g] for g in ind.genes]
    available_fids = [f for f in fids if f in feature_matrix.columns]
    if not available_fids:
        return {
            "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan,
            "returns": [], "equity": pd.Series(dtype=float), "n_periods": 0,
        }

    price_idx = prices_wide.index
    rebalance_dates = _rebalance_dates(price_idx, start, end)

    records = []
    for entry_date in rebalance_dates:
        if entry_date not in feature_matrix.index.get_level_values("date"):
            continue
        day_features = feature_matrix.xs(entry_date, level="date")[available_fids].fillna(0)
        if day_features.empty:
            continue
        score = day_features.sum(axis=1)
        if score.nunique() <= 1:
            continue
        top_tickers = score.rank(pct=True).nlargest(ind.top_k).index.tolist()
        try:
            after   = prices_wide.index[prices_wide.index >= entry_date]
            exit_i  = min(H, len(after) - 1)
            ep      = prices_wide.loc[after[0], top_tickers].dropna()
            xp      = prices_wide.loc[after[exit_i], top_tickers].dropna()
            common  = ep.index.intersection(xp.index)
            if len(common) == 0: continue
            ret = float((xp.loc[common] / ep.loc[common] - 1).mean())
            records.append({"date": entry_date, "return": ret, "tickers": list(common)})
        except Exception:
            continue

    if not records:
        return {
            "sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan,
            "returns": [], "equity": pd.Series(dtype=float), "n_periods": 0,
        }

    df = pd.DataFrame(records).set_index("date")
    rets = df["return"].tolist()
    n = len(rets)
    equity = (1 + pd.Series(rets)).cumprod()
    arr = np.array(rets)
    std = arr.std(ddof=1) if n >= 2 else np.nan
    sharpe = float(arr.mean() / std * np.sqrt(4)) if (std and std > 0) else np.nan
    n_years = n / 4
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.25)) - 1)
    rolling_max = equity.cummax()
    max_dd = float(((equity - rolling_max) / rolling_max).min())

    return {
        "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd,
        "returns": rets, "equity": equity, "n_periods": n,
        "returns_dated": df["return"],  # pd.Series indexed by entry_date
        "details": df,                  # DataFrame con columnas return + tickers
    }
