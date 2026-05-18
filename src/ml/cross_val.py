"""
Walk-forward cross-validation para Experimento 2A.

Pipeline:
  - Label: retorno H-day-ahead de cada FIBRA, rank-normalizado 0-1 entre tickers
  - CV: ventana expandible, mín 252 días, paso 63 días
  - Por cada fold: entrenar en ≤ t, predecir [t+1..t+63], seleccionar top-k FIBRAs
  - Retorno del portfolio: igual peso, hold 63 días
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

H = 63  # horizonte de predicción (días)


def make_labels(
    prices_wide: pd.DataFrame,
    H: int = H,
) -> pd.DataFrame:
    """
    Retorno H días hacia adelante, rank-normalizado 0-1 entre tickers por fecha.
    Shape: (date, ticker)
    """
    fwd = prices_wide.pct_change(H).shift(-H)
    ranks = fwd.rank(axis=1, pct=True)
    return ranks


def _portfolio_return(
    prices_wide: pd.DataFrame,
    top_k_tickers: list[str],
    entry_date: pd.Timestamp,
    H: int,
) -> float:
    """Retorno de igual peso del portfolio seleccionado desde entry_date en H días."""
    try:
        after = prices_wide.index[prices_wide.index >= entry_date]
        if len(after) < 2:
            return np.nan
        exit_idx = min(H, len(after) - 1)
        entry_prices = prices_wide.loc[after[0], top_k_tickers]
        exit_prices  = prices_wide.loc[after[exit_idx], top_k_tickers]
        rets = (exit_prices / entry_prices - 1).dropna()
        return float(rets.mean()) if len(rets) > 0 else np.nan
    except Exception:
        return np.nan


def walk_forward_cv(
    feature_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    prices_wide: pd.DataFrame,
    model,
    top_k: int = 5,
    min_train_days: int = 252,
    step_days: int = H,
) -> list[float]:
    """
    Walk-forward CV dentro del conjunto de entrenamiento.

    Retorna lista de retornos trimestrales del portfolio (uno por fold).
    """
    dates = sorted(feature_matrix.index.get_level_values("date").unique())
    returns = []

    for i in range(min_train_days, len(dates) - step_days, step_days):
        train_dates = dates[:i]
        pred_date   = dates[i]

        # Build train set
        train_idx = feature_matrix.index.get_level_values("date").isin(train_dates)
        X_train = feature_matrix.loc[train_idx].dropna(how="all", axis=1)
        y_train = labels.stack().reindex(X_train.index)
        valid   = y_train.notna() & X_train.notna().all(axis=1)
        if valid.sum() < 50:
            continue

        X_tr = X_train.loc[valid].fillna(0)
        y_tr = y_train.loc[valid]
        cols = X_tr.columns

        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            log.debug(f"Model fit failed at {pred_date}: {e}")
            continue

        # Predict on pred_date
        if pred_date not in feature_matrix.index.get_level_values("date"):
            continue
        X_pred_all = feature_matrix.xs(pred_date, level="date").reindex(columns=cols).fillna(0)
        if X_pred_all.empty:
            continue

        scores = pd.Series(model.predict(X_pred_all), index=X_pred_all.index)
        top_tickers = scores.nlargest(top_k).index.tolist()

        ret = _portfolio_return(prices_wide, top_tickers, pred_date, H)
        if not np.isnan(ret):
            returns.append(ret)

    return returns


def sharpe_from_returns(rets: list[float], annualize: float = 4.0) -> float:
    """Sharpe anualizado de retornos trimestrales."""
    if len(rets) < 3:
        return -np.inf
    arr = np.array(rets)
    mu  = arr.mean()
    sd  = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(mu / sd * np.sqrt(annualize))


def evaluate_on_period(
    feature_matrix: pd.DataFrame,
    labels: pd.DataFrame,
    prices_wide: pd.DataFrame,
    model,
    period_dates: list,
    train_dates: list,
    top_k: int = 5,
) -> dict:
    """Evalúa el modelo (ya entrenado en train_dates) sobre period_dates."""
    results = []
    step = H

    for i in range(0, len(period_dates) - step, step):
        entry = period_dates[i]
        if entry not in feature_matrix.index.get_level_values("date"):
            continue
        X_pred = feature_matrix.xs(entry, level="date")
        X_pred = X_pred.reindex(columns=feature_matrix.columns).fillna(0)
        if X_pred.empty:
            continue
        try:
            scores = pd.Series(model.predict(X_pred), index=X_pred.index)
        except Exception:
            continue
        top_tickers = scores.nlargest(top_k).index.tolist()
        ret = _portfolio_return(prices_wide, top_tickers, entry, step)
        if not np.isnan(ret):
            results.append({"date": entry, "return": ret, "tickers": top_tickers})

    if not results:
        return {"returns": [], "sharpe": -np.inf, "cagr": np.nan, "max_dd": np.nan}

    df = pd.DataFrame(results).set_index("date")
    rets = df["return"].tolist()
    equity = (1 + pd.Series(rets)).cumprod()
    n_years = len(rets) / 4
    cagr = float(equity.iloc[-1] ** (1 / max(n_years, 0.25)) - 1) if n_years > 0 else np.nan
    rolling_max = equity.cummax()
    max_dd = float(((equity - rolling_max) / rolling_max).min())

    return {
        "returns":        rets,
        "returns_dated":  df["return"],   # pd.Series indexed by entry_date
        "sharpe":         sharpe_from_returns(rets),
        "cagr":           cagr,
        "max_dd":         max_dd,
        "equity":         equity,
        "details":        df,
    }
