"""
Métricas de performance para el backtest de FIBRAs.

Todas reciben una pd.Series de valor del portafolio con índice de fechas.
"""

import numpy as np
import pandas as pd


def cagr(portfolio: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    return (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / n_years) - 1


def annualized_volatility(portfolio: pd.Series) -> float:
    """Volatilidad anualizada de retornos diarios (252 días hábiles)."""
    returns = portfolio.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(252)


def max_drawdown(portfolio: pd.Series) -> float:
    """Máximo drawdown (valor negativo, e.g. -0.35 = -35%)."""
    rolling_max = portfolio.cummax()
    drawdown = (portfolio - rolling_max) / rolling_max
    return drawdown.min()


def drawdown_series(portfolio: pd.Series) -> pd.Series:
    """Serie diaria de drawdown (entre 0 y -1)."""
    rolling_max = portfolio.cummax()
    return (portfolio - rolling_max) / rolling_max


def sharpe(portfolio: pd.Series, rf_daily: pd.Series | None = None) -> float:
    """
    Sharpe ratio anualizado.
    rf_daily: serie de tasa libre de riesgo diaria (retorno, no %).
    Si None, Rf=0.
    """
    daily_returns = portfolio.pct_change().dropna()
    if rf_daily is not None:
        rf_aligned = rf_daily.reindex(daily_returns.index).ffill().fillna(0)
        excess = daily_returns - rf_aligned
    else:
        excess = daily_returns
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


def calmar(portfolio: pd.Series) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    mdd = max_drawdown(portfolio)
    if mdd == 0:
        return 0.0
    return cagr(portfolio) / abs(mdd)


def annual_returns(portfolio: pd.Series) -> pd.Series:
    """Retorno por año calendario."""
    return portfolio.resample("YE").last().pct_change().dropna()


def consistency(portfolio: pd.Series) -> float:
    """
    % de años calendario con retorno positivo.
    Con ventanas cortas, usa retornos trimestrales.
    """
    yr = annual_returns(portfolio)
    if len(yr) == 0:
        qr = portfolio.resample("QE").last().pct_change().dropna()
        if len(qr) == 0:
            return 0.0
        return (qr > 0).mean()
    return (yr > 0).mean()


def dividend_yield_trailing(dividends: pd.DataFrame, prices: pd.DataFrame,
                            ticker: str, as_of: pd.Timestamp, window_days: int = 365) -> float:
    """
    Yield anualizado trailing: suma dividendos últimos `window_days` / precio actual.
    dividends: df con columnas [date, ticker, dividend].
    prices: wide df con ticker como columna.
    """
    if ticker not in prices.columns:
        return 0.0
    start = as_of - pd.Timedelta(days=window_days)
    mask = (
        (dividends["ticker"] == ticker) &
        (dividends["date"] >= start) &
        (dividends["date"] <= as_of)
    )
    ttm_div = dividends.loc[mask, "dividend"].sum()
    price = prices.loc[:as_of, ticker].dropna()
    if price.empty or price.iloc[-1] == 0:
        return 0.0
    return ttm_div / price.iloc[-1]


def summary(portfolio: pd.Series, rf_daily: pd.Series | None = None) -> dict:
    """Diccionario con todas las métricas principales."""
    return {
        "CAGR":        round(cagr(portfolio) * 100, 2),
        "Volatilidad": round(annualized_volatility(portfolio) * 100, 2),
        "Sharpe":      round(sharpe(portfolio, rf_daily), 3),
        "Calmar":      round(calmar(portfolio), 3),
        "MaxDD":       round(max_drawdown(portfolio) * 100, 2),
        "Consistencia":round(consistency(portfolio) * 100, 1),
        "Valor final": round(portfolio.iloc[-1], 0),
    }
