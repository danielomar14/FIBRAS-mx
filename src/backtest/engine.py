"""
Motor de backtest para FIBRAs mexicanas.

Parámetros de simulación:
- Rebalanceo: primer día hábil de cada trimestre (ene, abr, jul, oct)
- DRIP 100%: dividendos reinvertidos al precio de cierre del día ex-div
- Comisión: 0.25% × 1.16 IVA = 0.29% sobre monto operado
- Slippage: 0.10% sobre monto operado (compra más caro, vende más barato)
- Capital inicial: 200,000 MXN
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

COMMISSION   = 0.0025 * 1.16   # 0.29%
SLIPPAGE     = 0.001            # 0.10%
INITIAL_CAP  = 200_000.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga precios (wide), dividendos y métricas trimestrales.
    Devuelve (prices_wide, dividends, metrics).
    """
    prices_long = pd.read_parquet(PROCESSED / "precios_diarios.parquet")
    prices_wide = prices_long.pivot_table(
        index="date", columns="ticker", values="close"
    )
    prices_wide.index = pd.to_datetime(prices_wide.index)

    dividends = pd.read_parquet(PROCESSED / "distribuciones.parquet")
    dividends["date"] = pd.to_datetime(dividends["date"])

    metrics = pd.read_parquet(PROCESSED / "metricas_trimestrales.parquet")
    metrics["date"] = pd.to_datetime(metrics["date"])

    return prices_wide, dividends, metrics


def _rebalance_dates(prices_wide: pd.DataFrame,
                     start: str, end: str) -> list[pd.Timestamp]:
    """
    Primer día hábil de cada trimestre dentro del rango [start, end].
    """
    trading_days = prices_wide.loc[start:end].index
    quarter_starts = pd.date_range(start=start, end=end, freq="QS-JAN")  # ene, abr, jul, oct
    dates = []
    for qs in quarter_starts:
        after = trading_days[trading_days >= qs]
        if len(after):
            dates.append(after[0])
    return dates


def _execute_rebalance(
    holdings: dict[str, float],
    target_weights: dict[str, float],
    prices: pd.Series,
    cash: float,
) -> tuple[dict[str, float], float, float]:
    """
    Ejecuta el rebalanceo: vende lo que sobra, compra lo que falta.
    Aplica comisión + slippage sobre monto operado.
    Devuelve (new_holdings, new_cash, total_cost).
    """
    # Valor actual del portafolio
    portfolio_value = cash + sum(
        holdings.get(t, 0) * prices.get(t, np.nan)
        for t in holdings
        if not np.isnan(prices.get(t, np.nan))
    )

    # Sólo FIBRAs con precio disponible hoy
    available = {t: w for t, w in target_weights.items() if not np.isnan(prices.get(t, np.nan)) and w > 0}
    total_w = sum(available.values())
    if total_w == 0:
        return holdings, cash, 0.0
    norm_weights = {t: w / total_w for t, w in available.items()}

    # Valor objetivo por FIBRA
    target_value = {t: portfolio_value * w for t, w in norm_weights.items()}

    total_cost = 0.0
    new_holdings = dict(holdings)

    # Primero: liquidar posiciones no deseadas
    tickers_to_sell = set(new_holdings) - set(norm_weights)
    for t in tickers_to_sell:
        shares = new_holdings.pop(t, 0)
        if shares > 0 and not np.isnan(prices.get(t, np.nan)):
            exec_price = prices[t] * (1 - SLIPPAGE)
            proceeds = shares * exec_price
            cost = shares * prices[t] * (COMMISSION + SLIPPAGE)
            cash += proceeds - cost
            total_cost += cost

    # Luego: ajustar posiciones
    for t, tv in target_value.items():
        current_shares = new_holdings.get(t, 0)
        current_value  = current_shares * prices[t] if not np.isnan(prices.get(t, np.nan)) else 0
        delta_value    = tv - current_value

        if abs(delta_value) < 1.0:  # ignorar micro-ajustes < $1 MXN
            continue

        notional = abs(delta_value)
        cost = notional * (COMMISSION + SLIPPAGE)
        total_cost += cost

        if delta_value > 0:
            # Comprar
            exec_price  = prices[t] * (1 + SLIPPAGE)
            shares_buy  = (notional - cost) / exec_price
            new_holdings[t] = current_shares + shares_buy
            cash -= notional
        else:
            # Vender
            exec_price   = prices[t] * (1 - SLIPPAGE)
            shares_sell  = notional / exec_price
            shares_sell  = min(shares_sell, current_shares)
            new_holdings[t] = current_shares - shares_sell
            cash += shares_sell * exec_price - cost

    # Limpiar posiciones residuales
    new_holdings = {t: s for t, s in new_holdings.items() if s > 1e-6}
    return new_holdings, max(cash, 0.0), total_cost


# ── Motor principal ───────────────────────────────────────────────────────────

def run_backtest(
    strategy_fn,
    start: str = "2018-01-01",
    end: str   = "2025-12-31",
    initial_capital: float = INITIAL_CAP,
    prices_wide: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None   = None,
    metrics: pd.DataFrame | None     = None,
    banxico_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Corre el backtest de una estrategia y devuelve DataFrame con columnas:
        date, portfolio_value, cash, n_positions, rebalance, dividends_received, cost
    """
    if prices_wide is None or dividends is None or metrics is None:
        prices_wide, dividends, metrics = _load_data()

    # Recortar al rango pedido
    pw = prices_wide.loc[start:end].copy()
    # Forward fill hasta 5 días (fines de semana / feriados)
    pw = pw.ffill(limit=5)

    trading_days = pw.index.tolist()
    if not trading_days:
        return pd.DataFrame()

    rebalance_dates = set(_rebalance_dates(pw, start, end))

    # Índice de dividendos para lookup rápido
    div_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for _, row in dividends.iterrows():
        d = row["date"]
        if d not in div_by_date:
            div_by_date[d] = []
        div_by_date[d].append(row)

    holdings: dict[str, float] = {}
    cash = float(initial_capital)
    records = []

    for day in trading_days:
        prices_today = pw.loc[day].dropna()

        # ── Rebalanceo trimestral ─────────────────────────────────────────
        is_rebalance = day in rebalance_dates
        cost = 0.0
        if is_rebalance:
            weights = strategy_fn(
                date=day,
                prices_wide=pw.loc[:day],
                dividends=dividends,
                metrics=metrics,
                banxico_rates=banxico_rates,
            )
            holdings, cash, cost = _execute_rebalance(holdings, weights, prices_today, cash)

        # ── DRIP: reinversión de dividendos ───────────────────────────────
        divs_received = 0.0
        if day in div_by_date:
            for div_row in div_by_date[day]:
                t   = div_row["ticker"]
                amt = div_row["dividend"]
                if t in holdings and t in prices_today.index and prices_today[t] > 0:
                    div_cash = holdings[t] * amt
                    divs_received += div_cash
                    # Reinvertir: comprar más CBFIs al precio de hoy
                    new_shares = div_cash / prices_today[t]
                    holdings[t] = holdings.get(t, 0) + new_shares

        # ── Valor del portafolio ──────────────────────────────────────────
        equity = sum(
            holdings.get(t, 0) * prices_today.get(t, 0)
            for t in holdings
        )
        portfolio_value = equity + cash

        records.append({
            "date":              day,
            "portfolio_value":   portfolio_value,
            "cash":              cash,
            "n_positions":       len(holdings),
            "rebalance":         is_rebalance,
            "dividends_received":divs_received,
            "cost":              cost,
        })

    df = pd.DataFrame(records).set_index("date")
    log.info(
        f"Backtest {start}–{end}: "
        f"${df['portfolio_value'].iloc[-1]:,.0f} MXN "
        f"(CAGR {((df['portfolio_value'].iloc[-1]/initial_capital)**(365.25/((pw.index[-1]-pw.index[0]).days))-1)*100:.1f}%)"
    )
    return df
