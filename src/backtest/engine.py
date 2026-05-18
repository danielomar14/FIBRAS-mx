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
    prices_fallback: pd.Series | None = None,
) -> tuple[dict[str, float], float, float]:
    """
    Ejecuta el rebalanceo: vende lo que sobra, compra lo que falta.
    Aplica comisión + slippage sobre monto operado.
    prices_fallback: precios con ffill largo, usados para vender posiciones
                     ilíquidas que no cotizan hoy en prices (evita pérdida de capital).
    Devuelve (new_holdings, new_cash, total_cost).
    """
    def _price(t: str) -> float:
        """Precio efectivo: primario o fallback; NaN si no hay ninguno."""
        p = prices.get(t, np.nan)
        if not np.isnan(p):
            return p
        if prices_fallback is not None:
            return prices_fallback.get(t, np.nan)
        return np.nan

    # Valor actual del portafolio
    portfolio_value = cash + sum(
        holdings.get(t, 0) * _price(t)
        for t in holdings
        if not np.isnan(_price(t))
    )

    # Sólo FIBRAs con precio disponible hoy (solo en prices primarios para comprar)
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
        p = _price(t)
        if shares > 0 and not np.isnan(p):
            # Vende al precio disponible (primario o fallback)
            exec_price = p * (1 - SLIPPAGE)
            proceeds   = shares * exec_price
            cost       = shares * p * (COMMISSION + SLIPPAGE)
            cash += proceeds - cost
            total_cost += cost
        # Si p es NaN (FIBRA deslistada sin precio conocido), la posición
        # se elimina sin recuperar efectivo (pérdida total, caso extremo).

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

def _monthly_contribution_dates(trading_days: list, contribution: float) -> dict:
    """Primer día hábil de cada mes → monto de aportación."""
    if contribution <= 0:
        return {}
    seen = set()
    result = {}
    for d in trading_days:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            result[d] = contribution
    return result


def run_backtest(
    strategy_fn,
    start: str = "2021-01-01",
    end: str   = "2025-12-31",
    initial_capital: float = INITIAL_CAP,
    monthly_contribution: float = 10_000.0,
    prices_wide: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None   = None,
    metrics: pd.DataFrame | None     = None,
    banxico_rates: pd.DataFrame | None = None,
    return_positions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corre el backtest de una estrategia. Devuelve DataFrame con columnas:
        date, portfolio_value, cash, invested_capital, n_positions,
        rebalance, contribution, dividends_received, cost

    Si return_positions=True, devuelve (df, pos_df) donde pos_df tiene
    columnas {ticker}_mxn y efectivo_mxn por día hábil.

    monthly_contribution: monto en MXN que se inyecta el primer día hábil
        de cada mes (simula aportación periódica). Se suma al cash y se
        despliega en el siguiente rebalanceo trimestral.
    """
    if prices_wide is None or dividends is None or metrics is None:
        prices_wide, dividends, metrics = _load_data()

    pw = prices_wide.loc[start:end].copy()
    pw_trade = pw.ffill(limit=5)
    pw_value = pw.ffill(limit=60)

    trading_days = pw_trade.index.tolist()
    if not trading_days:
        return (pd.DataFrame(), pd.DataFrame()) if return_positions else pd.DataFrame()

    rebalance_dates  = set(_rebalance_dates(pw_trade, start, end))
    contribution_map = _monthly_contribution_dates(trading_days, monthly_contribution)

    div_by_date: dict[pd.Timestamp, list] = {}
    for _, row in dividends.iterrows():
        d = row["date"]
        if d not in div_by_date:
            div_by_date[d] = []
        div_by_date[d].append(row)

    holdings: dict[str, float] = {}
    cash            = float(initial_capital)
    invested_capital = float(initial_capital)
    records = []
    pos_records = []

    for day in trading_days:
        prices_trade = pw_trade.loc[day].dropna()
        prices_val   = pw_value.loc[day].dropna()

        # ── Aportación mensual ────────────────────────────────────────────
        contribution = contribution_map.get(day, 0.0)
        if contribution > 0:
            cash             += contribution
            invested_capital += contribution

        # ── Rebalanceo trimestral ─────────────────────────────────────────
        is_rebalance = day in rebalance_dates
        cost = 0.0
        if is_rebalance:
            weights = strategy_fn(
                date=day,
                prices_wide=pw_trade.loc[:day],
                dividends=dividends,
                metrics=metrics,
                banxico_rates=banxico_rates,
            )
            holdings, cash, cost = _execute_rebalance(
                holdings, weights, prices_trade, cash, prices_val
            )

        # ── DRIP: reinversión de dividendos ───────────────────────────────
        divs_received = 0.0
        if day in div_by_date:
            for div_row in div_by_date[day]:
                t   = div_row["ticker"]
                amt = div_row["dividend"]
                if t in holdings and t in prices_trade.index and prices_trade[t] > 0:
                    div_cash = holdings[t] * amt
                    divs_received += div_cash
                    new_shares = div_cash / prices_trade[t]
                    holdings[t] = holdings.get(t, 0) + new_shares

        # ── Valor del portafolio ──────────────────────────────────────────
        fibra_vals: dict[str, float] = {}
        equity = 0.0
        for t, shares in holdings.items():
            p = prices_val.get(t, 0)
            val = shares * p
            fibra_vals[t] = val
            equity += val
        portfolio_value = equity + cash

        records.append({
            "date":               day,
            "portfolio_value":    portfolio_value,
            "invested_capital":   invested_capital,
            "cash":               cash,
            "n_positions":        len(holdings),
            "rebalance":          is_rebalance,
            "contribution":       contribution,
            "dividends_received": divs_received,
            "cost":               cost,
        })

        if return_positions:
            pos_row = {"date": day, "efectivo_mxn": cash}
            pos_row.update({f"{t}_mxn": v for t, v in fibra_vals.items()})
            pos_records.append(pos_row)

    df = pd.DataFrame(records).set_index("date")
    total_invested = df["invested_capital"].iloc[-1]
    total_return = (df["portfolio_value"].iloc[-1] / total_invested - 1) * 100
    log.info(
        f"Backtest {start}–{end}: "
        f"${df['portfolio_value'].iloc[-1]:,.0f} MXN "
        f"(aportado ${total_invested:,.0f}, retorno total {total_return:.1f}%)"
    )

    if return_positions:
        pos_df = pd.DataFrame(pos_records).set_index("date").fillna(0.0)
        return df, pos_df
    return df
