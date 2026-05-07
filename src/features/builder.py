"""
build_feature_matrix: construye el DataFrame de features diarias.

Output shape: (n_days × n_tickers, n_features)
Index: MultiIndex (date, ticker)
Columns: feature_id (int 0-399)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.registry import FEATURE_REGISTRY, GENE_IDS

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

SECTOR_MAP: dict[str, str] = {
    "FUNO11":    "Diversificada",
    "FIBRAPL14": "Industrial",
    "FIBRAMQ12": "Industrial+Comercial",
    "DANHOS13":  "Comercial+Oficinas",
    "FMTY14":    "Mixta",
    "FIHO12":    "Hotelero",
    "FINN13":    "Hotelero",
    "FSHOP13":   "Comercial",
    "FIBRAUP18": "Industrial-PyMEs",
    "FNOVA17":   "Industrial+Mixta",
    "FPLUS16":   "Diversificada",
    "STORAGE18": "Self-Storage",
    "FSITES20":  "Infraestructura",
    "EDUCA18":   "Educativo",
    "NEXT25":    "Industrial",
}


def _load_banxico() -> pd.DataFrame | None:
    path = PROCESSED / "banxico_rates.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_ipc() -> pd.Series | None:
    path = PROCESSED / "ipc.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    col = "close" if "close" in df.columns else df.columns[0]
    return df[col]


def _div_yield_ttm(
    prices_wide: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula dividend yield TTM (trailing 12 meses) por ticker y fecha."""
    result = {}
    for ticker in prices_wide.columns:
        p = prices_wide[ticker].ffill(limit=10)
        d = dividends[dividends["ticker"] == ticker].copy()
        if d.empty:
            result[ticker] = pd.Series(np.nan, index=prices_wide.index)
            continue
        d = d.set_index("date")["dividend"].reindex(prices_wide.index, fill_value=0)
        ttm = d.rolling(252, min_periods=1).sum()
        result[ticker] = ttm / p.replace(0, np.nan)
    return pd.DataFrame(result)


def build_feature_matrix(
    prices_wide: pd.DataFrame,
    dividends: pd.DataFrame,
    metrics: pd.DataFrame,
    banxico: pd.DataFrame | None = None,
    ipc: pd.Series | None = None,
    gene_ids: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Construye la matriz de features.

    Parámetros
    ----------
    prices_wide : (date × ticker) con precios close
    dividends   : DataFrame con columnas [date, ticker, dividend]
    metrics     : DataFrame con columnas [date, ticker, ...]
    banxico     : DataFrame con columnas [date, rate, usdmxn?]
    ipc         : Series con precio de cierre del IPC
    gene_ids    : subset de gene IDs a calcular (default: todos 400)
    start, end  : recorte temporal

    Retorna
    -------
    DataFrame con MultiIndex (date, ticker) y columnas = gene_ids
    """
    if gene_ids is None:
        gene_ids = GENE_IDS

    if start is not None:
        prices_wide = prices_wide.loc[start:]
    if end is not None:
        prices_wide = prices_wide.loc[:end]

    prices_wide = prices_wide.ffill(limit=5)
    tickers = [t for t in prices_wide.columns if t in SECTOR_MAP]

    # Precalcular div yield TTM
    dy_wide = _div_yield_ttm(prices_wide[tickers], dividends)

    # Alinear banxico al índice de precios
    if banxico is not None:
        if "date" in banxico.columns:
            banxico = banxico.set_index("date")
        banxico = banxico.reindex(prices_wide.index, method="ffill")

    # Alinear IPC
    if ipc is not None:
        ipc = ipc.reindex(prices_wide.index, method="ffill")

    # Métricas por ticker (dict ticker -> DataFrame subset)
    metrics_by_ticker: dict[str, pd.DataFrame] = {}
    if metrics is not None and not metrics.empty:
        for ticker in tickers:
            sub = metrics[metrics["ticker"] == ticker].copy()
            if not sub.empty:
                sub = sub.set_index("date").sort_index()
                metrics_by_ticker[ticker] = sub

    all_records: list[pd.DataFrame] = []

    for ticker in tickers:
        prices = prices_wide[ticker].dropna()
        if prices.empty:
            continue

        m_df = metrics_by_ticker.get(ticker)
        dy   = dy_wide[ticker] if ticker in dy_wide.columns else None

        ticker_data: dict[int, pd.Series] = {}
        for gid in gene_ids:
            if gid not in FEATURE_REGISTRY:
                continue
            fdef = FEATURE_REGISTRY[gid]
            try:
                s = fdef.compute(prices, m_df, banxico, dy, ipc)
                if not isinstance(s, pd.Series):
                    s = pd.Series(s, index=prices.index)
                ticker_data[gid] = s.reindex(prices.index)
            except Exception:
                ticker_data[gid] = pd.Series(np.nan, index=prices.index)

        df_ticker = pd.DataFrame(ticker_data, index=prices.index)
        df_ticker.index.name = "date"
        df_ticker["ticker"] = ticker
        all_records.append(df_ticker)

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records)
    result = result.set_index("ticker", append=True)
    result = result.reorder_levels(["date", "ticker"])
    result.columns = [int(c) for c in result.columns]
    log.info(f"Feature matrix: {result.shape} (date×ticker, features)")
    return result


def load_and_build(
    gene_ids: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Carga datos del disco y construye la feature matrix."""
    prices_long = pd.read_parquet(PROCESSED / "precios_diarios.parquet")
    prices_wide = prices_long.pivot_table(index="date", columns="ticker", values="close")
    prices_wide.index = pd.to_datetime(prices_wide.index)

    dividends = pd.read_parquet(PROCESSED / "distribuciones.parquet")
    dividends["date"] = pd.to_datetime(dividends["date"])

    metrics = pd.read_parquet(PROCESSED / "metricas_trimestrales.parquet")
    metrics["date"] = pd.to_datetime(metrics["date"])

    banxico = _load_banxico()
    ipc     = _load_ipc()

    return build_feature_matrix(
        prices_wide, dividends, metrics, banxico, ipc,
        gene_ids=gene_ids, start=start, end=end,
    )
