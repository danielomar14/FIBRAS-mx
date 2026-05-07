"""
Multi-source data fetcher for Mexican FIBRAs.

Priority cascade:
  Prices    -> 1. yfinance (.MX)          2. manual CSV (data/raw/manual/)
  Dividends -> 1. FibrasMX Supabase       2. yfinance   3. manual CSV
  Metrics   -> 1. FibrasMX Supabase (quarterly: occupancy, LTV, NOI, FFO)
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.data.fibrasmx import fetch_dividends, fetch_fibras, fetch_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
MANUAL   = ROOT / "data" / "raw" / "manual"
PROCESSED = ROOT / "data" / "processed"

for p in (PROCESSED, MANUAL):
    p.mkdir(parents=True, exist_ok=True)

START = "2016-01-01"
END   = "2025-12-31"

FIBRAS: dict[str, dict] = {
    "FUNO11":    {"yahoo": "FUNO11.MX",    "ipo": "2011-03-18", "sector": "Diversificada"},
    "FIBRAPL14": {"yahoo": "FIBRAPL14.MX", "ipo": "2014-06-18", "sector": "Industrial"},
    "FIBRAMQ12": {"yahoo": "FIBRAMQ12.MX", "ipo": "2012-08-09", "sector": "Industrial+Comercial"},
    "DANHOS13":  {"yahoo": "DANHOS13.MX",  "ipo": "2013-10-08", "sector": "Comercial+Oficinas"},
    "FMTY14":    {"yahoo": "FMTY14.MX",    "ipo": "2014-03-27", "sector": "Mixta"},
    "FIHO12":    {"yahoo": "FIHO12.MX",    "ipo": "2012-10-18", "sector": "Hotelero"},
    "FINN13":    {"yahoo": "FINN13.MX",    "ipo": "2013-07-04", "sector": "Hotelero"},
    "FSHOP13":   {"yahoo": "FSHOP13.MX",   "ipo": "2013-10-17", "sector": "Comercial"},
    "FIBRAUP18": {"yahoo": "FIBRAUP18.MX", "ipo": "2018-11-22", "sector": "Industrial-PyMEs"},
    "FNOVA17":   {"yahoo": "FNOVA17.MX",   "ipo": "2017-10-03", "sector": "Industrial+Mixta"},
    "FPLUS16":   {"yahoo": "FPLUS16.MX",   "ipo": "2016-12-14", "sector": "Diversificada"},
    "STORAGE18": {"yahoo": "STORAGE18.MX", "ipo": "2018-10-12", "sector": "Self-Storage"},
    "FSITES20":  {"yahoo": "FSITES20.MX",  "ipo": "2020-08-13", "sector": "Infraestructura"},
    "EDUCA18":   {"yahoo": "EDUCA18.MX",   "ipo": "2018-08-03", "sector": "Educativo"},
    "NEXT25":    {"yahoo": "NEXT25.MX",    "ipo": "2025-01-01", "sector": "Industrial"},
}

# ── Layer 1: yfinance prices ───────────────────────────────────────────────

def _fetch_yfinance_prices(ticker_bmv: str, meta: dict) -> pd.DataFrame | None:
    yahoo = meta["yahoo"]
    start = max(START, meta["ipo"])
    try:
        df = yf.download(
            yahoo, start=start, end=END,
            auto_adjust=False,
            actions=False,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker_bmv
        df["source"] = "yfinance"
        return df
    except Exception as e:
        log.warning(f"yfinance price error {yahoo}: {e}")
        return None


def _fetch_yfinance_dividends(ticker_bmv: str, meta: dict) -> pd.DataFrame | None:
    yahoo = meta["yahoo"]
    start = max(START, meta["ipo"])
    try:
        t = yf.Ticker(yahoo)
        divs = t.dividends
        if divs is None or divs.empty:
            return None
        divs = divs.loc[start:END].copy()
        if divs.empty:
            return None
        df = divs.reset_index()
        df.columns = ["date", "dividend"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = ticker_bmv
        df["source"] = "yfinance"
        return df[["date", "ticker", "dividend", "source"]]
    except Exception as e:
        log.warning(f"yfinance dividend error {yahoo}: {e}")
        return None


def _load_manual(ticker_bmv: str, kind: str) -> pd.DataFrame | None:
    suffix = "_dividendos" if kind == "divs" else ""
    path = MANUAL / f"{ticker_bmv}{suffix}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df["ticker"] = ticker_bmv
        df["source"] = "manual"
        return df
    except Exception as e:
        log.warning(f"Manual CSV error {path}: {e}")
        return None

# ── Orchestrator ──────────────────────────────────────────────────────────

def fetch_all(force: bool = False) -> dict:
    prices_path  = PROCESSED / "precios_diarios.parquet"
    divs_path    = PROCESSED / "distribuciones.parquet"
    metrics_path = PROCESSED / "metricas_trimestrales.parquet"
    report_path  = PROCESSED / "data_quality_report.json"

    if all(p.exists() for p in [prices_path, divs_path, metrics_path]) and not force:
        log.info("Archivos procesados ya existen. Usa force=True para re-descargar.")
        return _load_report(report_path)

    # ── Step 1: Dividendos y métricas desde FibrasMX Supabase ────────────
    log.info("Descargando dividendos y métricas desde FibrasMX Supabase…")
    fibras_master = fetch_fibras()
    sb_divs       = fetch_dividends(fibras_master)
    sb_metrics    = fetch_metrics(fibras_master)
    log.info(f"  Supabase dividendos: {len(sb_divs)} registros")
    log.info(f"  Supabase métricas:   {len(sb_metrics)} registros")

    # Agrupar dividendos Supabase por ticker para fallback rápido
    sb_divs_by_ticker: dict[str, pd.DataFrame] = {}
    if not sb_divs.empty:
        for ticker, grp in sb_divs.groupby("ticker"):
            sb_divs_by_ticker[ticker] = grp

    # ── Step 2: Precios desde yfinance ───────────────────────────────────
    all_prices: list[pd.DataFrame] = []
    all_divs:   list[pd.DataFrame] = []
    coverage:   dict[str, dict]    = {}

    for ticker, meta in tqdm(FIBRAS.items(), desc="Precios (yfinance)"):
        # Prices
        prices = _fetch_yfinance_prices(ticker, meta)
        source_price = "yfinance"
        if prices is None or prices.empty:
            prices = _load_manual(ticker, "prices")
            source_price = "manual" if prices is not None else "missing"

        # Dividends: Supabase first, then yfinance, then manual
        if ticker in sb_divs_by_ticker and not sb_divs_by_ticker[ticker].empty:
            divs = sb_divs_by_ticker[ticker][["date", "ticker", "dividend", "source"]].copy()
            # filter to our backtest window
            divs = divs[(divs["date"] >= START) & (divs["date"] <= END)]
            source_divs = "fibrasmx"
        else:
            divs = _fetch_yfinance_dividends(ticker, meta)
            source_divs = "yfinance" if divs is not None else "missing"
            if divs is None:
                divs = _load_manual(ticker, "divs")
                source_divs = "manual" if divs is not None else "missing"

        # Coverage stats
        expected_start = pd.Timestamp(max(START, meta["ipo"]))
        expected_end   = pd.Timestamp(END)
        expected_days  = len(pd.bdate_range(expected_start, expected_end))
        actual_days    = len(prices) if prices is not None else 0

        coverage[ticker] = {
            "sector":                meta["sector"],
            "ipo":                   meta["ipo"],
            "source_prices":         source_price,
            "source_divs":           source_divs,
            "expected_trading_days": expected_days,
            "actual_rows":           actual_days,
            "coverage_pct":          round(actual_days / expected_days * 100, 1) if expected_days else 0,
            "dividend_count":        len(divs) if divs is not None else 0,
        }

        if prices is not None and not prices.empty:
            all_prices.append(prices)
        if divs is not None and not divs.empty:
            all_divs.append(divs)

    # ── Step 3: Guardar ───────────────────────────────────────────────────
    if all_prices:
        df_prices = pd.concat(all_prices).sort_index()
        df_prices.to_parquet(prices_path)
        log.info(f"Precios guardados -> {prices_path}  ({len(df_prices):,} filas)")

    if all_divs:
        df_divs = pd.concat(all_divs, ignore_index=True).sort_values(["ticker", "date"])
        df_divs.to_parquet(divs_path)
        log.info(f"Dividendos guardados -> {divs_path}  ({len(df_divs):,} filas)")

    if not sb_metrics.empty:
        sb_metrics.to_parquet(metrics_path)
        log.info(f"Métricas guardadas -> {metrics_path}  ({len(sb_metrics):,} filas)")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)
    log.info(f"Reporte de calidad -> {report_path}")

    return coverage


def _load_report(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "precios_diarios.parquet")


def load_dividends() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "distribuciones.parquet")


def load_metrics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "metricas_trimestrales.parquet")


if __name__ == "__main__":
    summary = fetch_all(force=True)
    print("\n--- Cobertura por FIBRA ---")
    for ticker, info in summary.items():
        pct  = info.get("coverage_pct", 0)
        flag = "OK" if pct >= 80 else "ALERTA"
        src  = info.get("source_divs", "?")
        print(f"  {ticker:<12} precios={pct:5.1f}%  divs={info.get('dividend_count',0):3d} [{src}]  [{flag}]")
