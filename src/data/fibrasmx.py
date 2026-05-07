"""
Cliente Supabase para FibrasMX.

Fuente: base de datos pública de fibrasmx.com
Tablas:
  - fibras                -> master (id, ticker, sector, listing_date, dividend_frequency)
  - fibra_dividends       -> historial completo de distribuciones por ex_date
  - fibra_premium_metrics -> fundamentales trimestrales (ocupación, NOI, FFO)
"""

import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

SUPABASE_URL = "https://rahdmovgkchllkldwgpd.supabase.co"
SUPABASE_KEY = "sb_publishable_YQJFfqkcI86loGsYcf22bQ_mCPKsvwW"

_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}
BASE = f"{SUPABASE_URL}/rest/v1"

# Sólo las FIBRAs de nuestro universo de inversión
UNIVERSE = {
    "FUNO11", "FIBRAPL14", "FIBRAMQ12", "DANHOS13", "FMTY14",
    "FIHO12", "FINN13", "FSHOP13", "FIBRAUP18", "FNOVA17",
    "FPLUS16", "STORAGE18", "FSITES20", "EDUCA18", "NEXT25",
}


def _get(table: str, select: str = "*", order: str = "", page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        params = f"select={select}&limit={page_size}&offset={offset}"
        if order:
            params += f"&order={order}"
        r = requests.get(f"{BASE}/{table}?{params}", headers=_HEADERS, timeout=20)
        if r.status_code != 200:
            log.warning(f"Supabase {table} HTTP {r.status_code}: {r.text[:200]}")
            break
        batch: list[dict] = r.json()
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_fibras() -> pd.DataFrame:
    rows = _get(
        "fibras",
        select="id,ticker,name,sector,listing_date,dividend_frequency",
    )
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].str.upper()
    return df


def fetch_dividends(fibras_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if fibras_df is None:
        fibras_df = fetch_fibras()

    rows = _get(
        "fibra_dividends",
        select="fibra_id,ex_date,amount_per_unit,type,fiscal_year,fiscal_quarter,"
               "price_at_ex_date,fiscal_result_amount,capital_return_amount",
        order="ex_date.asc",
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    id_to_ticker = fibras_df.set_index("id")["ticker"].to_dict()
    df["ticker"] = df["fibra_id"].map(id_to_ticker)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df = df.rename(columns={"amount_per_unit": "dividend", "ex_date": "date"})
    df = df[df["ticker"].isin(UNIVERSE)].copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["source"] = "fibrasmx"

    keep = ["date", "ticker", "dividend", "type", "fiscal_year", "fiscal_quarter",
            "fiscal_result_amount", "capital_return_amount", "source"]
    return df[keep]


def fetch_metrics(fibras_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if fibras_df is None:
        fibras_df = fetch_fibras()

    rows = _get(
        "fibra_premium_metrics",
        select=(
            "fibra_id,period,"
            "distribution_per_cbfi,occupancy_portfolio,noi_margin,ffo_per_cbfi,"
            "nav_per_cbfi,book_value_per_cbfi,ltv_ratio,cbfis_outstanding,"
            "total_debt,property_appraised_value,total_assets"
        ),
        order="period.asc",
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    id_to_ticker = fibras_df.set_index("id")["ticker"].to_dict()
    df["ticker"] = df["fibra_id"].map(id_to_ticker)
    df = df[df["ticker"].isin(UNIVERSE)].copy()

    df["year"]    = df["period"].str[:4].astype(int)
    df["quarter"] = df["period"].str[5].astype(int)
    df["date"]    = pd.PeriodIndex(df["period"].str.replace("Q", "q", n=1), freq="Q").to_timestamp()
    df["source"]  = "fibrasmx"

    # nav_per_cbfi: usar book_value_per_cbfi como fallback si nav está vacío
    df["nav_per_cbfi"] = df["nav_per_cbfi"].combine_first(df["book_value_per_cbfi"])

    keep = [
        "date", "ticker", "period", "year", "quarter",
        "distribution_per_cbfi", "occupancy_portfolio", "noi_margin", "ffo_per_cbfi",
        "nav_per_cbfi", "ltv_ratio", "cbfis_outstanding",
        "total_debt", "property_appraised_value", "total_assets",
        "source",
    ]
    return df[keep].sort_values(["ticker", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    fibras = fetch_fibras()
    print(f"FIBRAs en Supabase: {len(fibras)}")

    divs = fetch_dividends(fibras)
    print(f"\nDividendos: {len(divs)} registros")
    print(divs.groupby("ticker")["dividend"].agg(["count", "sum"]).to_string())

    metrics = fetch_metrics(fibras)
    print(f"\nMétricas trimestrales: {len(metrics)} registros")
    print(metrics.groupby("ticker")["period"].agg(["count", "first", "last"]).to_string())
