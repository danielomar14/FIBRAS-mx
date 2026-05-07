"""
Data quality validator for FIBRA price and dividend series.

Checks:
  - Coverage >= 80% of expected trading days
  - No negative prices
  - No gaps > 5 consecutive business days
  - At least 3 dividend payments per year since IPO
"""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
MAX_GAP_DAYS = 5
MIN_COVERAGE = 80.0
MIN_DIVS_PER_YEAR = 3


def validate_prices(df: pd.DataFrame) -> dict:
    """
    Validate price DataFrame (index=date, columns include 'close', 'ticker').
    Returns dict: ticker -> list of issues.
    """
    issues: dict[str, list[str]] = {}

    for ticker, group in df.groupby("ticker"):
        ticker_issues: list[str] = []
        prices = group["close"].dropna()

        # Negative prices
        neg = (prices < 0).sum()
        if neg > 0:
            ticker_issues.append(f"{neg} filas con precio negativo")

        # Gaps > MAX_GAP_DAYS business days
        idx = pd.DatetimeIndex(group.index)
        bday_gaps = pd.Series(idx).diff().dt.days.dropna()
        large_gaps = bday_gaps[bday_gaps > MAX_GAP_DAYS + 2]  # +2 for weekends
        if not large_gaps.empty:
            ticker_issues.append(
                f"{len(large_gaps)} huecos > {MAX_GAP_DAYS} días hábiles"
            )

        # Zero-price rows (possible data error)
        zeros = (prices == 0).sum()
        if zeros > 0:
            ticker_issues.append(f"{zeros} filas con precio = 0")

        issues[ticker] = ticker_issues

    return issues


def validate_dividends(df_divs: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    """
    Validate dividend DataFrame.
    Returns dict: ticker -> list of issues.
    """
    issues: dict[str, list[str]] = {}

    tickers_in_prices = df_prices["ticker"].unique() if "ticker" in df_prices.columns else []

    for ticker in tickers_in_prices:
        ticker_issues: list[str] = []
        sub = df_divs[df_divs["ticker"] == ticker] if "ticker" in df_divs.columns else pd.DataFrame()

        if sub.empty:
            ticker_issues.append("Sin dividendos registrados")
            issues[ticker] = ticker_issues
            continue

        # Check negative dividends
        if "dividend" in sub.columns:
            neg_divs = (sub["dividend"] < 0).sum()
            if neg_divs:
                ticker_issues.append(f"{neg_divs} dividendos negativos")

        # Check annual frequency
        if "date" in sub.columns:
            sub = sub.copy()
            sub["year"] = pd.to_datetime(sub["date"]).dt.year
            annual = sub.groupby("year").size()
            low_years = annual[annual < MIN_DIVS_PER_YEAR]
            if not low_years.empty:
                years_str = ", ".join(str(y) for y in low_years.index)
                ticker_issues.append(
                    f"< {MIN_DIVS_PER_YEAR} pagos en año(s): {years_str}"
                )

        issues[ticker] = ticker_issues

    return issues


def run_full_validation() -> dict:
    """
    Load processed files and run all validations.
    Returns combined report.
    """
    prices_path = PROCESSED / "precios_diarios.parquet"
    divs_path   = PROCESSED / "distribuciones.parquet"
    report_path = PROCESSED / "data_quality_report.json"

    if not prices_path.exists():
        log.error("No se encontró precios_diarios.parquet — ejecuta fetcher.py primero")
        return {}

    df_prices = pd.read_parquet(prices_path)
    df_divs   = pd.read_parquet(divs_path) if divs_path.exists() else pd.DataFrame()

    # Load existing coverage report
    coverage: dict = {}
    if report_path.exists():
        with open(report_path) as f:
            coverage = json.load(f)

    price_issues = validate_prices(df_prices)
    div_issues   = validate_dividends(df_divs, df_prices)

    # Merge into coverage report
    for ticker in coverage:
        coverage[ticker]["price_issues"] = price_issues.get(ticker, [])
        coverage[ticker]["div_issues"]   = div_issues.get(ticker, [])
        coverage[ticker]["status"] = (
            "OK" if (
                coverage[ticker].get("coverage_pct", 0) >= MIN_COVERAGE
                and not price_issues.get(ticker)
            ) else "ALERTA"
        )

    # Save updated report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    return coverage


def print_report(report: dict) -> None:
    print(f"\n{'Ticker':<12} {'Cobertura':>10} {'Divs':>5} {'Estado':<8} Problemas")
    print("-" * 70)
    for ticker, info in report.items():
        pct    = info.get("coverage_pct", 0)
        divs   = info.get("dividend_count", 0)
        status = info.get("status", "?")
        issues = info.get("price_issues", []) + info.get("div_issues", [])
        issue_str = " | ".join(issues) if issues else "-"
        print(f"  {ticker:<12} {pct:8.1f}%  {divs:4d}  {status:<8} {issue_str}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    report = run_full_validation()
    print_report(report)
