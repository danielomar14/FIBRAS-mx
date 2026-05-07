"""
Datos de benchmarks externos para comparaciones de mercado.

- IPC México (^MXX) vía yfinance
- CETES 28 días y Tasa objetivo Banxico vía Banxico SIE API (o fallback hardcoded)
- Market caps actuales de FIBRAs vía FibrasMX /api/prices
"""

import logging
from datetime import date

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────

BANXICO_BASE = "https://www.banxico.org.mx/SieAPIRest/service/v1"
SERIE_CETES28  = "SF43936"   # CETES 28 días, tasa primaria de colocación
SERIE_OBJETIVO = "SF61745"   # Tasa objetivo de política monetaria

FIBRASMX_PRICES_URL = "https://fibrasmx.com/api/prices"

# Fallback hardcoded: tasa CETES 28d mensual conocida (%, fuente Banxico)
# Usada cuando no hay token BMX disponible
_CETES_FALLBACK_MONTHLY = {
    # (año, mes): tasa %
    (2018,1):7.25,(2018,2):7.40,(2018,3):7.47,(2018,4):7.52,(2018,5):7.61,
    (2018,6):7.84,(2018,7):7.79,(2018,8):7.78,(2018,9):7.82,(2018,10):7.94,
    (2018,11):8.06,(2018,12):8.22,
    (2019,1):8.22,(2019,2):8.19,(2019,3):8.17,(2019,4):8.16,(2019,5):8.59,
    (2019,6):8.23,(2019,7):8.02,(2019,8):7.95,(2019,9):7.73,(2019,10):7.65,
    (2019,11):7.50,(2019,12):7.25,
    (2020,1):7.23,(2020,2):7.13,(2020,3):6.65,(2020,4):5.91,(2020,5):5.24,
    (2020,6):5.00,(2020,7):4.84,(2020,8):4.67,(2020,9):4.56,(2020,10):4.39,
    (2020,11):4.26,(2020,12):4.25,
    (2021,1):4.20,(2021,2):4.14,(2021,3):4.05,(2021,4):4.07,(2021,5):4.20,
    (2021,6):4.25,(2021,7):4.37,(2021,8):4.45,(2021,9):4.69,(2021,10):5.05,
    (2021,11):5.33,(2021,12):5.55,
    (2022,1):5.83,(2022,2):5.95,(2022,3):6.31,(2022,4):6.76,(2022,5):7.23,
    (2022,6):7.64,(2022,7):8.13,(2022,8):8.56,(2022,9):9.16,(2022,10):9.73,
    (2022,11):10.09,(2022,12):10.46,
    (2023,1):10.66,(2023,2):10.88,(2023,3):11.21,(2023,4):11.52,(2023,5):11.52,
    (2023,6):11.41,(2023,7):11.25,(2023,8):11.24,(2023,9):11.28,(2023,10):11.26,
    (2023,11):11.25,(2023,12):11.24,
    (2024,1):11.20,(2024,2):11.15,(2024,3):11.07,(2024,4):11.03,(2024,5):10.93,
    (2024,6):10.81,(2024,7):10.74,(2024,8):10.67,(2024,9):10.55,(2024,10):10.42,
    (2024,11):10.12,(2024,12):10.00,
    (2025,1):9.75,(2025,2):9.50,(2025,3):9.25,(2025,4):9.00,(2025,5):9.00,
    (2025,6):9.00,(2025,7):9.00,(2025,8):9.00,(2025,9):9.00,(2025,10):9.00,
    (2025,11):9.00,(2025,12):9.00,
}

# Tasa objetivo Banxico (cambios en reuniones del banco)
_BANXICO_RATE_CHANGES = [
    ("2018-01-01", 7.25), ("2018-02-08", 7.50), ("2018-03-29", 7.50),
    ("2018-05-17", 7.50), ("2018-06-21", 7.75), ("2018-08-02", 7.75),
    ("2018-09-27", 7.75), ("2018-11-15", 8.00), ("2018-12-20", 8.25),
    ("2019-02-07", 8.25), ("2019-03-28", 8.25), ("2019-05-16", 8.25),
    ("2019-06-27", 8.25), ("2019-08-15", 8.00), ("2019-09-26", 7.75),
    ("2019-11-14", 7.50), ("2019-12-19", 7.25),
    ("2020-02-13", 7.00), ("2020-03-20", 6.50), ("2020-04-21", 6.00),
    ("2020-05-14", 5.50), ("2020-06-25", 5.00), ("2020-08-13", 4.50),
    ("2020-09-24", 4.25), ("2021-02-11", 4.00), ("2021-03-25", 4.00),
    ("2021-05-13", 4.00), ("2021-06-24", 4.00), ("2021-08-12", 4.50),
    ("2021-09-30", 4.75), ("2021-11-11", 5.00), ("2021-12-16", 5.50),
    ("2022-02-10", 6.00), ("2022-03-24", 6.50), ("2022-05-12", 7.00),
    ("2022-06-23", 7.75), ("2022-08-11", 8.50), ("2022-09-29", 9.25),
    ("2022-11-10", 10.00), ("2022-12-15", 10.50),
    ("2023-02-09", 11.00), ("2023-03-30", 11.25), ("2023-05-18", 11.25),
    ("2023-06-22", 11.25), ("2023-08-10", 11.25), ("2023-09-28", 11.25),
    ("2023-11-09", 11.25), ("2023-12-14", 11.25),
    ("2024-02-08", 11.25), ("2024-03-21", 11.00), ("2024-05-09", 11.00),
    ("2024-06-27", 11.00), ("2024-08-08", 10.75), ("2024-09-26", 10.50),
    ("2024-11-14", 10.25), ("2024-12-19", 10.00),
    ("2025-02-06", 9.50), ("2025-03-27", 9.00), ("2025-05-15", 8.50),
    ("2025-06-26", 8.50), ("2025-08-14", 8.50), ("2025-12-31", 8.50),
]


# ── IPC México ────────────────────────────────────────────────────────────

def fetch_ipc(start: str = "2018-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Descarga el IPC México (^MXX) de yfinance. Devuelve df con columna 'close'."""
    try:
        df = yf.download("^MXX", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].rename(columns={"Close": "close"})
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        return df.dropna()
    except Exception as e:
        log.warning(f"Error descargando IPC: {e}")
        return pd.DataFrame()


# ── Banxico SIE ───────────────────────────────────────────────────────────

def fetch_banxico_serie(serie: str, start: str, end: str, token: str) -> pd.DataFrame:
    """
    Consulta una serie del Banxico SIE API.
    Requiere token BMX gratuito: https://www.banxico.org.mx/SieAPIRest/service/v1/token
    Devuelve df con columnas ['date', 'value'].
    """
    url = f"{BANXICO_BASE}/series/{serie}/datos/{start}/{end}"
    headers = {"Bmx-Token": token, "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Banxico SIE HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    series_data = data["bmx"]["series"][0]["datos"]
    df = pd.DataFrame(series_data)
    df["date"]  = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    df["value"] = pd.to_numeric(df["dato"].str.replace(",", ""), errors="coerce")
    return df[["date", "value"]].dropna().set_index("date").sort_index()


def fetch_cetes(start: str = "2018-01-01", end: str = "2025-12-31",
                token: str | None = None) -> pd.DataFrame:
    """
    Tasa CETES 28 días (% anual) como serie diaria.
    - Con token Banxico: consulta SF43936 (frecuencia semanal → ffill a diario)
    - Sin token: usa tabla mensual hardcoded → ffill a diario
    Devuelve df con columna 'cetes_pct', índice 'date'.
    """
    if token:
        try:
            df = fetch_banxico_serie(SERIE_CETES28, start, end, token)
            df = df.rename(columns={"value": "cetes_pct"})
            idx = pd.date_range(start, end, freq="D")
            df = df.reindex(idx).ffill().dropna()
            df.index.name = "date"
            return df
        except Exception as e:
            log.warning(f"Error Banxico SIE CETES, usando fallback: {e}")

    # Fallback: construir serie diaria desde tabla mensual
    idx = pd.date_range(start, end, freq="D")
    rates = []
    for d in idx:
        key = (d.year, d.month)
        rate = _CETES_FALLBACK_MONTHLY.get(
            key, _CETES_FALLBACK_MONTHLY.get((d.year, min(d.month, 12)), 9.0)
        )
        rates.append(rate)
    df = pd.DataFrame({"cetes_pct": rates}, index=idx)
    df.index.name = "date"
    return df


def fetch_banxico_rate(start: str = "2018-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """
    Tasa objetivo de política monetaria de Banxico como serie diaria.
    Construida desde la tabla hardcoded de cambios de tasa.
    Devuelve df con columna 'tasa_objetivo', índice 'date'.
    """
    changes = [(pd.Timestamp(d), r) for d, r in _BANXICO_RATE_CHANGES]
    idx = pd.date_range(start, end, freq="D")
    rates = []
    current_rate = changes[0][1]
    change_idx = 0
    for d in idx:
        while change_idx < len(changes) - 1 and changes[change_idx + 1][0] <= d:
            change_idx += 1
            current_rate = changes[change_idx][1]
        rates.append(current_rate)
    df = pd.DataFrame({"tasa_objetivo": rates}, index=idx)
    df.index.name = "date"
    return df.loc[start:end]


# ── Market caps ───────────────────────────────────────────────────────────

def fetch_market_caps() -> dict[str, float]:
    """
    Market caps actuales (MXN) de todas las FIBRAs desde FibrasMX /api/prices.
    Devuelve dict {ticker: marketCap}.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    try:
        r = requests.get(FIBRASMX_PRICES_URL, headers=headers, timeout=10)
        data = r.json().get("prices", {})
        return {
            ticker: info.get("marketCap", 0)
            for ticker, info in data.items()
            if info.get("marketCap") and info["marketCap"] > 0
        }
    except Exception as e:
        log.warning(f"Error obteniendo market caps: {e}")
        return {}


# ── Utilidades ────────────────────────────────────────────────────────────

def cetes_daily_return(cetes_pct: pd.Series) -> pd.Series:
    """Convierte tasa anual % a retorno diario compuesto."""
    return (1 + cetes_pct / 100) ** (1 / 365) - 1


def cumulative_value(daily_returns: pd.Series, initial: float = 200_000) -> pd.Series:
    """Valor acumulado a partir de retornos diarios."""
    return initial * (1 + daily_returns).cumprod()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("IPC México:")
    ipc = fetch_ipc()
    print(f"  {len(ipc)} días, {ipc.index[0].date()} → {ipc.index[-1].date()}")

    print("\nCETES (fallback):")
    cetes = fetch_cetes()
    print(f"  {len(cetes)} días, rango {cetes['cetes_pct'].min():.2f}% – {cetes['cetes_pct'].max():.2f}%")

    print("\nTasa Banxico:")
    banxico = fetch_banxico_rate()
    print(banxico.drop_duplicates().head(10).to_string())

    print("\nMarket caps:")
    caps = fetch_market_caps()
    total = sum(caps.values())
    for t, v in sorted(caps.items(), key=lambda x: -x[1]):
        print(f"  {t:<14} ${v/1e9:6.1f}B MXN  ({v/total*100:.1f}%)")
