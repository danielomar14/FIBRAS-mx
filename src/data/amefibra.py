"""
Scraper de AMEFIBRA (https://amefibra.com) para doble verificación de datos.

Extrae:
  1. Lista de FIBRAs asociadas con sector y metadata
  2. Catálogo de reportes trimestrales/anuales disponibles (PDFs)
  3. (Futuro) Parseo de PDFs para validar distribuciones y ocupación

El sitio es WordPress SSR — sin SPA, sin token requerido.
robots.txt permite scraping (solo bloquea /wp-admin/).
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

BASE_URL  = "https://amefibra.com"
HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}

# Mapa slug AMEFIBRA → ticker BMV
SLUG_TO_TICKER = {
    "fibra-uno-funo":          "FUNO11",
    "fibrahotel":              "FIHO12",
    "fibra-macquarie":         "FIBRAMQ12",
    "fibra-soma":              None,          # No en nuestro universo
    "fibra-inn":               "FINN13",
    "fibrashop":               "FSHOP13",
    "fibra-danhos":            "DANHOS13",
    "fibra-prologis":          "FIBRAPL14",
    "fibramty-fibra-monterrey":"FMTY14",
    "fibra-nova":              "FNOVA17",
    "fibra-plus":              "FPLUS16",
    "fibra-upsite":            "FIBRAUP18",
    "fibra-educa":             "EDUCA18",
    "fibra-storage":           "STORAGE18",
    "fibra-next":              "NEXT25",
}


def _get(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "lxml")
            log.warning(f"HTTP {r.status_code}: {url}")
        except Exception as e:
            log.warning(f"Error en intento {attempt+1}: {e}")
        time.sleep(1)
    return None


# ── 1. Lista de FIBRAs asociadas ──────────────────────────────────────────

def fetch_asociados() -> pd.DataFrame:
    """
    Scrape /asociados/ para obtener nombre, slug, URL y sector de cada FIBRA.
    """
    soup = _get(f"{BASE_URL}/asociados/")
    if soup is None:
        return pd.DataFrame()

    records = []
    # Los portfolio-items tienen enlaces con /portfolio-item/{slug}/
    links = soup.find_all("a", href=lambda h: h and "/portfolio-item/" in h)
    seen = set()
    for a in links:
        href = a.get("href", "")
        slug = href.rstrip("/").split("/")[-1]
        if slug in seen:
            continue
        seen.add(slug)

        name = a.get_text(strip=True)
        if not name:
            img = a.find("img")
            name = img.get("alt", slug) if img else slug

        ticker = SLUG_TO_TICKER.get(slug)
        records.append({
            "slug":   slug,
            "nombre": name,
            "ticker": ticker,
            "url":    href,
        })

    # Enriquecer con el sector de cada página individual
    for rec in records:
        time.sleep(0.3)
        detail = _get(rec["url"])
        if detail is None:
            continue
        # El sector aparece como categoría de portfolio
        cats = detail.find_all("a", href=lambda h: h and "/portfolio-category/" in h)
        rec["sector_amefibra"] = ", ".join(c.get_text(strip=True) for c in cats) or None

        # Buscar sitio web corporativo
        links_ext = detail.find_all("a", href=lambda h: h and h.startswith("http") and "amefibra" not in h)
        websites = [l["href"] for l in links_ext if "twitter" not in l["href"]
                    and "facebook" not in l["href"] and "linkedin" not in l["href"]
                    and "instagram" not in l["href"] and "youtube" not in l["href"]]
        rec["sitio_web"] = websites[0] if websites else None

    df = pd.DataFrame(records)
    log.info(f"AMEFIBRA asociados: {len(df)} FIBRAs encontradas")
    return df


# ── 2. Catálogo de reportes ───────────────────────────────────────────────

def fetch_reportes() -> pd.DataFrame:
    """
    Scrape /reportes-de-fibras/ para listar todos los reportes PDF disponibles.
    Devuelve df con: nombre, fecha, url_descarga, fibra (si detectable).
    """
    soup = _get(f"{BASE_URL}/reportes-de-fibras/")
    if soup is None:
        return pd.DataFrame()

    records = []
    # WordPress Download Manager genera links con /download/ o class wpdm
    download_links = soup.find_all("a", href=lambda h: h and (
        "/download/" in h or "wpdm" in h or ".pdf" in h.lower()
    ))
    for a in download_links:
        href  = a.get("href", "")
        title = a.get_text(strip=True) or a.get("title", "")
        if not title:
            continue

        # Intentar detectar ticker en el nombre del reporte
        ticker = None
        for slug, t in SLUG_TO_TICKER.items():
            if t and (t.lower() in title.lower() or
                      slug.replace("-", "").lower() in title.lower().replace(" ", "")):
                ticker = t
                break

        records.append({
            "nombre":      title,
            "ticker":      ticker,
            "url_reporte": href,
        })

    # Buscar también en los posts de noticias que pueden contener reportes
    posts = soup.find_all(["li", "div"], class_=lambda c: c and "wpdm" in str(c).lower())
    for p in posts:
        a = p.find("a", href=True)
        if a:
            title = p.get_text(strip=True)[:100]
            records.append({
                "nombre":      title,
                "ticker":      None,
                "url_reporte": a["href"],
            })

    df = pd.DataFrame(records).drop_duplicates(subset=["url_reporte"])
    log.info(f"AMEFIBRA reportes: {len(df)} reportes encontrados")
    return df


# ── 3. Validación cruzada contra nuestros datos ───────────────────────────

def validate_sector_crosscheck(df_asociados: pd.DataFrame) -> pd.DataFrame:
    """
    Compara el sector de AMEFIBRA contra el sector que tenemos en nuestro SECTOR_MAP.
    Devuelve tabla de diferencias.
    """
    from src.data.benchmarks import FIBRASMX_PRICES_URL  # solo para importar SECTOR_MAP

    # SECTOR_MAP local (igual al de 1_faqs.py)
    our_sectors = {
        "FUNO11": "Diversificada", "FIBRAPL14": "Industrial",
        "FIBRAMQ12": "Industrial",  "DANHOS13": "Comercial",
        "FMTY14": "Mixta",          "FIHO12": "Hotelero",
        "FINN13": "Hotelero",       "FSHOP13": "Comercial",
        "FNOVA17": "Industrial",    "FIBRAUP18": "Industrial",
        "FPLUS16": "Diversificada", "STORAGE18": "Almacenaje",
        "FSITES20": "Infraestructura","EDUCA18": "Educativo",
        "NEXT25": "Industrial",
    }

    rows = []
    for _, row in df_asociados[df_asociados["ticker"].notna()].iterrows():
        ticker       = row["ticker"]
        sector_ame   = row.get("sector_amefibra", "N/D")
        sector_nues  = our_sectors.get(ticker, "N/D")
        match        = "✓" if sector_ame and sector_nues and sector_nues.lower() in str(sector_ame).lower() else "⚠"
        rows.append({
            "Ticker":         ticker,
            "Sector AMEFIBRA":  sector_ame,
            "Sector nuestro":   sector_nues,
            "Match":            match,
        })
    return pd.DataFrame(rows)


# ── Runner ────────────────────────────────────────────────────────────────

def run(save: bool = True) -> dict[str, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    log.info("Scrapeando AMEFIBRA asociados…")
    df_aso = fetch_asociados()
    log.info(f"  {len(df_aso)} FIBRAs")

    log.info("Scrapeando AMEFIBRA reportes…")
    df_rep = fetch_reportes()
    log.info(f"  {len(df_rep)} reportes")

    if save and not df_aso.empty:
        df_aso.to_csv(PROCESSED / "amefibra_asociados.csv", index=False)
        log.info(f"Guardado: amefibra_asociados.csv")
    if save and not df_rep.empty:
        df_rep.to_csv(PROCESSED / "amefibra_reportes.csv", index=False)
        log.info(f"Guardado: amefibra_reportes.csv")

    return {"asociados": df_aso, "reportes": df_rep}


if __name__ == "__main__":
    results = run()

    print("\n=== FIBRAs en AMEFIBRA ===")
    print(results["asociados"][["ticker", "nombre", "sector_amefibra", "sitio_web"]].to_string())

    print("\n=== Reportes disponibles ===")
    print(results["reportes"][["nombre", "ticker"]].head(20).to_string())
