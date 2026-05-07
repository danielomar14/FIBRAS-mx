"""
Página 1: FAQs — Análisis de mercado de FIBRAs mexicanas (2018–2025).

8 preguntas respondidas con datos reales:
  1. FIBRAs como mercado (7 años)
  2. ¿Ha crecido la liquidez?
  3. FIBRAs vs CETES (con y sin dividendos)
  4. Banxico y las FIBRAs (correlación tasas)
  5. ¿Cuánto pesa cada FIBRA?
  6. Recortes de distribuciones
  7. Liquidez en estrés (Amihud ratio)
  8. COVID: FIBRAs vs IPC
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.benchmarks import (
    cumulative_value,
    cetes_daily_return,
    fetch_banxico_rate,
    fetch_cetes,
    fetch_ipc,
    fetch_market_caps,
)

st.set_page_config(page_title="FAQs — FIBRAs MX", layout="wide")
st.title("FAQs — Mercado de FIBRAs Mexicanas")
st.caption("Análisis histórico 2021–2025 · Datos: yfinance, FibrasMX Supabase, Banxico SIE")

PROCESSED = ROOT / "data" / "processed"
FAQ_START  = "2021-01-01"
FAQ_END    = "2025-12-31"

# FIBRAs con suficiente historia para el período de análisis
INDEX_TICKERS = [
    "FUNO11", "FIBRAPL14", "FIBRAMQ12", "DANHOS13", "FMTY14",
    "FIHO12", "FINN13", "FSHOP13", "FNOVA17",
    "EDUCA18", "STORAGE18", "FSITES20",
]
SECTOR_MAP = {
    "FUNO11": "Diversificada", "FIBRAPL14": "Industrial", "FIBRAMQ12": "Industrial",
    "DANHOS13": "Comercial",   "FMTY14": "Mixta",        "FIHO12": "Hotelero",
    "FINN13": "Hotelero",      "FSHOP13": "Comercial",   "FNOVA17": "Industrial",
    "FIBRAUP18": "Industrial", "FPLUS16": "Diversificada","STORAGE18": "Almacenaje",
    "FSITES20": "Infraestructura","EDUCA18": "Educativo", "NEXT25": "Industrial",
}


# ── Loaders con caché ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "precios_diarios.parquet")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

@st.cache_data(ttl=3600)
def load_divs() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "distribuciones.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df

@st.cache_data(ttl=3600)
def get_ipc() -> pd.DataFrame:
    return fetch_ipc(FAQ_START, FAQ_END)

@st.cache_data(ttl=3600)
def get_cetes() -> pd.DataFrame:
    return fetch_cetes(FAQ_START, FAQ_END)

@st.cache_data(ttl=3600)
def get_banxico_rate() -> pd.DataFrame:
    return fetch_banxico_rate(FAQ_START, FAQ_END)

@st.cache_data(ttl=3600)
def get_market_caps() -> dict:
    return fetch_market_caps()


def build_fibra_index(df_prices: pd.DataFrame, tickers: list[str],
                      start: str, end: str) -> pd.Series:
    """Índice equiponderado de FIBRAs (retorno de precio), normado a 100."""
    pivot = (
        df_prices[df_prices["ticker"].isin(tickers)]
        .reset_index()
        .pivot(index="date", columns="ticker", values="close")
        .loc[start:end]
    )
    returns = pivot.pct_change().fillna(0)
    # Equal-weight promedio de retornos disponibles cada día
    index_ret = returns.mean(axis=1)
    index_level = 100 * (1 + index_ret).cumprod()
    index_level.iloc[0] = 100
    return index_level


def build_total_return_index(df_prices: pd.DataFrame, df_divs: pd.DataFrame,
                              tickers: list[str], start: str, end: str) -> pd.Series:
    """Índice equiponderado con retorno total (precio + dividendos)."""
    pivot = (
        df_prices[df_prices["ticker"].isin(tickers)]
        .reset_index()
        .pivot(index="date", columns="ticker", values="close")
        .loc[start:end]
    )
    returns = pivot.pct_change().fillna(0)

    # Añadir dividend yield en fecha ex-dividendo
    for _, row in df_divs[df_divs["ticker"].isin(tickers)].iterrows():
        d, ticker, div = row["date"], row["ticker"], row["dividend"]
        if d in returns.index and ticker in returns.columns:
            prev = pivot.loc[:d, ticker].iloc[-2] if len(pivot.loc[:d, ticker]) > 1 else pivot.loc[d, ticker]
            if prev and prev > 0:
                returns.loc[d, ticker] += div / prev

    index_ret = returns.mean(axis=1)
    index_level = 100 * (1 + index_ret).cumprod()
    index_level.iloc[0] = 100
    return index_level


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    return dd.min()


def annualized_cagr(series: pd.Series) -> float:
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1


def annualized_vol(series: pd.Series) -> float:
    return series.pct_change().dropna().std() * np.sqrt(252)


# ── Verificar datos ───────────────────────────────────────────────────────

if not (PROCESSED / "precios_diarios.parquet").exists():
    st.error("No hay datos. Ve a la página **0 — Datos** y presiona **Descargar datos**.")
    st.stop()

df_prices = load_prices()
df_divs   = load_divs()

# ── TABS ──────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📈 Mercado (7 años)",
    "💧 Liquidez",
    "⚔️ vs CETES",
    "🏦 Banxico",
    "🥧 Pesos de mercado",
    "✂️ Recortes distrib.",
    "🚨 Estrés",
    "📉 COVID vs IPC",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — FIBRAs como mercado
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("¿Cómo se ha comportado el mercado de FIBRAs desde 2021?")
    st.caption(f"Índice equiponderado de {len(INDEX_TICKERS)} FIBRAs con cobertura ≥80%, normado a 100 en {FAQ_START}.")

    idx = build_fibra_index(df_prices, INDEX_TICKERS, FAQ_START, FAQ_END)

    cagr = annualized_cagr(idx)
    vol  = annualized_vol(idx)
    mdd  = max_drawdown(idx)
    ret_total = idx.iloc[-1] / 100 - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno total (precio)", f"{ret_total*100:.1f}%")
    c2.metric("CAGR anualizado",        f"{cagr*100:.1f}%")
    c3.metric("Volatilidad anual",      f"{vol*100:.1f}%")
    c4.metric("Máx. Drawdown",          f"{mdd*100:.1f}%")

    # Drawdown series
    roll_max = idx.cummax()
    dd_series = (idx - roll_max) / roll_max * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=idx.index, y=idx.values, name="Índice FIBRAs",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_hrect(y0=100, y1=100, line_width=1, line_dash="dot",
                  line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, name="Drawdown %",
                             fill="tozeroy", fillcolor="rgba(214,39,40,0.25)",
                             line=dict(color="#d62728", width=1)), row=2, col=1)
    fig.update_layout(height=500, legend=dict(orientation="h"),
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(title_text="Índice (base 100)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("FIBRAs incluidas: " + ", ".join(INDEX_TICKERS))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Liquidez
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("¿Ha crecido la liquidez del mercado de FIBRAs?")

    df_liq = df_prices[df_prices["ticker"].isin(INDEX_TICKERS)].copy()
    df_liq = df_liq.reset_index()
    df_liq["vol_mxn"] = df_liq["volume"] * df_liq["close"]
    df_liq["year"]    = df_liq["date"].dt.year
    df_liq = df_liq[(df_liq["date"] >= FAQ_START) & (df_liq["date"] <= FAQ_END)]

    # Volumen total anual por FIBRA
    annual_vol = df_liq.groupby(["year", "ticker"])["vol_mxn"].sum().reset_index()
    annual_vol["vol_mxn_B"] = annual_vol["vol_mxn"] / 1e9

    fig2 = px.bar(annual_vol, x="year", y="vol_mxn_B", color="ticker",
                  labels={"vol_mxn_B": "Volumen (miles de millones MXN)", "year": "Año"},
                  title="Volumen anual transado por FIBRA (MXN)")
    fig2.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(t=40))
    st.plotly_chart(fig2, use_container_width=True)

    # Volumen promedio diario por FIBRA y año
    daily_avg = df_liq.groupby(["year", "ticker"])["vol_mxn"].mean().reset_index()
    daily_avg["vol_mxn_M"] = (daily_avg["vol_mxn"] / 1e6).round(1)
    pivot_liq = daily_avg.pivot(index="ticker", columns="year", values="vol_mxn_M").fillna(0)

    st.markdown("**Volumen promedio diario por FIBRA (millones MXN)**")
    def _color_vol(val):
        if val < 1:    return "background-color:#fdd; color:#700"
        if val < 10:   return "background-color:#ffd; color:#550"
        return              "background-color:#dfd; color:#050"
    st.dataframe(pivot_liq.style.map(_color_vol).format("{:.1f}"),
                 use_container_width=True)
    st.caption("🔴 < $1M MXN/día = ilíquida  |  🟡 $1–10M = media  |  🟢 > $10M = líquida")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — FIBRAs vs CETES
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("FIBRAs vs CETES: 2 escenarios")
    st.caption("Capital inicial: $200,000 MXN · Período: 2021–2025")

    cetes_df  = get_cetes()
    idx_price = build_fibra_index(df_prices, INDEX_TICKERS, FAQ_START, FAQ_END)
    idx_tr    = build_total_return_index(df_prices, df_divs, INDEX_TICKERS, FAQ_START, FAQ_END)

    # Alinear al índice de precios
    cetes_daily = cetes_daily_return(cetes_df["cetes_pct"])
    cetes_aligned = cetes_daily.reindex(idx_price.index).ffill()
    cetes_value = cumulative_value(cetes_aligned)

    val_precio = cumulative_value(idx_price.pct_change().fillna(0))
    val_tr     = cumulative_value(idx_tr.pct_change().fillna(0))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=val_precio.index, y=val_precio.values,
                              name="FIBRAs (solo precio)",
                              line=dict(color="#ff7f0e", width=2)))
    fig3.add_trace(go.Scatter(x=val_tr.index, y=val_tr.values,
                              name="FIBRAs (precio + dividendos)",
                              line=dict(color="#1f77b4", width=2.5)))
    fig3.add_trace(go.Scatter(x=cetes_value.index, y=cetes_value.values,
                              name="CETES 28d",
                              line=dict(color="#2ca02c", width=2, dash="dash")))
    fig3.add_hline(y=200_000, line_dash="dot", line_color="gray", line_width=1)
    fig3.update_layout(
        title="$200,000 MXN invertidos en 2018 → valor al cierre 2025",
        yaxis_title="Valor de portafolio (MXN)",
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h"),
        height=450, margin=dict(t=50),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Métricas
    def cagr_from_value(s: pd.Series) -> float:
        n = (s.index[-1] - s.index[0]).days / 365.25
        return (s.iloc[-1] / s.iloc[0]) ** (1/n) - 1

    m1, m2, m3 = st.columns(3)
    m1.metric("FIBRAs solo precio", f"${val_precio.iloc[-1]:,.0f}",
              f"CAGR {cagr_from_value(val_precio)*100:.1f}%/año")
    m2.metric("FIBRAs + dividendos", f"${val_tr.iloc[-1]:,.0f}",
              f"CAGR {cagr_from_value(val_tr)*100:.1f}%/año")
    m3.metric("CETES 28 días", f"${cetes_value.iloc[-1]:,.0f}",
              f"CAGR {cagr_from_value(cetes_value)*100:.1f}%/año")

    cetes_pct = "sin token Banxico (tabla de respaldo)" if True else "Banxico SIE"
    st.info(f"Tasas CETES: {cetes_pct}. Para datos oficiales, configura tu token BMX en `src/data/benchmarks.py`.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Banxico y las FIBRAs
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("¿Qué pasa con las FIBRAs cuando Banxico mueve tasas?")

    banxico_df = get_banxico_rate()
    idx_fibras = build_fibra_index(df_prices, INDEX_TICKERS, FAQ_START, FAQ_END)

    # Alinear
    banxico_aligned = banxico_df["tasa_objetivo"].reindex(idx_fibras.index).ffill()

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=idx_fibras.index, y=idx_fibras.values,
                              name="Índice FIBRAs (base 100)",
                              line=dict(color="#1f77b4", width=2)), secondary_y=False)
    fig4.add_trace(go.Scatter(x=banxico_aligned.index, y=banxico_aligned.values,
                              name="Tasa objetivo Banxico (%)",
                              line=dict(color="#d62728", width=2, dash="dot")),
                  secondary_y=True)
    fig4.update_yaxes(title_text="Índice FIBRAs", secondary_y=False)
    fig4.update_yaxes(title_text="Tasa objetivo (%)", secondary_y=True)
    fig4.update_layout(height=420, legend=dict(orientation="h"),
                       margin=dict(t=20, l=10, r=10))
    st.plotly_chart(fig4, use_container_width=True)

    # Correlación rolling 12 meses
    monthly_idx   = idx_fibras.resample("ME").last().pct_change().dropna()
    monthly_rate  = banxico_aligned.resample("ME").last().diff().dropna()
    common_idx    = monthly_idx.index.intersection(monthly_rate.index)
    rolling_corr  = monthly_idx.loc[common_idx].rolling(12).corr(monthly_rate.loc[common_idx]).dropna()

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values,
                              fill="tozeroy",
                              line=dict(color="#9467bd", width=1.5),
                              name="Correlación rolling 12m"))
    fig5.add_hline(y=0, line_dash="dot", line_color="gray")
    fig5.update_layout(title="Correlación rolling (12 meses): cambio de tasa vs retorno FIBRAs",
                       yaxis=dict(range=[-1, 1]), height=300, margin=dict(t=40))
    st.plotly_chart(fig5, use_container_width=True)

    # Tabla de ciclos
    st.markdown("**Rendimiento del índice FIBRAs por ciclo de tasas:**")
    cycles = [
        ("Alza 2018",         "2018-01-01", "2018-12-31", "6.5% → 8.25%"),
        ("Baja COVID 2020",   "2020-01-01", "2020-12-31", "8.25% → 4.25%"),
        ("Alza 2022–2023",    "2022-01-01", "2023-12-31", "5.5% → 11.25%"),
        ("Baja 2024–2025",    "2024-01-01", "2025-12-31", "11.25% → 8.5%"),
    ]
    rows = []
    for label, s, e, tasa in cycles:
        sub = idx_fibras.loc[s:e]
        if len(sub) < 2: continue
        ret = (sub.iloc[-1] / sub.iloc[0] - 1) * 100
        rows.append({"Período": label, "Tasas Banxico": tasa,
                     "Retorno índice FIBRAs": f"{ret:+.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Pesos de mercado
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("¿Cuánto representa cada FIBRA del mercado?")
    st.caption("Market cap actual (MXN). Fuente: FibrasMX.")

    caps = get_market_caps()
    if not caps:
        st.warning("No se pudo obtener market caps. Verifica conexión a internet.")
    else:
        cap_df = pd.DataFrame([
            {"Ticker": t, "Sector": SECTOR_MAP.get(t, "Otro"),
             "Market Cap (MXN)": v, "Market Cap (MXN B)": round(v/1e9, 1)}
            for t, v in sorted(caps.items(), key=lambda x: -x[1])
        ])
        total_mxn = cap_df["Market Cap (MXN)"].sum()
        cap_df["% del total"] = (cap_df["Market Cap (MXN)"] / total_mxn * 100).round(1)

        fig6 = px.treemap(cap_df, path=["Sector", "Ticker"],
                          values="Market Cap (MXN)",
                          color="Sector",
                          hover_data={"Market Cap (MXN B)": True, "% del total": True},
                          title=f"Market cap total: ${total_mxn/1e9:.0f}B MXN")
        fig6.update_traces(texttemplate="%{label}<br>%{customdata[1]:.1f}%")
        fig6.update_layout(height=480, margin=dict(t=50))
        st.plotly_chart(fig6, use_container_width=True)

        st.dataframe(cap_df[["Ticker","Sector","Market Cap (MXN B)","% del total"]],
                     use_container_width=True, hide_index=True)
        st.caption("⚠️ Los market caps son los reportados por FibrasMX en tiempo real. Verificar con BMV para FIBRAs pequeñas donde pueden existir inconsistencias.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Recortes de distribuciones
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("¿Cuántas FIBRAs han recortado distribuciones, y cuándo?")

    divs_faq = df_divs[
        (df_divs["date"] >= FAQ_START) & (df_divs["date"] <= FAQ_END)
        & df_divs["ticker"].isin(INDEX_TICKERS)
    ].copy()
    divs_faq["year"] = divs_faq["date"].dt.year

    annual_divs = divs_faq.groupby(["ticker", "year"])["dividend"].sum().reset_index()
    pivot_divs  = annual_divs.pivot(index="ticker", columns="year", values="dividend").round(4)

    # Calcular variación YoY
    yoy_pct = pivot_divs.pct_change(axis=1) * 100

    fig7 = px.imshow(pivot_divs.fillna(0),
                     color_continuous_scale="Greens",
                     title="Dividendo total anual por CBFI (MXN)",
                     text_auto=".2f",
                     aspect="auto")
    fig7.update_layout(height=420, margin=dict(t=50))
    st.plotly_chart(fig7, use_container_width=True)

    # Contar recortes por año
    cuts_by_year = (yoy_pct < -10).sum()
    cuts_by_year.name = "FIBRAs con recorte >10%"
    increases_by_year = (yoy_pct > 0).sum()
    increases_by_year.name = "FIBRAs con aumento"

    fig8 = go.Figure()
    years = [c for c in yoy_pct.columns if not pd.isna(c)]
    fig8.add_trace(go.Bar(x=years, y=increases_by_year.reindex(years).fillna(0),
                          name="Con aumento", marker_color="#2ca02c"))
    fig8.add_trace(go.Bar(x=years, y=cuts_by_year.reindex(years).fillna(0),
                          name="Con recorte >10%", marker_color="#d62728"))
    fig8.update_layout(barmode="group", title="FIBRAs que aumentaron vs recortaron distribución por año",
                       height=300, margin=dict(t=40))
    st.plotly_chart(fig8, use_container_width=True)

    # FIBRAs más afectadas en COVID
    st.markdown("**Variación YoY de distribuciones (%)**")
    def _color_yoy(val):
        if pd.isna(val): return ""
        if val < -15: return "background-color:#fcc; color:#600"
        if val < 0:   return "background-color:#fec; color:#640"
        if val > 10:  return "background-color:#cfc; color:#040"
        return ""
    if not yoy_pct.empty:
        st.dataframe(yoy_pct.style.map(_color_yoy).format("{:+.1f}%", na_rep="—"),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — Liquidez en estrés
# ═══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("¿Qué tan líquidas son las FIBRAs en momentos de estrés?")
    st.info(
        "**Proxy:** Amihud Illiquidity Ratio = |retorno diario| / (volumen × precio). "
        "A mayor ratio → menor liquidez. "
        "Bid-ask spread histórico no está disponible en fuentes gratuitas."
    )

    df_amihud = df_prices[df_prices["ticker"].isin(INDEX_TICKERS)].copy().reset_index()
    df_amihud["ret"] = df_amihud.groupby("ticker")["close"].pct_change().abs()
    df_amihud["vol_mxn"] = df_amihud["volume"] * df_amihud["close"]
    df_amihud["amihud"] = (df_amihud["ret"] / df_amihud["vol_mxn"].replace(0, np.nan)) * 1e9
    df_amihud = df_amihud.dropna(subset=["amihud"])
    df_amihud["date"] = pd.to_datetime(df_amihud["date"])

    periods = {
        "Baseline\n(2019)":          ("2019-01-01", "2020-01-15"),
        "COVID\n(feb–jun 2020)":     ("2020-02-01", "2020-06-30"),
        "Alza tasas\n(2022–2023)":   ("2022-03-01", "2023-12-31"),
    }

    records = []
    for period_name, (ps, pe) in periods.items():
        sub = df_amihud[(df_amihud["date"] >= ps) & (df_amihud["date"] <= pe)]
        avg = sub.groupby("ticker")["amihud"].median().reset_index()
        avg["period"] = period_name
        records.append(avg)
    df_plot = pd.concat(records, ignore_index=True)

    fig9 = px.bar(df_plot, x="ticker", y="amihud", color="period", barmode="group",
                  title="Amihud Illiquidity Ratio mediano por período (mayor = menos líquido)",
                  labels={"amihud": "Ratio Amihud (×10⁹)", "ticker": "FIBRA"})
    fig9.update_layout(height=420, legend=dict(orientation="h", y=-0.25),
                       margin=dict(t=40))
    st.plotly_chart(fig9, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — COVID vs IPC
# ═══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("COVID-19: ¿Las FIBRAs cayeron más que el IPC?")

    COVID_START = "2020-01-20"
    COVID_END   = "2020-12-31"

    ipc_df = get_ipc()
    if ipc_df.empty:
        st.warning("No se pudo descargar el IPC (^MXX). Verifica conexión.")
    else:
        idx_covid = build_fibra_index(df_prices, INDEX_TICKERS, COVID_START, COVID_END)
        ipc_covid = ipc_df.loc[COVID_START:COVID_END, "close"]

        # Normalizar a 100
        idx_n = idx_covid / idx_covid.iloc[0] * 100
        ipc_n = ipc_covid / ipc_covid.iloc[0] * 100

        # Por sector
        sector_series = {}
        for sector, tickers_s in {
            "Industrial":    ["FIBRAPL14", "FIBRAMQ12", "FNOVA17"],
            "Hotelero":      ["FIHO12", "FINN13"],
            "Comercial":     ["DANHOS13", "FSHOP13"],
            "Diversificada": ["FUNO11", "FMTY14"],
        }.items():
            valid = [t for t in tickers_s if t in INDEX_TICKERS]
            if valid:
                s = build_fibra_index(df_prices, valid, COVID_START, COVID_END)
                sector_series[sector] = s / s.iloc[0] * 100

        fig10 = go.Figure()
        colors_sector = {"Industrial":"#1f77b4","Hotelero":"#d62728",
                         "Comercial":"#ff7f0e","Diversificada":"#9467bd"}
        for sector, s in sector_series.items():
            fig10.add_trace(go.Scatter(x=s.index, y=s.values, name=f"FIBRAs {sector}",
                                       line=dict(color=colors_sector.get(sector,"gray"),
                                                 width=1.5, dash="dot")))
        fig10.add_trace(go.Scatter(x=idx_n.index, y=idx_n.values,
                                   name="Índice FIBRAs total",
                                   line=dict(color="#2ca02c", width=2.5)))
        fig10.add_trace(go.Scatter(x=ipc_n.index, y=ipc_n.values,
                                   name="IPC México (^MXX)",
                                   line=dict(color="#000000", width=2.5, dash="dash")))
        fig10.add_hline(y=100, line_dash="dot", line_color="gray", line_width=1)

        # Marcar trough del IPC
        ipc_trough_idx = ipc_n.idxmin()
        fig10.add_trace(go.Scatter(
            x=[ipc_trough_idx, ipc_trough_idx], y=[50, 110],
            mode="lines", line=dict(color="red", dash="dash", width=1),
            name=f"Mínimo IPC ({ipc_trough_idx.strftime('%d %b')})",
            showlegend=True,
        ))

        fig10.update_layout(
            title="COVID-19: FIBRAs vs IPC México (base 100 = 20 ene 2020)",
            yaxis_title="Índice normalizado",
            height=480, legend=dict(orientation="h", y=-0.25),
            margin=dict(t=50),
        )
        st.plotly_chart(fig10, use_container_width=True)

        # Métricas
        trough_fibras = idx_n.min()
        trough_ipc    = ipc_n.min()
        rec_fibras    = idx_n.iloc[-1]
        rec_ipc       = ipc_n.iloc[-1]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Índice FIBRAs**")
            st.metric("Caída máxima", f"{trough_fibras - 100:.1f}%")
            st.metric("Retorno dic-2020", f"{rec_fibras - 100:+.1f}%")
        with c2:
            st.markdown("**IPC México**")
            st.metric("Caída máxima", f"{trough_ipc - 100:.1f}%")
            st.metric("Retorno dic-2020", f"{rec_ipc - 100:+.1f}%")

        if trough_fibras < trough_ipc:
            st.error(f"Las FIBRAs cayeron **más** que el IPC ({trough_fibras-100:.1f}% vs {trough_ipc-100:.1f}%)")
        else:
            st.success(f"Las FIBRAs cayeron **menos** que el IPC ({trough_fibras-100:.1f}% vs {trough_ipc-100:.1f}%)")
