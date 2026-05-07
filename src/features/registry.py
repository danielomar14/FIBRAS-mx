"""
Feature registry: 40 variables × 10 parametrizaciones = 400 features.

Cada FeatureDef define:
  - var_id    : int  0-39
  - param_idx : int  0-9
  - block     : str  (A-F)
  - var_name  : str
  - param_name: str
  - compute   : Callable(prices_single: pd.Series,
                          metrics_single: pd.DataFrame | None,
                          banxico: pd.DataFrame | None,
                          ipc: pd.Series | None,
                          div_yield_ttm: pd.Series | None) -> pd.Series

Todas las funciones compute devuelven una pd.Series indexada por fecha.
El caller debe unirlas con .ffill() si es necesario antes de usarlas.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── tipos ──────────────────────────────────────────────────────────────────────

@dataclass
class FeatureDef:
    var_id: int
    param_idx: int
    block: str
    var_name: str
    param_name: str
    compute: Callable

# ── helpers comunes ────────────────────────────────────────────────────────────

def _rank_norm(s: pd.Series) -> pd.Series:
    """Rank 0-1 entre todos los valores no-NaN de la serie (cross-sectional no aplica aquí;
    se usa para time-series rank). Para cross-sectional se hace en builder."""
    return s.rank(pct=True)


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def _rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, min_periods=span // 2).mean()


def _trailing_ret(s: pd.Series, window: int) -> pd.Series:
    return s.pct_change(window)


def _realized_vol(s: pd.Series, window: int) -> pd.Series:
    lr = np.log(s / s.shift(1))
    return lr.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def _ffill_metric(metric_col: pd.Series, ref_index: pd.Index) -> pd.Series:
    """Reindex metric (quarterly) al índice de precios con ffill."""
    return metric_col.reindex(ref_index, method="ffill")


def _slope(s: pd.Series, window: int) -> pd.Series:
    """Pendiente de regresión lineal sobre ventana rodante (OLS)."""
    def _s(arr):
        n = len(arr)
        if n < 3 or np.isnan(arr).all():
            return np.nan
        x = np.arange(n)
        valid = ~np.isnan(arr)
        if valid.sum() < 3:
            return np.nan
        return np.polyfit(x[valid], arr[valid], 1)[0]
    return s.rolling(window, min_periods=window // 2).apply(_s, raw=True)


# ── Bloque A: variables de estrategias existentes (var_id 0-14) ───────────────

def _a01(prices, _, __, ___, ____, w):
    return _trailing_ret(prices, w)

def _a02(prices, _, __, ___, ____, w):
    return np.log(prices / prices.shift(w))

def _a03_vol(prices, _, __, ___, ____, w):
    return _realized_vol(prices, w)

def _a03_ratio(prices, _, __, ___, ____):
    return _realized_vol(prices, 21) / _realized_vol(prices, 252).replace(0, np.nan)

def _a04(prices, _, __, ___, ____, w):
    return prices / _ma(prices, w)

_a05_pairs = [(20,50),(20,100),(50,100),(50,200),(20,200),(100,200),(30,100),(30,200),(60,150),(10,50)]
def _a05(prices, _, __, ___, ____, fast, slow):
    return (_ma(prices, fast) > _ma(prices, slow)).astype(float)

_a06_windows = [50, 100, 200, 50, 100, 200, 50, 100, 200, 200]
_a06_signs   = [1, 1, 1, -1, -1, -1, 1, 1, 1, 1]
_a06_log     = [False]*6 + [True, True, True, True]
def _a06(prices, _, __, ___, ____, ma_w, sign, use_log):
    dist = (prices - _ma(prices, ma_w)) / _ma(prices, ma_w).replace(0, np.nan)
    if use_log:
        dist = np.log1p(dist)
    return sign * dist

def _a07_raw(prices, metrics, _, div_yield, ____):
    if div_yield is None: return pd.Series(np.nan, index=prices.index)
    return _ffill_metric(div_yield, prices.index)

def _a07_thresh(prices, metrics, _, div_yield, ____, thresh):
    s = _a07_raw(prices, metrics, _, div_yield, ____)
    return (s >= thresh).astype(float)

def _a07_top25(prices, metrics, _, div_yield, ____):
    s = _a07_raw(prices, metrics, _, div_yield, ____)
    return (s >= s.quantile(0.75)).astype(float)

def _a07_rank(prices, metrics, _, div_yield, ____):
    s = _a07_raw(prices, metrics, _, div_yield, ____)
    return _rank_norm(s)

def _metric_val(metrics, col, ref_index):
    if metrics is None or col not in metrics.columns:
        return pd.Series(np.nan, index=ref_index)
    s = metrics.groupby("date")[col].mean()
    return _ffill_metric(s, ref_index)

def _a08_raw(prices, metrics, _, __, ____):
    nav = _metric_val(metrics, "nav_per_cbfi", prices.index)
    return prices / nav.replace(0, np.nan)

def _a09_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "occupancy", prices.index)

def _a10_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "ffo_yield", prices.index)

def _a11_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "payout_ratio", prices.index)

def _a12_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "ltv", prices.index)

def _a13_raw(prices, _, banxico, __, ____):
    if banxico is None: return pd.Series(np.nan, index=prices.index)
    rate = banxico["rate"].reindex(prices.index, method="ffill") if "rate" in banxico.columns else pd.Series(np.nan, index=prices.index)
    return rate.diff(126)

def _a14_raw(prices, _, __, ___, ____):
    return _ma(prices, 30)

# Sector dummies — se inyectan desde el builder como constantes por ticker
# pero necesitamos placeholders aquí; el builder llena sector info.
def _a15_sector(prices, _, __, ___, ____, sector_fn):
    return sector_fn(prices.index)


# ── Bloque B: fundamentales nuevos (var_id 15-19) ─────────────────────────────

def _b01_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "noi_margin", prices.index)

def _b02_raw(prices, metrics, _, __, ____):
    return _metric_val(metrics, "debt_to_assets", prices.index)

def _b03_raw(prices, metrics, _, __, ____):
    s = _metric_val(metrics, "cbfi_count", prices.index)
    return s.pct_change(252)  # YoY % change

def _b04_raw(prices, metrics, _, __, ____):
    s = _metric_val(metrics, "appraised_value", prices.index)
    return s.pct_change(252)

def _b05_raw(prices, metrics, _, __, ____):
    dist = _metric_val(metrics, "dist_per_cbfi", prices.index)
    cbfi = _metric_val(metrics, "cbfi_count", prices.index)
    assets = _metric_val(metrics, "total_assets", prices.index)
    return dist * cbfi / assets.replace(0, np.nan)


# ── Bloque C: técnicas (var_id 20-24) ─────────────────────────────────────────

_rsi_periods = [7, 9, 14, 21, 28, 42, 63]

def _c01_rsi(prices, _, __, ___, ____, period):
    return _rsi(prices, period)

def _c01_below30(prices, _, __, ___, ____):
    return (_rsi(prices, 14) < 30).astype(float)

def _c01_above70(prices, _, __, ___, ____):
    return (_rsi(prices, 14) > 70).astype(float)

def _c01_dist50(prices, _, __, ___, ____):
    return _rsi(prices, 14) - 50

def _bb_pct(prices, window, n_std):
    ma = _ma(prices, window)
    sd = prices.rolling(window, min_periods=window//2).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    return (prices - lower) / (upper - lower).replace(0, np.nan)

def _c02(prices, _, __, ___, ____, window, n_std):
    return _bb_pct(prices, window, n_std)

def _c03_zvol(prices, _, __, ___, ____, window):
    vol = prices.pct_change().abs()  # proxy for volume unavailable here; use |return|
    return _zscore(vol, window)

def _c04_52wh(prices, _, __, ___, ____):
    return prices / prices.rolling(252, min_periods=126).max()

def _c05_amihud(prices, _, __, ___, ____, window):
    lr = prices.pct_change().abs()
    # volume not in prices Series; approximate with 1 (ratio stays relative)
    return lr.rolling(window, min_periods=window//2).mean()


# ── Bloque D: tendencia (var_id 25-29) ────────────────────────────────────────

def _d01(prices, _, __, ___, ____, h1, h2):
    return _trailing_ret(prices, h1) - _trailing_ret(prices, h2)

def _d02(prices, _, __, ___, ____, w1, w2, w3, w4):
    total = w1 + w2 + w3 + w4
    return (w1*_trailing_ret(prices,21) + w2*_trailing_ret(prices,63)
            + w3*_trailing_ret(prices,126) + w4*_trailing_ret(prices,252)) / total

def _d03(prices, _, __, ___, ____, window):
    lr = prices.pct_change()
    return (lr > 0).rolling(window, min_periods=window//2).mean()

def _d04(prices, _, __, ___, ____, fast, slow):
    return _ema(prices, fast) / _ma(prices, slow).replace(0, np.nan)

def _d05_channel(prices, _, __, ___, ____, window):
    hi = prices.rolling(window, min_periods=window//2).max()
    lo = prices.rolling(window, min_periods=window//2).min()
    return (prices - lo) / (hi - lo).replace(0, np.nan)


# ── Bloque E: mercado (var_id 30-34) ──────────────────────────────────────────

def _e01(prices, _, __, ___, ipc, window):
    if ipc is None: return pd.Series(np.nan, index=prices.index)
    ipc_al = ipc.reindex(prices.index, method="ffill")
    return _trailing_ret(prices, window) - _trailing_ret(ipc_al, window)

def _e02_stub(prices, _, __, ___, ipc, window):
    return _trailing_ret(prices, window)

def _e03_vol_slope(prices, _, __, ___, ____, window):
    lr = prices.pct_change().abs()
    return _slope(lr, window)

def _e04_corr(prices, _, __, ___, ipc, window):
    if ipc is None: return pd.Series(np.nan, index=prices.index)
    ipc_al = ipc.reindex(prices.index, method="ffill")
    lr_f = prices.pct_change()
    lr_i = ipc_al.pct_change()
    return lr_f.rolling(window, min_periods=window//2).corr(lr_i)

def _e05_size(prices, _, __, ___, ____):
    return _ma(prices, 30)


# ── Bloque F: macro (var_id 35-39) ────────────────────────────────────────────

def _f01_real_rate(prices, _, banxico, div_yield, ____):
    if banxico is None: return pd.Series(np.nan, index=prices.index)
    rate = banxico["rate"].reindex(prices.index, method="ffill") if "rate" in banxico.columns else pd.Series(np.nan, index=prices.index)
    cpi_proxy = rate.diff(252) * 0.5  # simplified CPI proxy
    return rate - cpi_proxy

def _f02_spread(prices, _, banxico, div_yield, ____):
    if banxico is None or div_yield is None:
        return pd.Series(np.nan, index=prices.index)
    cetes = banxico["rate"].reindex(prices.index, method="ffill") if "rate" in banxico.columns else pd.Series(np.nan, index=prices.index)
    dy = _ffill_metric(div_yield, prices.index)
    return dy - cetes

def _f03_phase(prices, _, banxico, __, ____):
    if banxico is None: return pd.Series(np.nan, index=prices.index)
    rate = banxico["rate"].reindex(prices.index, method="ffill") if "rate" in banxico.columns else pd.Series(np.nan, index=prices.index)
    return (rate.diff(63) > 0).astype(float)

def _f04_usdmxn(prices, _, banxico, __, ____, window):
    if banxico is None: return pd.Series(np.nan, index=prices.index)
    if "usdmxn" not in banxico.columns:
        return pd.Series(np.nan, index=prices.index)
    fx = banxico["usdmxn"].reindex(prices.index, method="ffill")
    return _trailing_ret(fx, window)

def _f05_ipc(prices, _, __, ___, ipc, window):
    if ipc is None: return pd.Series(np.nan, index=prices.index)
    ipc_al = ipc.reindex(prices.index, method="ffill")
    return _trailing_ret(ipc_al, window)


# ── Registro: 400 FeatureDef ──────────────────────────────────────────────────

FEATURE_REGISTRY: dict[int, FeatureDef] = {}

_gid = 0  # global feature id counter

def _reg(var_id, param_idx, block, var_name, param_name, fn):
    global _gid
    fid = var_id * 10 + param_idx
    FEATURE_REGISTRY[fid] = FeatureDef(var_id, param_idx, block, var_name, param_name, fn)
    _gid += 1

# ── A01: Retorno trailing ──────────────────────────────────────────────────────
_a01_windows = [21,42,63,84,126,168,252,378,504,756]
for _pi, _w in enumerate(_a01_windows):
    _reg(0, _pi, "A", "Retorno trailing", f"{_w}d",
         (lambda w: lambda p,m,b,d,i: _trailing_ret(p, w))(_w))

# ── A02: Log-retorno trailing ──────────────────────────────────────────────────
for _pi, _w in enumerate(_a01_windows):
    _reg(1, _pi, "A", "Log-retorno trailing", f"{_w}d",
         (lambda w: lambda p,m,b,d,i: np.log(p/p.shift(w)))(_w))

# ── A03: Volatilidad realizada ─────────────────────────────────────────────────
_a03_windows = [21,42,63,84,126,168,252,378,504]
for _pi, _w in enumerate(_a03_windows):
    _reg(2, _pi, "A", "Volatilidad realizada", f"{_w}d",
         (lambda w: lambda p,m,b,d,i: _realized_vol(p, w))(_w))
_reg(2, 9, "A", "Volatilidad realizada", "ratio(21/252)",
     lambda p,m,b,d,i: _realized_vol(p,21) / _realized_vol(p,252).replace(0,np.nan))

# ── A04: Precio / MA ───────────────────────────────────────────────────────────
_a04_windows = [20,30,50,60,100,120,150,200,250,300]
for _pi, _w in enumerate(_a04_windows):
    _reg(3, _pi, "A", "Precio/MA", f"MA{_w}",
         (lambda w: lambda p,m,b,d,i: p/_ma(p,w).replace(0,np.nan))(_w))

# ── A05: Golden Cross ──────────────────────────────────────────────────────────
for _pi, (fast, slow) in enumerate(_a05_pairs):
    _reg(4, _pi, "A", "Golden Cross", f"MA{fast}>MA{slow}",
         (lambda f,s: lambda p,m,b,d,i: (_ma(p,f)>_ma(p,s)).astype(float))(fast,slow))

# ── A06: Distancia a MA ────────────────────────────────────────────────────────
_a06_specs = [
    (50,1,False),(100,1,False),(200,1,False),
    (50,-1,False),(100,-1,False),(200,-1,False),
    (50,1,True),(100,1,True),(200,1,True),(200,-1,True),
]
for _pi, (mw, sg, lg) in enumerate(_a06_specs):
    lbl = f"{'log' if lg else ''}dist_MA{mw}{'_inv' if sg<0 else ''}"
    _reg(5, _pi, "A", "Distancia a MA", lbl,
         (lambda mw,sg,lg: lambda p,m,b,d,i: _a06(p,m,b,d,i,mw,sg,lg))(mw,sg,lg))

# ── A07: Dividend yield TTM ────────────────────────────────────────────────────
_reg(6, 0, "A", "Div yield TTM", "raw",       lambda p,m,b,d,i: _a07_raw(p,m,b,d,i))
_reg(6, 1, "A", "Div yield TTM", "rank",      lambda p,m,b,d,i: _rank_norm(_a07_raw(p,m,b,d,i)))
for _pi, _th in enumerate([3,4,5,6,7,8,9]):
    _reg(6, 2+_pi, "A", "Div yield TTM", f">{_th}%",
         (lambda t: lambda p,m,b,d,i: (_a07_raw(p,m,b,d,i)>=t/100).astype(float))(_th))
_reg(6, 9, "A", "Div yield TTM", "top25%",    lambda p,m,b,d,i: _a07_top25(p,m,b,d,i))

# ── A08: Precio/NAV ────────────────────────────────────────────────────────────
def _a08_thresh(thresh, above=False):
    def fn(p,m,b,d,i):
        s = _a08_raw(p,m,b,d,i)
        return (s > thresh if above else s < thresh).astype(float)
    return fn
_reg(7, 0, "A", "Precio/NAV", "raw",       lambda p,m,b,d,i: _a08_raw(p,m,b,d,i))
_reg(7, 1, "A", "Precio/NAV", "rank",      lambda p,m,b,d,i: _rank_norm(_a08_raw(p,m,b,d,i)))
for _pi, _th in enumerate([0.80,0.90,0.95,1.0,1.05,1.10]):
    _reg(7, 2+_pi, "A", "Precio/NAV", f"<{_th}",
         (lambda t: lambda p,m,b,d,i: (_a08_raw(p,m,b,d,i)<t).astype(float))(_th))
_reg(7, 8, "A", "Precio/NAV", "descuento%",
     lambda p,m,b,d,i: (1 - _a08_raw(p,m,b,d,i))*100)
_reg(7, 9, "A", "Precio/NAV", "rank_desc",
     lambda p,m,b,d,i: _rank_norm(1 - _a08_raw(p,m,b,d,i)))

# ── A09: Ocupación ─────────────────────────────────────────────────────────────
_a09_threshs = [0.75,0.80,0.85,0.87,0.90,0.92,0.95]
_reg(8, 0, "A", "Ocupación", "raw",    lambda p,m,b,d,i: _a09_raw(p,m,b,d,i))
_reg(8, 1, "A", "Ocupación", "rank",   lambda p,m,b,d,i: _rank_norm(_a09_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_a09_threshs):
    _reg(8, 2+_pi, "A", "Ocupación", f">={_th:.0%}",
         (lambda t: lambda p,m,b,d,i: (_a09_raw(p,m,b,d,i)>=t).astype(float))(_th))
_reg(8, 9, "A", "Ocupación", "QoQ",
     lambda p,m,b,d,i: _a09_raw(p,m,b,d,i).diff(63))

# ── A10: FFO yield ─────────────────────────────────────────────────────────────
_a10_threshs = [4,5,6,7,8,9,10]
_reg(9, 0, "A", "FFO yield", "raw",  lambda p,m,b,d,i: _a10_raw(p,m,b,d,i))
_reg(9, 1, "A", "FFO yield", "rank", lambda p,m,b,d,i: _rank_norm(_a10_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_a10_threshs):
    _reg(9, 2+_pi, "A", "FFO yield", f">{_th}%",
         (lambda t: lambda p,m,b,d,i: (_a10_raw(p,m,b,d,i)>=t/100).astype(float))(_th))
_reg(9, 9, "A", "FFO yield", "anual×4",
     lambda p,m,b,d,i: _a10_raw(p,m,b,d,i)*4)

# ── A11: Payout ratio ─────────────────────────────────────────────────────────
_a11_threshs = [0.60,0.70,0.75,0.80,0.90,1.0]
_reg(10, 0, "A", "Payout ratio", "raw",  lambda p,m,b,d,i: _a11_raw(p,m,b,d,i))
_reg(10, 1, "A", "Payout ratio", "rank", lambda p,m,b,d,i: _rank_norm(_a11_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_a11_threshs):
    _reg(10, 2+_pi, "A", "Payout ratio", f"<{_th}",
         (lambda t: lambda p,m,b,d,i: (_a11_raw(p,m,b,d,i)<t).astype(float))(_th))
_reg(10, 8, "A", "Payout ratio", "QoQ",  lambda p,m,b,d,i: _a11_raw(p,m,b,d,i).diff(63))
_reg(10, 9, "A", "Payout ratio", "inv",  lambda p,m,b,d,i: 1/_a11_raw(p,m,b,d,i).replace(0,np.nan))

# ── A12: LTV ──────────────────────────────────────────────────────────────────
_a12_threshs = [0.30,0.35,0.40,0.45,0.50]
_reg(11, 0, "A", "LTV", "raw",  lambda p,m,b,d,i: _a12_raw(p,m,b,d,i))
_reg(11, 1, "A", "LTV", "rank", lambda p,m,b,d,i: _rank_norm(_a12_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_a12_threshs):
    _reg(11, 2+_pi, "A", "LTV", f"<{_th}",
         (lambda t: lambda p,m,b,d,i: (_a12_raw(p,m,b,d,i)<t).astype(float))(_th))
_reg(11, 7, "A", "LTV", "QoQ",  lambda p,m,b,d,i: _a12_raw(p,m,b,d,i).diff(63))
_reg(11, 8, "A", "LTV", "inv",  lambda p,m,b,d,i: 1/_a12_raw(p,m,b,d,i).replace(0,np.nan))
_reg(11, 9, "A", "LTV", ">0.50", lambda p,m,b,d,i: (_a12_raw(p,m,b,d,i)>0.50).astype(float))

# ── A13: Cambio Banxico 6M ────────────────────────────────────────────────────
def _bnx_rate(p, b):
    if b is None: return pd.Series(np.nan, index=p.index)
    return b["rate"].reindex(p.index, method="ffill") if "rate" in b.columns else pd.Series(np.nan, index=p.index)

_reg(12, 0, "A", "Banxico 6M", "raw",        lambda p,m,b,d,i: _bnx_rate(p,b).diff(126))
_reg(12, 1, "A", "Banxico 6M", ">+50bps",    lambda p,m,b,d,i: (_bnx_rate(p,b).diff(126)>0.005).astype(float))
_reg(12, 2, "A", "Banxico 6M", ">+100bps",   lambda p,m,b,d,i: (_bnx_rate(p,b).diff(126)>0.01).astype(float))
_reg(12, 3, "A", "Banxico 6M", "<-50bps",    lambda p,m,b,d,i: (_bnx_rate(p,b).diff(126)<-0.005).astype(float))
_reg(12, 4, "A", "Banxico 6M", "<-100bps",   lambda p,m,b,d,i: (_bnx_rate(p,b).diff(126)<-0.01).astype(float))
_reg(12, 5, "A", "Banxico 6M", "abs>50bps",  lambda p,m,b,d,i: (_bnx_rate(p,b).diff(126).abs()>0.005).astype(float))
_reg(12, 6, "A", "Banxico 6M", "dirección",  lambda p,m,b,d,i: np.sign(_bnx_rate(p,b).diff(126)))
_reg(12, 7, "A", "Banxico 6M", "aceleración",lambda p,m,b,d,i: _bnx_rate(p,b).diff(126).diff(63))
_reg(12, 8, "A", "Banxico 6M", "rolling12M", lambda p,m,b,d,i: _bnx_rate(p,b).diff(252))
_reg(12, 9, "A", "Banxico 6M", "zscore",     lambda p,m,b,d,i: _zscore(_bnx_rate(p,b).diff(126)))

# ── A14: Size proxy ───────────────────────────────────────────────────────────
_reg(13, 0, "A", "Size proxy", "raw",      lambda p,m,b,d,i: _ma(p,30))
_reg(13, 1, "A", "Size proxy", "rank",     lambda p,m,b,d,i: _rank_norm(_ma(p,30)))
_reg(13, 2, "A", "Size proxy", "top3",     lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.80).astype(float))
_reg(13, 3, "A", "Size proxy", "top5",     lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.67).astype(float))
_reg(13, 4, "A", "Size proxy", "top7",     lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.53).astype(float))
_reg(13, 5, "A", "Size proxy", "top50%",   lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.50).astype(float))
_reg(13, 6, "A", "Size proxy", "percentil",lambda p,m,b,d,i: _rank_norm(_ma(p,30)))
_reg(13, 7, "A", "Size proxy", "zscore",   lambda p,m,b,d,i: _zscore(_ma(p,30)))
_reg(13, 8, "A", "Size proxy", "log",      lambda p,m,b,d,i: np.log(_ma(p,30).replace(0,np.nan)))
_reg(13, 9, "A", "Size proxy", "norm",     lambda p,m,b,d,i: _ma(p,30)/_ma(p,252).replace(0,np.nan))

# ── A15: Sector dummy — computed by builder with ticker metadata ───────────────
# Placeholders; builder injects sector info via closure
for _pi in range(10):
    _reg(14, _pi, "A", "Sector dummy", f"sector_p{_pi}",
         lambda p,m,b,d,i: pd.Series(0.0, index=p.index))

# ── B01: NOI margin ───────────────────────────────────────────────────────────
_reg(15, 0, "B", "NOI margin", "raw",        lambda p,m,b,d,i: _b01_raw(p,m,b,d,i))
_reg(15, 1, "B", "NOI margin", "rank",       lambda p,m,b,d,i: _rank_norm(_b01_raw(p,m,b,d,i)))
for _pi, _th in enumerate([30,40,50,60,65]):
    _reg(15, 2+_pi, "B", "NOI margin", f">{_th}%",
         (lambda t: lambda p,m,b,d,i: (_b01_raw(p,m,b,d,i)>t/100).astype(float))(_th))
_reg(15, 7, "B", "NOI margin", "QoQ",        lambda p,m,b,d,i: _b01_raw(p,m,b,d,i).diff(63))
_reg(15, 8, "B", "NOI margin", "YoY",        lambda p,m,b,d,i: _b01_raw(p,m,b,d,i).diff(252))
_reg(15, 9, "B", "NOI margin", "trend4Q",    lambda p,m,b,d,i: _slope(_b01_raw(p,m,b,d,i),252))

# ── B02: Deuda/Activos ────────────────────────────────────────────────────────
_b02_threshs = [0.35,0.40,0.45,0.50,0.55]
_reg(16, 0, "B", "Deuda/Activos", "raw",     lambda p,m,b,d,i: _b02_raw(p,m,b,d,i))
_reg(16, 1, "B", "Deuda/Activos", "rank",    lambda p,m,b,d,i: _rank_norm(_b02_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_b02_threshs):
    _reg(16, 2+_pi, "B", "Deuda/Activos", f"<{_th}",
         (lambda t: lambda p,m,b,d,i: (_b02_raw(p,m,b,d,i)<t).astype(float))(_th))
_reg(16, 7, "B", "Deuda/Activos", "QoQ",     lambda p,m,b,d,i: _b02_raw(p,m,b,d,i).diff(63))
_reg(16, 8, "B", "Deuda/Activos", "inv",     lambda p,m,b,d,i: 1/_b02_raw(p,m,b,d,i).replace(0,np.nan))
_reg(16, 9, "B", "Deuda/Activos", "trend4Q", lambda p,m,b,d,i: _slope(_b02_raw(p,m,b,d,i),252))

# ── B03: Dilución de CBFIs ────────────────────────────────────────────────────
_reg(17, 0, "B", "Dilución CBFIs", "%YoY",    lambda p,m,b,d,i: _b03_raw(p,m,b,d,i))
_reg(17, 1, "B", "Dilución CBFIs", "pos",     lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)>0).astype(float))
_reg(17, 2, "B", "Dilución CBFIs", "neg",     lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<0).astype(float))
_reg(17, 3, "B", "Dilución CBFIs", "abs",     lambda p,m,b,d,i: _b03_raw(p,m,b,d,i).abs())
_reg(17, 4, "B", "Dilución CBFIs", "<1%",     lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<0.01).astype(float))
_reg(17, 5, "B", "Dilución CBFIs", "<3%",     lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<0.03).astype(float))
_reg(17, 6, "B", "Dilución CBFIs", "<5%",     lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<0.05).astype(float))
_reg(17, 7, "B", "Dilución CBFIs", "<-1%",    lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<-0.01).astype(float))
_reg(17, 8, "B", "Dilución CBFIs", "<-3%",    lambda p,m,b,d,i: (_b03_raw(p,m,b,d,i)<-0.03).astype(float))
_reg(17, 9, "B", "Dilución CBFIs", "zscore",  lambda p,m,b,d,i: _zscore(_b03_raw(p,m,b,d,i)))

# ── B04: Apreciación propiedades ──────────────────────────────────────────────
_reg(18, 0, "B", "Aprec. propiedades", "%YoY",    lambda p,m,b,d,i: _b04_raw(p,m,b,d,i))
_reg(18, 1, "B", "Aprec. propiedades", "rank",    lambda p,m,b,d,i: _rank_norm(_b04_raw(p,m,b,d,i)))
_reg(18, 2, "B", "Aprec. propiedades", ">0%",     lambda p,m,b,d,i: (_b04_raw(p,m,b,d,i)>0).astype(float))
_reg(18, 3, "B", "Aprec. propiedades", ">2%",     lambda p,m,b,d,i: (_b04_raw(p,m,b,d,i)>0.02).astype(float))
_reg(18, 4, "B", "Aprec. propiedades", ">5%",     lambda p,m,b,d,i: (_b04_raw(p,m,b,d,i)>0.05).astype(float))
_reg(18, 5, "B", "Aprec. propiedades", ">8%",     lambda p,m,b,d,i: (_b04_raw(p,m,b,d,i)>0.08).astype(float))
_reg(18, 6, "B", "Aprec. propiedades", "<0%",     lambda p,m,b,d,i: (_b04_raw(p,m,b,d,i)<0).astype(float))
_reg(18, 7, "B", "Aprec. propiedades", "trend4Q", lambda p,m,b,d,i: _slope(_b04_raw(p,m,b,d,i),252))
_reg(18, 8, "B", "Aprec. propiedades", "QoQ",     lambda p,m,b,d,i: _b04_raw(p,m,b,d,i).diff(63))
_reg(18, 9, "B", "Aprec. propiedades", "zscore",  lambda p,m,b,d,i: _zscore(_b04_raw(p,m,b,d,i)))

# ── B05: Distribución/Activos ─────────────────────────────────────────────────
_b05_threshs = [1,2,3,4,5]
_reg(19, 0, "B", "Dist/Activos", "raw",     lambda p,m,b,d,i: _b05_raw(p,m,b,d,i))
_reg(19, 1, "B", "Dist/Activos", "rank",    lambda p,m,b,d,i: _rank_norm(_b05_raw(p,m,b,d,i)))
for _pi, _th in enumerate(_b05_threshs):
    _reg(19, 2+_pi, "B", "Dist/Activos", f">{_th}%",
         (lambda t: lambda p,m,b,d,i: (_b05_raw(p,m,b,d,i)>t/100).astype(float))(_th))
_reg(19, 7, "B", "Dist/Activos", "trend4Q", lambda p,m,b,d,i: _slope(_b05_raw(p,m,b,d,i),252))
_reg(19, 8, "B", "Dist/Activos", "QoQ",     lambda p,m,b,d,i: _b05_raw(p,m,b,d,i).diff(63))
_reg(19, 9, "B", "Dist/Activos", "zscore",  lambda p,m,b,d,i: _zscore(_b05_raw(p,m,b,d,i)))

# ── C01: RSI ──────────────────────────────────────────────────────────────────
_rsi_ps = [7,9,14,21,28,42,63]
for _pi, _per in enumerate(_rsi_ps):
    _reg(20, _pi, "C", "RSI", f"RSI-{_per}",
         (lambda per: lambda p,m,b,d,i: _rsi(p,per))(_per))
_reg(20, 7, "C", "RSI", "RSI<30",     lambda p,m,b,d,i: (_rsi(p,14)<30).astype(float))
_reg(20, 8, "C", "RSI", "RSI>70",     lambda p,m,b,d,i: (_rsi(p,14)>70).astype(float))
_reg(20, 9, "C", "RSI", "dist-50",    lambda p,m,b,d,i: _rsi(p,14)-50)

# ── C02: Bollinger ────────────────────────────────────────────────────────────
_c02_specs = [
    (20,2),(20,1.5),(50,2),(20,2),(20,2),(20,2),(20,2),(20,2),(20,2),(20,2),
]
_c02_labels = ["BB(20,2)","BB(20,1.5)","BB(50,2)","%B(0-1)","%B<0.2","%B>0.8",
               "BB-width","BB-width-z","squeeze","inv-%B"]
def _bb_width(p, window=20, nstd=2):
    ma = _ma(p,window)
    sd = p.rolling(window, min_periods=window//2).std()
    return (nstd*2*sd) / ma.replace(0,np.nan)
def _c02_fn(pi):
    if pi == 0: return lambda p,m,b,d,i: _bb_pct(p,20,2)
    if pi == 1: return lambda p,m,b,d,i: _bb_pct(p,20,1.5)
    if pi == 2: return lambda p,m,b,d,i: _bb_pct(p,50,2)
    if pi == 3: return lambda p,m,b,d,i: _bb_pct(p,20,2)
    if pi == 4: return lambda p,m,b,d,i: (_bb_pct(p,20,2)<0.2).astype(float)
    if pi == 5: return lambda p,m,b,d,i: (_bb_pct(p,20,2)>0.8).astype(float)
    if pi == 6: return lambda p,m,b,d,i: _bb_width(p)
    if pi == 7: return lambda p,m,b,d,i: _zscore(_bb_width(p))
    if pi == 8: return lambda p,m,b,d,i: (_bb_width(p)<_bb_width(p).rolling(126,min_periods=63).min()*1.05).astype(float)
    if pi == 9: return lambda p,m,b,d,i: 1-_bb_pct(p,20,2)
for _pi in range(10):
    _reg(21, _pi, "C", "Bollinger", _c02_labels[_pi], _c02_fn(_pi))

# ── C03: Z-score volumen (proxy: |return|) ────────────────────────────────────
_c03_wins = [21,63,126,252]
for _pi, _w in enumerate(_c03_wins):
    _reg(22, _pi, "C", "Vol z-score", f"z{_w}d",
         (lambda w: lambda p,m,b,d,i: _zscore(p.pct_change().abs(), w))(_w))
_reg(22, 4, "C", "Vol z-score", ">+1σ",  lambda p,m,b,d,i: (_zscore(p.pct_change().abs(),63)>1).astype(float))
_reg(22, 5, "C", "Vol z-score", ">+2σ",  lambda p,m,b,d,i: (_zscore(p.pct_change().abs(),63)>2).astype(float))
_reg(22, 6, "C", "Vol z-score", "<-1σ",  lambda p,m,b,d,i: (_zscore(p.pct_change().abs(),63)<-1).astype(float))
_reg(22, 7, "C", "Vol z-score", "trend", lambda p,m,b,d,i: _slope(p.pct_change().abs(),63))
_reg(22, 8, "C", "Vol z-score", "r21/63",lambda p,m,b,d,i: _zscore(p.pct_change().abs(),21)/_zscore(p.pct_change().abs(),63).replace(0,np.nan))
_reg(22, 9, "C", "Vol z-score", "r5/21", lambda p,m,b,d,i: _zscore(p.pct_change().abs(),5)/_zscore(p.pct_change().abs(),21).replace(0,np.nan))

# ── C04: Precio/52w high ──────────────────────────────────────────────────────
_reg(23, 0, "C", "P/52wH", "raw",        lambda p,m,b,d,i: p/p.rolling(252,min_periods=126).max())
_reg(23, 1, "C", "P/52wH", ">0.90",      lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()>0.90).astype(float))
_reg(23, 2, "C", "P/52wH", ">0.95",      lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()>0.95).astype(float))
_reg(23, 3, "C", "P/52wH", ">0.98",      lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()>0.98).astype(float))
_reg(23, 4, "C", "P/52wH", "<0.80",      lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()<0.80).astype(float))
_reg(23, 5, "C", "P/52wH", "<0.70",      lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()<0.70).astype(float))
_reg(23, 6, "C", "P/52wH", "dist%",      lambda p,m,b,d,i: p/p.rolling(252,min_periods=126).max()-1)
_reg(23, 7, "C", "P/52wH", "dd-52w",     lambda p,m,b,d,i: p/p.rolling(252,min_periods=126).max()-1)
_reg(23, 8, "C", "P/52wH", "nuevo-max",  lambda p,m,b,d,i: (p>=p.rolling(252,min_periods=126).max()).astype(float))
_reg(23, 9, "C", "P/52wH", "recuper",    lambda p,m,b,d,i: (p/p.rolling(252,min_periods=126).max()-p.rolling(252,min_periods=126).min()/p.rolling(252,min_periods=126).max()))

# ── C05: Amihud illiquidity (proxy) ──────────────────────────────────────────
_c05_wins = [21,63,252]
for _pi, _w in enumerate(_c05_wins):
    _reg(24, _pi, "C", "Amihud", f"ratio({_w}d)",
         (lambda w: lambda p,m,b,d,i: p.pct_change().abs().rolling(w,min_periods=w//2).mean())(_w))
_reg(24, 3, "C", "Amihud", "rank_liq",   lambda p,m,b,d,i: 1-_rank_norm(p.pct_change().abs().rolling(63,min_periods=30).mean()))
_reg(24, 4, "C", "Amihud", "rank_iliq",  lambda p,m,b,d,i: _rank_norm(p.pct_change().abs().rolling(63,min_periods=30).mean()))
_reg(24, 5, "C", "Amihud", "high",       lambda p,m,b,d,i: (p.pct_change().abs().rolling(63,min_periods=30).mean()>p.pct_change().abs().rolling(252,min_periods=126).mean()).astype(float))
_reg(24, 6, "C", "Amihud", "low",        lambda p,m,b,d,i: (p.pct_change().abs().rolling(63,min_periods=30).mean()<p.pct_change().abs().rolling(252,min_periods=126).mean()).astype(float))
_reg(24, 7, "C", "Amihud", "trend4Q",    lambda p,m,b,d,i: _slope(p.pct_change().abs().rolling(63,min_periods=30).mean(),252))
_reg(24, 8, "C", "Amihud", "zscore",     lambda p,m,b,d,i: _zscore(p.pct_change().abs().rolling(63,min_periods=30).mean()))
_reg(24, 9, "C", "Amihud", "vs-sector",  lambda p,m,b,d,i: _zscore(p.pct_change().abs().rolling(63,min_periods=30).mean()))

# ── D01: Aceleración momentum ─────────────────────────────────────────────────
_d01_specs = [(63,126),(21,63),(126,252),(63,126),(63,126),(63,126),(63,126),(21,63),(21,63),(63,126)]
_d01_labels = ["r63-r126","r21-r63","r126-r252","r63/r126","señal_pos","zscore","rank","2da_deriv","neg","norm"]
def _d01_fn(pi):
    if pi == 0: return lambda p,m,b,d,i: _trailing_ret(p,63)-_trailing_ret(p,126)
    if pi == 1: return lambda p,m,b,d,i: _trailing_ret(p,21)-_trailing_ret(p,63)
    if pi == 2: return lambda p,m,b,d,i: _trailing_ret(p,126)-_trailing_ret(p,252)
    if pi == 3: return lambda p,m,b,d,i: _trailing_ret(p,63)/_trailing_ret(p,126).replace(0,np.nan)
    if pi == 4: return lambda p,m,b,d,i: ((_trailing_ret(p,63)-_trailing_ret(p,126))>0).astype(float)
    if pi == 5: return lambda p,m,b,d,i: _zscore(_trailing_ret(p,63)-_trailing_ret(p,126))
    if pi == 6: return lambda p,m,b,d,i: _rank_norm(_trailing_ret(p,63)-_trailing_ret(p,126))
    if pi == 7: return lambda p,m,b,d,i: (_trailing_ret(p,63)-_trailing_ret(p,126)).diff(21)
    if pi == 8: return lambda p,m,b,d,i: -(_trailing_ret(p,63)-_trailing_ret(p,126))
    if pi == 9: return lambda p,m,b,d,i: (_trailing_ret(p,63)-_trailing_ret(p,126))/_trailing_ret(p,252).replace(0,np.nan).abs()
for _pi in range(10):
    _reg(25, _pi, "D", "Aceleración mom.", _d01_labels[_pi], _d01_fn(_pi))

# ── D02: Score multi-timeframe ────────────────────────────────────────────────
_d02_combos = [
    (0.25,0.25,0.25,0.25),(0.4,0.3,0.2,0.1),(0.1,0.2,0.3,0.4),(0.5,0.3,0.15,0.05),
    (0.05,0.15,0.3,0.5),(0.33,0.33,0.33,0),(0.5,0.5,0,0),(0,0,0.5,0.5),
    (0.6,0.2,0.1,0.1),(0.1,0.1,0.2,0.6),
]
def _d02_fn(w1,w2,w3,w4):
    total = w1+w2+w3+w4
    return lambda p,m,b,d,i: (w1*_trailing_ret(p,21)+w2*_trailing_ret(p,63)+w3*_trailing_ret(p,126)+w4*_trailing_ret(p,252))/(total or 1)
for _pi, (w1,w2,w3,w4) in enumerate(_d02_combos):
    _reg(26, _pi, "D", "Multi-TF score", f"w({w1},{w2},{w3},{w4})", _d02_fn(w1,w2,w3,w4))

# ── D03: Consistencia retornos ────────────────────────────────────────────────
_d03_wins = [21,63,126,252]
for _pi, _w in enumerate(_d03_wins):
    _reg(27, _pi, "D", "Consistencia ret.", f"%pos_{_w}d",
         (lambda w: lambda p,m,b,d,i: (p.pct_change()>0).rolling(w,min_periods=w//2).mean())(_w))
_reg(27, 4, "D", "Consistencia ret.", ">50%",   lambda p,m,b,d,i: ((p.pct_change()>0).rolling(63,min_periods=30).mean()>0.5).astype(float))
_reg(27, 5, "D", "Consistencia ret.", ">60%",   lambda p,m,b,d,i: ((p.pct_change()>0).rolling(63,min_periods=30).mean()>0.6).astype(float))
_reg(27, 6, "D", "Consistencia ret.", ">70%",   lambda p,m,b,d,i: ((p.pct_change()>0).rolling(63,min_periods=30).mean()>0.7).astype(float))
_reg(27, 7, "D", "Consistencia ret.", "mejora", lambda p,m,b,d,i: (p.pct_change()>0).rolling(63,min_periods=30).mean()-(p.pct_change()>0).rolling(252,min_periods=126).mean())
_reg(27, 8, "D", "Consistencia ret.", "zscore", lambda p,m,b,d,i: _zscore((p.pct_change()>0).rolling(63,min_periods=30).mean()))
_reg(27, 9, "D", "Consistencia ret.", "rank",   lambda p,m,b,d,i: _rank_norm((p.pct_change()>0).rolling(63,min_periods=30).mean()))

# ── D04: EMA vs SMA ───────────────────────────────────────────────────────────
_d04_specs = [(21,21),(50,50),(21,50),(21,21),(21,21),(21,50),(21,50),(21,50),(21,50),(21,50)]
_d04_labels = ["EMA21/SMA21","EMA50/SMA50","EMA/SMA",">1 sube","<1 baja","dist","ratio","señal","zscore","rank"]
def _d04_fn(pi):
    if pi == 0: return lambda p,m,b,d,i: _ema(p,21)/_ma(p,21).replace(0,np.nan)
    if pi == 1: return lambda p,m,b,d,i: _ema(p,50)/_ma(p,50).replace(0,np.nan)
    if pi == 2: return lambda p,m,b,d,i: _ema(p,21)/_ma(p,50).replace(0,np.nan)
    if pi == 3: return lambda p,m,b,d,i: (_ema(p,21)>_ma(p,21)).astype(float)
    if pi == 4: return lambda p,m,b,d,i: (_ema(p,21)<_ma(p,21)).astype(float)
    if pi == 5: return lambda p,m,b,d,i: (_ema(p,21)-_ma(p,50))/_ma(p,50).replace(0,np.nan)
    if pi == 6: return lambda p,m,b,d,i: _ema(p,21)/_ma(p,50).replace(0,np.nan)
    if pi == 7: return lambda p,m,b,d,i: (_ema(p,21)>_ma(p,50)).astype(float)
    if pi == 8: return lambda p,m,b,d,i: _zscore(_ema(p,21)/_ma(p,50).replace(0,np.nan))
    if pi == 9: return lambda p,m,b,d,i: _rank_norm(_ema(p,21)/_ma(p,50).replace(0,np.nan))
for _pi in range(10):
    _reg(28, _pi, "D", "EMA vs SMA", _d04_labels[_pi], _d04_fn(_pi))

# ── D05: Canal de precio ──────────────────────────────────────────────────────
_d05_wins = [20,63,126]
for _pi, _w in enumerate(_d05_wins):
    _reg(29, _pi, "D", "Canal de precio", f"%chan_{_w}d",
         (lambda w: lambda p,m,b,d,i: _d05_channel(p,m,b,d,i,w))(w=_w))
_reg(29, 3, "D", "Canal de precio", ">80%",      lambda p,m,b,d,i: (_d05_channel(p,m,b,d,i,20)>0.8).astype(float))
_reg(29, 4, "D", "Canal de precio", ">90%",      lambda p,m,b,d,i: (_d05_channel(p,m,b,d,i,20)>0.9).astype(float))
_reg(29, 5, "D", "Canal de precio", "<20%",      lambda p,m,b,d,i: (_d05_channel(p,m,b,d,i,20)<0.2).astype(float))
_reg(29, 6, "D", "Canal de precio", "<10%",      lambda p,m,b,d,i: (_d05_channel(p,m,b,d,i,20)<0.1).astype(float))
_reg(29, 7, "D", "Canal de precio", "break_up",  lambda p,m,b,d,i: ((_d05_channel(p,m,b,d,i,20)>0.95) & (_d05_channel(p,m,b,d,i,20).shift(1)<=0.95)).astype(float))
_reg(29, 8, "D", "Canal de precio", "break_dn",  lambda p,m,b,d,i: ((_d05_channel(p,m,b,d,i,20)<0.05) & (_d05_channel(p,m,b,d,i,20).shift(1)>=0.05)).astype(float))
_reg(29, 9, "D", "Canal de precio", "norm",      lambda p,m,b,d,i: _d05_channel(p,m,b,d,i,63))

# ── E01: Fuerza relativa vs IPC ───────────────────────────────────────────────
_e01_wins = [21,63,126,252]
for _pi, _w in enumerate(_e01_wins):
    _reg(30, _pi, "E", "RS vs IPC", f"rs_{_w}d",
         (lambda w: lambda p,m,b,d,i: _e01(p,m,b,d,i,w))(_w))
_reg(30, 4, "E", "RS vs IPC", "rank",      lambda p,m,b,d,i: _rank_norm(_e01(p,m,b,d,i,63)))
_reg(30, 5, "E", "RS vs IPC", ">0%",       lambda p,m,b,d,i: (_e01(p,m,b,d,i,63)>0).astype(float))
_reg(30, 6, "E", "RS vs IPC", ">5%",       lambda p,m,b,d,i: (_e01(p,m,b,d,i,63)>0.05).astype(float))
_reg(30, 7, "E", "RS vs IPC", ">10%",      lambda p,m,b,d,i: (_e01(p,m,b,d,i,63)>0.10).astype(float))
_reg(30, 8, "E", "RS vs IPC", "zscore",    lambda p,m,b,d,i: _zscore(_e01(p,m,b,d,i,63)))
_reg(30, 9, "E", "RS vs IPC", "tendencia", lambda p,m,b,d,i: _slope(_e01(p,m,b,d,i,63),126))

# ── E02: Momentum sectorial (stub = self momentum; builder agrega peers) ───────
_e02_wins = [21,63,126]
for _pi, _w in enumerate(_e02_wins):
    _reg(31, _pi, "E", "Mom. sectorial", f"ret_{_w}d",
         (lambda w: lambda p,m,b,d,i: _trailing_ret(p,w))(_w))
for _pi in range(7):
    _reg(31, 3+_pi, "E", "Mom. sectorial", f"sector_p{3+_pi}",
         lambda p,m,b,d,i: _trailing_ret(p,63))

# ── E03: Tendencia de volumen ─────────────────────────────────────────────────
_e03_wins = [21,63,126]
for _pi, _w in enumerate(_e03_wins):
    _reg(32, _pi, "E", "Tend. volumen", f"slope_{_w}d",
         (lambda w: lambda p,m,b,d,i: _slope(p.pct_change().abs(),w))(_w))
_reg(32, 3, "E", "Tend. volumen", ">0%",    lambda p,m,b,d,i: (_slope(p.pct_change().abs(),63)>0).astype(float))
_reg(32, 4, "E", "Tend. volumen", ">+20%",  lambda p,m,b,d,i: (_slope(p.pct_change().abs(),63)>0.0002).astype(float))
_reg(32, 5, "E", "Tend. volumen", ">+50%",  lambda p,m,b,d,i: (_slope(p.pct_change().abs(),63)>0.0005).astype(float))
_reg(32, 6, "E", "Tend. volumen", "r5/15d", lambda p,m,b,d,i: p.pct_change().abs().rolling(5,min_periods=3).mean()/p.pct_change().abs().rolling(15,min_periods=8).mean().replace(0,np.nan))
_reg(32, 7, "E", "Tend. volumen", "zscore", lambda p,m,b,d,i: _zscore(_slope(p.pct_change().abs(),63)))
_reg(32, 8, "E", "Tend. volumen", "pos",    lambda p,m,b,d,i: (_slope(p.pct_change().abs(),21)>0).astype(float))
_reg(32, 9, "E", "Tend. volumen", "rank",   lambda p,m,b,d,i: _rank_norm(_slope(p.pct_change().abs(),63)))

# ── E04: Correlación con IPC ──────────────────────────────────────────────────
_e04_wins = [63,126,252]
for _pi, _w in enumerate(_e04_wins):
    _reg(33, _pi, "E", "Corr IPC", f"corr_{_w}d",
         (lambda w: lambda p,m,b,d,i: _e04_corr(p,m,b,d,i,w))(_w))
_reg(33, 3, "E", "Corr IPC", "<0.3",   lambda p,m,b,d,i: (_e04_corr(p,m,b,d,i,63)<0.3).astype(float))
_reg(33, 4, "E", "Corr IPC", ">0.7",   lambda p,m,b,d,i: (_e04_corr(p,m,b,d,i,63)>0.7).astype(float))
_reg(33, 5, "E", "Corr IPC", "beta63", lambda p,m,b,d,i: _e04_corr(p,m,b,d,i,63)*_realized_vol(p,63)/_realized_vol(p.reindex(p.index,method="ffill") if i is None else i.reindex(p.index,method="ffill"),63).replace(0,np.nan) if i is not None else pd.Series(np.nan,index=p.index))
_reg(33, 6, "E", "Corr IPC", "beta126",lambda p,m,b,d,i: _e04_corr(p,m,b,d,i,126))
_reg(33, 7, "E", "Corr IPC", "beta252",lambda p,m,b,d,i: _e04_corr(p,m,b,d,i,252))
_reg(33, 8, "E", "Corr IPC", "te",     lambda p,m,b,d,i: (p.pct_change()-(i.reindex(p.index,method="ffill").pct_change() if i is not None else pd.Series(0,index=p.index))).rolling(63,min_periods=30).std())
_reg(33, 9, "E", "Corr IPC", "corr-inv",lambda p,m,b,d,i: 1-_e04_corr(p,m,b,d,i,63).abs())

# ── E05: Tamaño relativo ──────────────────────────────────────────────────────
_reg(34, 0, "E", "Tamaño relativo", "rank_top5",  lambda p,m,b,d,i: _rank_norm(_ma(p,30)))
_reg(34, 1, "E", "Tamaño relativo", "rank_top3",  lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.80).astype(float))
_reg(34, 2, "E", "Tamaño relativo", "decil",      lambda p,m,b,d,i: (_rank_norm(_ma(p,30))*10).astype(int))
_reg(34, 3, "E", "Tamaño relativo", "tercil",     lambda p,m,b,d,i: (_rank_norm(_ma(p,30))*3).astype(int))
_reg(34, 4, "E", "Tamaño relativo", "log_p30",    lambda p,m,b,d,i: np.log(_ma(p,30).replace(0,np.nan)))
_reg(34, 5, "E", "Tamaño relativo", "rank_YoY",   lambda p,m,b,d,i: _rank_norm(_ma(p,30)).diff(252))
_reg(34, 6, "E", "Tamaño relativo", "stable_lg",  lambda p,m,b,d,i: (_rank_norm(_ma(p,30))>=0.67).astype(float))
_reg(34, 7, "E", "Tamaño relativo", "conc",       lambda p,m,b,d,i: _rank_norm(_ma(p,30))**2)
_reg(34, 8, "E", "Tamaño relativo", "peso_eq",    lambda p,m,b,d,i: pd.Series(1.0/15, index=p.index))
_reg(34, 9, "E", "Tamaño relativo", "peso_prop",  lambda p,m,b,d,i: _rank_norm(_ma(p,30)))

# ── F01: Tasa real ────────────────────────────────────────────────────────────
_reg(35, 0, "F", "Tasa real", "raw",      lambda p,m,b,d,i: _f01_real_rate(p,m,b,d,i))
_reg(35, 1, "F", "Tasa real", ">5%",      lambda p,m,b,d,i: (_f01_real_rate(p,m,b,d,i)>0.05).astype(float))
_reg(35, 2, "F", "Tasa real", ">6%",      lambda p,m,b,d,i: (_f01_real_rate(p,m,b,d,i)>0.06).astype(float))
_reg(35, 3, "F", "Tasa real", ">7%",      lambda p,m,b,d,i: (_f01_real_rate(p,m,b,d,i)>0.07).astype(float))
_reg(35, 4, "F", "Tasa real", ">8%",      lambda p,m,b,d,i: (_f01_real_rate(p,m,b,d,i)>0.08).astype(float))
_reg(35, 5, "F", "Tasa real", "<4%",      lambda p,m,b,d,i: (_f01_real_rate(p,m,b,d,i)<0.04).astype(float))
_reg(35, 6, "F", "Tasa real", "cambio",   lambda p,m,b,d,i: _f01_real_rate(p,m,b,d,i).diff(126))
_reg(35, 7, "F", "Tasa real", "zscore",   lambda p,m,b,d,i: _zscore(_f01_real_rate(p,m,b,d,i)))
_reg(35, 8, "F", "Tasa real", "rank",     lambda p,m,b,d,i: _rank_norm(_f01_real_rate(p,m,b,d,i)))
_reg(35, 9, "F", "Tasa real", "tend",     lambda p,m,b,d,i: _slope(_f01_real_rate(p,m,b,d,i),126))

# ── F02: Spread yield-CETES ───────────────────────────────────────────────────
_f02_threshs = [0,1,2,3,4]
_reg(36, 0, "F", "Spread yield-CETES", "raw",    lambda p,m,b,d,i: _f02_spread(p,m,b,d,i))
for _pi, _th in enumerate(_f02_threshs):
    _reg(36, 1+_pi, "F", "Spread yield-CETES", f">{_th}%",
         (lambda t: lambda p,m,b,d,i: (_f02_spread(p,m,b,d,i)>t/100).astype(float))(_th))
_reg(36, 6, "F", "Spread yield-CETES", "zscore", lambda p,m,b,d,i: _zscore(_f02_spread(p,m,b,d,i)))
_reg(36, 7, "F", "Spread yield-CETES", "rank",   lambda p,m,b,d,i: _rank_norm(_f02_spread(p,m,b,d,i)))
_reg(36, 8, "F", "Spread yield-CETES", "tend",   lambda p,m,b,d,i: _slope(_f02_spread(p,m,b,d,i),126))
_reg(36, 9, "F", "Spread yield-CETES", "neg",    lambda p,m,b,d,i: (_f02_spread(p,m,b,d,i)<0).astype(float))

# ── F03: Fase ciclo de tasas ──────────────────────────────────────────────────
def _bnx_diff(p, b, d):
    if b is None: return pd.Series(np.nan, index=p.index)
    r = b["rate"].reindex(p.index, method="ffill") if "rate" in b.columns else pd.Series(np.nan, index=p.index)
    return r.diff(d)
_reg(37, 0, "F", "Fase tasa", "subiendo",      lambda p,m,b,d,i: (_bnx_diff(p,b,63)>0).astype(float))
_reg(37, 1, "F", "Fase tasa", "bajando",       lambda p,m,b,d,i: (_bnx_diff(p,b,63)<0).astype(float))
_reg(37, 2, "F", "Fase tasa", "pausa",         lambda p,m,b,d,i: (_bnx_diff(p,b,63)==0).astype(float))
_reg(37, 3, "F", "Fase tasa", "acel-suba",     lambda p,m,b,d,i: ((_bnx_diff(p,b,63)>0)&(_bnx_diff(p,b,63)>_bnx_diff(p,b,126))).astype(float))
_reg(37, 4, "F", "Fase tasa", "desacel-suba",  lambda p,m,b,d,i: ((_bnx_diff(p,b,63)>0)&(_bnx_diff(p,b,63)<_bnx_diff(p,b,126))).astype(float))
_reg(37, 5, "F", "Fase tasa", "cum6M",         lambda p,m,b,d,i: _bnx_diff(p,b,126))
_reg(37, 6, "F", "Fase tasa", "vel",           lambda p,m,b,d,i: _bnx_diff(p,b,21))
_reg(37, 7, "F", "Fase tasa", "zscore",        lambda p,m,b,d,i: _zscore(_bnx_diff(p,b,63)))
_reg(37, 8, "F", "Fase tasa", "señal",         lambda p,m,b,d,i: np.sign(_bnx_diff(p,b,63)))
_reg(37, 9, "F", "Fase tasa", "dirección",     lambda p,m,b,d,i: np.sign(_bnx_diff(p,b,126)))

# ── F04: Tendencia USD/MXN ────────────────────────────────────────────────────
_f04_wins = [21,63,126,252]
for _pi, _w in enumerate(_f04_wins):
    _reg(38, _pi, "F", "USD/MXN tend.", f"ret_{_w}d",
         (lambda w: lambda p,m,b,d,i: _f04_usdmxn(p,m,b,d,i,w))(_w))
_reg(38, 4, "F", "USD/MXN tend.", "peso_deb", lambda p,m,b,d,i: (_f04_usdmxn(p,m,b,d,i,63)>0).astype(float))
_reg(38, 5, "F", "USD/MXN tend.", "peso_fue", lambda p,m,b,d,i: (_f04_usdmxn(p,m,b,d,i,63)<0).astype(float))
_reg(38, 6, "F", "USD/MXN tend.", ">+5%",     lambda p,m,b,d,i: (_f04_usdmxn(p,m,b,d,i,63)>0.05).astype(float))
_reg(38, 7, "F", "USD/MXN tend.", "<-5%",     lambda p,m,b,d,i: (_f04_usdmxn(p,m,b,d,i,63)<-0.05).astype(float))
_reg(38, 8, "F", "USD/MXN tend.", "vol",      lambda p,m,b,d,i: _f04_usdmxn(p,m,b,d,i,63).rolling(63,min_periods=30).std() if b is not None and "usdmxn" in b.columns else pd.Series(np.nan,index=p.index))
_reg(38, 9, "F", "USD/MXN tend.", "zscore",   lambda p,m,b,d,i: _zscore(_f04_usdmxn(p,m,b,d,i,63)))

# ── F05: Momentum IPC ─────────────────────────────────────────────────────────
_f05_wins = [21,63,126,252]
for _pi, _w in enumerate(_f05_wins):
    _reg(39, _pi, "F", "Mom. IPC", f"ret_{_w}d",
         (lambda w: lambda p,m,b,d,i: _f05_ipc(p,m,b,d,i,w))(_w))
_reg(39, 4, "F", "Mom. IPC", ">0%",       lambda p,m,b,d,i: (_f05_ipc(p,m,b,d,i,63)>0).astype(float))
_reg(39, 5, "F", "Mom. IPC", ">5%",       lambda p,m,b,d,i: (_f05_ipc(p,m,b,d,i,63)>0.05).astype(float))
_reg(39, 6, "F", "Mom. IPC", ">10%",      lambda p,m,b,d,i: (_f05_ipc(p,m,b,d,i,63)>0.10).astype(float))
_reg(39, 7, "F", "Mom. IPC", "<0%",       lambda p,m,b,d,i: (_f05_ipc(p,m,b,d,i,63)<0).astype(float))
_reg(39, 8, "F", "Mom. IPC", "zscore",    lambda p,m,b,d,i: _zscore(_f05_ipc(p,m,b,d,i,63)))
_reg(39, 9, "F", "Mom. IPC", "IPC>MA200", lambda p,m,b,d,i: (i.reindex(p.index,method="ffill")>_ma(i.reindex(p.index,method="ffill"),200)).astype(float) if i is not None else pd.Series(np.nan,index=p.index))


# ── Verificación: exactamente 400 features ────────────────────────────────────
assert len(FEATURE_REGISTRY) == 400, f"Registry tiene {len(FEATURE_REGISTRY)} features, esperados 400"

GENE_IDS: list[int] = sorted(FEATURE_REGISTRY.keys())
VAR_IDS:  list[int] = list(range(40))
