"""
Experimento 2A: entrena los 9 modelos sobre la feature matrix completa.
Evalúa en Train, Test y Validation; guarda resultados en results/ml_results.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.builder import load_and_build
from src.ml.models import ALL_MODELS, build_model, feature_importance
from src.ml.cross_val import (
    make_labels,
    walk_forward_cv,
    sharpe_from_returns,
    evaluate_on_period,
)

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_START = "2017-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"
VAL_START   = "2026-01-01"

TOP_K_DEFAULT = 5


def _load_prices_wide() -> pd.DataFrame:
    from pathlib import Path as P
    processed = ROOT / "data" / "processed"
    prices_long = pd.read_parquet(processed / "precios_diarios.parquet")
    pw = prices_long.pivot_table(index="date", columns="ticker", values="close")
    pw.index = pd.to_datetime(pw.index)
    return pw.ffill(limit=5)


def run_all_models(
    model_names: list[str] | None = None,
    top_k: int = TOP_K_DEFAULT,
    progress_callback=None,
    cache: bool = True,
) -> dict:
    """
    Entrena todos los modelos y devuelve dict con métricas.

    Returns
    -------
    {
      model_name: {
        "sharpe_train": float,
        "sharpe_test":  float,
        "cagr_train":   float,
        "cagr_test":    float,
        "max_dd_test":  float,
        "equity_train": pd.Series,
        "equity_test":  pd.Series,
        "feature_importance": dict | None,
      }
    }
    """
    cache_path = RESULTS / "ml_results.pkl"
    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if model_names is None:
        model_names = ALL_MODELS

    log.info("Construyendo feature matrix…")
    fm_train = load_and_build(start=TRAIN_START, end=TRAIN_END)
    fm_test  = load_and_build(start=TEST_START,  end=TEST_END)
    fm_val   = load_and_build(start=VAL_START)

    prices_wide = _load_prices_wide()

    labels_train = make_labels(prices_wide.loc[TRAIN_START:TRAIN_END])
    labels_test  = make_labels(prices_wide.loc[TEST_START:TEST_END])

    train_dates = sorted(fm_train.index.get_level_values("date").unique().tolist())
    test_dates  = sorted(fm_test.index.get_level_values("date").unique().tolist())
    val_dates   = sorted(fm_val.index.get_level_values("date").unique().tolist())

    # Build full train X/y for final model fitting
    X_train_full = fm_train.dropna(how="all", axis=1).fillna(0)
    y_train_full = labels_train.stack().reindex(X_train_full.index).fillna(0.5)
    valid = y_train_full.notna() & X_train_full.notna().all(axis=1)
    X_tr = X_train_full.loc[valid]
    y_tr = y_train_full.loc[valid]

    results = {}

    for idx, name in enumerate(model_names):
        log.info(f"Entrenando {name} ({idx+1}/{len(model_names)})…")
        if progress_callback:
            progress_callback(name, idx, len(model_names))

        model = build_model(name)

        # Walk-forward CV on train for sharpe_train
        cv_model = build_model(name)
        try:
            train_rets = walk_forward_cv(
                fm_train, labels_train, prices_wide, cv_model, top_k=top_k
            )
        except Exception as e:
            log.warning(f"{name} CV failed: {e}")
            train_rets = []

        # Fit on full train
        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            log.warning(f"{name} final fit failed: {e}")
            results[name] = {"sharpe_train": -np.inf, "sharpe_test": -np.inf,
                             "cagr_train": np.nan, "cagr_test": np.nan, "max_dd_test": np.nan}
            continue

        # Evaluate test
        fm_test_aligned = fm_test.reindex(columns=X_tr.columns).fillna(0)
        test_res = evaluate_on_period(
            fm_test_aligned, labels_test, prices_wide, model, test_dates, train_dates, top_k
        )

        # In-sample evaluation on 2021-2023 for the results comparison page.
        # Uses final model (trained on all of 2017-2023) applied to 2021-2023 dates.
        IS_START = pd.Timestamp("2021-01-01")
        is_dates = [d for d in train_dates if d >= IS_START]
        fm_is_aligned = fm_train.reindex(columns=X_tr.columns).fillna(0)
        is_res = evaluate_on_period(
            fm_is_aligned, labels_train, prices_wide, model, is_dates, train_dates, top_k
        )

        fi = feature_importance(model)
        if fi:
            fi = {int(k): float(v) for k, v in fi.items()}

        results[name] = {
            "sharpe_train":       sharpe_from_returns(train_rets),
            "sharpe_test":        test_res["sharpe"],
            "cagr_train":         sharpe_from_returns(train_rets) * 0.1,  # proxy
            "cagr_test":          test_res.get("cagr", np.nan),
            "max_dd_test":        test_res.get("max_dd", np.nan),
            "equity_train":       (1 + pd.Series(train_rets)).cumprod() if train_rets else pd.Series(dtype=float),
            "equity_test":        test_res.get("equity", pd.Series(dtype=float)),
            # Dated returns for the full-comparison chart (2021 → now)
            "returns_dated_is":   is_res.get("returns_dated",  pd.Series(dtype=float)),
            "returns_dated_test": test_res.get("returns_dated", pd.Series(dtype=float)),
            # Full details (tickers per rebalancing date) for per-FIBRA CSV export
            "details_is":         is_res.get("details",   pd.DataFrame()),
            "details_test":       test_res.get("details", pd.DataFrame()),
            "feature_importance": fi,
            "model":              model,
        }
        log.info(f"  {name}: Sharpe_train={results[name]['sharpe_train']:.2f}, "
                 f"Sharpe_test={results[name]['sharpe_test']:.2f}")

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Resultados guardados -> {cache_path}")

    return results
