"""
Instancia los 9 modelos de ML con sus hiperparámetros por defecto.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


class LogisticRegressionRanker(BaseEstimator, RegressorMixin):
    """
    Usa LogisticRegression sobre labels binarizados (top-half = 1).
    predict() devuelve la probabilidad de clase 1, usable como score de ranking.
    """

    def __init__(self, C: float = 1.0, solver: str = "lbfgs", max_iter: int = 200):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self._model = None
        self._scaler = StandardScaler()

    def fit(self, X, y):
        y_bin = (np.array(y) >= 0.5).astype(int)
        X_s = self._scaler.fit_transform(X)
        self._model = LogisticRegression(
            C=self.C, solver=self.solver, max_iter=self.max_iter
        )
        self._model.fit(X_s, y_bin)
        return self

    def predict(self, X):
        X_s = self._scaler.transform(X)
        return self._model.predict_proba(X_s)[:, 1]

    @property
    def coef_(self):
        return self._model.coef_[0] if self._model else None


MODEL_DEFAULTS: dict[str, dict] = {
    "DecisionTree": {
        "max_depth": 3,
        "min_samples_split": 10,
    },
    "Ridge": {
        "alpha": 1.0,
    },
    "LogisticRegression": {
        "C": 1.0,
        "solver": "lbfgs",
    },
    "ElasticNet": {
        "alpha": 0.1,
        "l1_ratio": 0.5,
    },
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 4,
        "n_jobs": -1,
        "random_state": 42,
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    },
    "CatBoost": {
        "iterations": 100,
        "depth": 3,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    },
    "LightGBM": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": -1,
    },
    "ExtraTrees": {
        "n_estimators": 100,
        "max_depth": 4,
        "n_jobs": -1,
        "random_state": 42,
    },
}


def build_model(name: str, params: dict | None = None):
    """Retorna un estimador sklearn-compatible dado nombre y parámetros."""
    p = {**MODEL_DEFAULTS.get(name, {}), **(params or {})}

    if name == "DecisionTree":
        return DecisionTreeRegressor(**p)
    if name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**p))])
    if name == "LogisticRegression":
        return LogisticRegressionRanker(**p)
    if name == "ElasticNet":
        return Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(**p))])
    if name == "RandomForest":
        return RandomForestRegressor(**p)
    if name == "XGBoost":
        return XGBRegressor(**p)
    if name == "CatBoost":
        return CatBoostRegressor(**p)
    if name == "LightGBM":
        return LGBMRegressor(**p)
    if name == "ExtraTrees":
        return ExtraTreesRegressor(**p)
    raise ValueError(f"Unknown model: {name}")


ALL_MODELS = list(MODEL_DEFAULTS.keys())


def feature_importance(model) -> dict[int, float] | None:
    """Extrae feature importances si el modelo las soporta."""
    if hasattr(model, "feature_importances_"):
        return dict(enumerate(model.feature_importances_))
    if isinstance(model, LogisticRegressionRanker) and model.coef_ is not None:
        return dict(enumerate(abs(model.coef_)))
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if inner and hasattr(inner, "coef_"):
            coef = inner.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return dict(enumerate(abs(coef)))
    return None
