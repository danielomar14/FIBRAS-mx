"""
Algoritmo Genético — Experimento 2B.

Búsqueda combinatoria en el espacio de (variable, parametrización):
el cromosoma ganador ES la estrategia, sin ML.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.builder import load_and_build
from src.genetic.chromosome import Individual, ALL_GENES
from src.genetic.operators import crossover, mutate, random_individual, tournament_select
from src.genetic.fitness import evaluate_individual, evaluate_period

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN_START = "2017-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-31"
VAL_START   = "2026-01-01"
VAL_END     = "2026-12-31"


@dataclass
class GAResult:
    best_individual:     Individual
    history_best:        list[float]
    history_mean:        list[float]
    top10:               list[Individual]
    train_metrics:       dict
    test_metrics:        dict
    val_metrics:         dict
    var_frequency:       dict[int, int] = field(default_factory=dict)


def _load_prices_wide() -> pd.DataFrame:
    processed = ROOT / "data" / "processed"
    prices_long = pd.read_parquet(processed / "precios_diarios.parquet")
    pw = prices_long.pivot_table(index="date", columns="ticker", values="close")
    pw.index = pd.to_datetime(pw.index)
    return pw.ffill(limit=5)


def run_ga(
    n_generations:    int = 50,
    population_size:  int = 20,
    top_k_options:    list[int] | None = None,
    elite_size:       int = 2,
    tournament_k:     int = 3,
    no_improve_limit: int = 10,
    diversity_thresh: float = 0.8,
    seed:             int | None = None,
    cache:            bool = True,
    progress_callback = None,
) -> GAResult:
    """
    Corre el GA y devuelve GAResult.

    progress_callback(gen, best_fitness, mean_fitness) → se llama cada generación.
    """
    cache_path = RESULTS / "ga_results.pkl"
    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    log.info("Cargando datos para GA…")
    fm_train = load_and_build(start=TRAIN_START, end=TRAIN_END)
    fm_all   = load_and_build()  # Para test + val
    prices_wide = _load_prices_wide()

    # ── Inicialización ────────────────────────────────────────────────────────
    population: list[Individual] = [random_individual() for _ in range(population_size)]

    for ind in population:
        ind.fitness = evaluate_individual(ind, fm_train, prices_wide, TRAIN_START, TRAIN_END)

    history_best: list[float] = []
    history_mean: list[float] = []
    best_ind = max(population, key=lambda x: x.fitness)
    no_improve = 0

    for gen in range(n_generations):
        fitnesses = [ind.fitness for ind in population if not math.isinf(ind.fitness)]
        best_fit  = max(fitnesses) if fitnesses else -np.inf
        mean_fit  = float(np.mean(fitnesses)) if fitnesses else -np.inf

        history_best.append(best_fit)
        history_mean.append(mean_fit)

        current_best = max(population, key=lambda x: x.fitness)
        if current_best.fitness > best_ind.fitness:
            best_ind = current_best
            no_improve = 0
        else:
            no_improve += 1

        log.info(f"  Gen {gen+1:03d}: best={best_fit:.3f}  mean={mean_fit:.3f}  "
                 f"genes={len(best_ind.genes)}  top_k={best_ind.top_k}")

        if progress_callback:
            progress_callback(gen + 1, best_fit, mean_fit, best_ind)

        if no_improve >= no_improve_limit:
            log.info(f"  Parada temprana: {no_improve_limit} generaciones sin mejora")
            break

        # ── Re-diversificación si >80% población idéntica ─────────────────
        unique_ratio = len(set(population)) / len(population)
        if unique_ratio < (1 - diversity_thresh):
            log.info(f"  Re-diversificación (unique={unique_ratio:.0%})")
            n_immigrants = population_size // 3
            immigrants = [random_individual() for _ in range(n_immigrants)]
            for ind in immigrants:
                ind.fitness = evaluate_individual(ind, fm_train, prices_wide, TRAIN_START, TRAIN_END)
            population = sorted(population, key=lambda x: x.fitness, reverse=True)[:population_size - n_immigrants]
            population += immigrants

        # ── Nueva generación ──────────────────────────────────────────────
        elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:elite_size]
        new_pop = list(elites)

        while len(new_pop) < population_size:
            p1 = tournament_select(population, k=tournament_k)
            p2 = tournament_select(population, k=tournament_k)
            child = crossover(p1, p2)
            child = mutate(child)
            child.fitness = evaluate_individual(child, fm_train, prices_wide, TRAIN_START, TRAIN_END)
            new_pop.append(child)

        population = new_pop

    # ── Resultados finales ────────────────────────────────────────────────────
    top10 = sorted(set(population), key=lambda x: x.fitness, reverse=True)[:10]

    # Frecuencia de variables en top-10
    var_freq: dict[int, int] = {}
    for ind in top10:
        for g in ind.genes:
            var_freq[g.var_id] = var_freq.get(g.var_id, 0) + 1

    # Métricas en todos los períodos
    best_ind = max(population, key=lambda x: x.fitness)
    train_met = evaluate_period(best_ind, fm_train, prices_wide, TRAIN_START, TRAIN_END)
    test_met  = evaluate_period(best_ind, fm_all,   prices_wide, TEST_START,  TEST_END)
    val_met   = evaluate_period(best_ind, fm_all,   prices_wide, VAL_START,   VAL_END)

    result = GAResult(
        best_individual=best_ind,
        history_best=history_best,
        history_mean=history_mean,
        top10=top10,
        train_metrics=train_met,
        test_metrics=test_met,
        val_metrics=val_met,
        var_frequency=var_freq,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    log.info(f"Resultado GA guardado -> {cache_path}")
    log.info(f"Mejor individuo: {best_ind.summary()}")
    log.info(f"Fitness train: {best_ind.fitness:.3f}")

    return result
