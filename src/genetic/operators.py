"""
Operadores genéticos: inicialización, selección, cruzamiento, mutación.
"""

from __future__ import annotations

import random

from src.genetic.chromosome import ALL_GENES, Gene, Individual

TOP_K_OPTIONS = [3, 4, 5, 6, 7]


def random_individual() -> Individual:
    n = random.choice([1, 2, 3, 4, 5])
    genes = _sample_genes(n)
    return Individual(genes=genes, top_k=random.choice(TOP_K_OPTIONS))


def _sample_genes(n: int, exclude_var_ids: set[int] | None = None) -> list[Gene]:
    """Muestra n genes sin repetir var_id."""
    pool = ALL_GENES if not exclude_var_ids else [g for g in ALL_GENES if g.var_id not in exclude_var_ids]
    if len(pool) == 0:
        return []
    chosen_var_ids: set[int] = set()
    result: list[Gene] = []
    candidates = list(pool)
    random.shuffle(candidates)
    for g in candidates:
        if g.var_id not in chosen_var_ids:
            result.append(g)
            chosen_var_ids.add(g.var_id)
        if len(result) == n:
            break
    return result


def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def crossover(p1: Individual, p2: Individual) -> Individual:
    pool = list(set(p1.genes) | set(p2.genes))
    n = random.randint(1, 5)
    # avoid duplicate var_ids
    seen: set[int] = set()
    chosen: list[Gene] = []
    random.shuffle(pool)
    for g in pool:
        if g.var_id not in seen:
            chosen.append(g)
            seen.add(g.var_id)
        if len(chosen) == n:
            break
    if not chosen:
        chosen = _sample_genes(1)
    top_k = random.choice([p1.top_k, p2.top_k])
    return Individual(genes=chosen, top_k=top_k)


def mutate(ind: Individual, p_add: float = 0.3, p_remove: float = 0.3,
           p_swap: float = 0.3, p_param: float = 0.3, p_topk: float = 0.2) -> Individual:
    genes = list(ind.genes)
    top_k = ind.top_k

    if p_add > 0 and len(genes) < 5 and random.random() < p_add:
        existing_var_ids = {g.var_id for g in genes}
        new_genes = _sample_genes(1, exclude_var_ids=existing_var_ids)
        if new_genes:
            genes.append(new_genes[0])

    if p_remove > 0 and len(genes) > 1 and random.random() < p_remove:
        genes.pop(random.randrange(len(genes)))

    if p_swap > 0 and genes and random.random() < p_swap:
        idx = random.randrange(len(genes))
        existing_var_ids = {g.var_id for g in genes if g != genes[idx]}
        new_genes = _sample_genes(1, exclude_var_ids=existing_var_ids)
        if new_genes:
            genes[idx] = new_genes[0]

    if p_param > 0 and genes and random.random() < p_param:
        idx = random.randrange(len(genes))
        old = genes[idx]
        genes[idx] = Gene(var_id=old.var_id, param_idx=random.randint(0, 9))

    if p_topk > 0 and random.random() < p_topk:
        top_k = random.choice(TOP_K_OPTIONS)

    return Individual(genes=genes if genes else ind.genes, top_k=top_k)
