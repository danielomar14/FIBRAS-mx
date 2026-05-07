"""
Definición del cromosoma para el Algoritmo Genético (2B).

Gene   = (var_id, param_idx)  → identifica uno de los 400 features
Individual = lista de 1-5 genes + top_k
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass, field

from src.features.registry import VAR_IDS

Gene = namedtuple("Gene", ["var_id", "param_idx"])

ALL_GENES: list[Gene] = [
    Gene(var_id=v, param_idx=p)
    for v in VAR_IDS
    for p in range(10)
]

GENE_ID_MAP: dict[Gene, int] = {g: g.var_id * 10 + g.param_idx for g in ALL_GENES}
FEATURE_ID_MAP: dict[int, Gene] = {v: k for k, v in GENE_ID_MAP.items()}


@dataclass
class Individual:
    genes:   list[Gene]
    top_k:   int
    fitness: float = field(default=-math.inf)

    def feature_ids(self) -> list[int]:
        return [GENE_ID_MAP[g] for g in self.genes]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Individual):
            return False
        return (sorted(self.genes) == sorted(other.genes) and
                self.top_k == other.top_k)

    def __hash__(self) -> int:
        return hash((tuple(sorted(self.genes)), self.top_k))

    def summary(self) -> str:
        from src.features.registry import FEATURE_REGISTRY
        parts = []
        for g in self.genes:
            fid = GENE_ID_MAP[g]
            fd  = FEATURE_REGISTRY.get(fid)
            label = f"{fd.var_name}[{fd.param_name}]" if fd else f"gene({fid})"
            parts.append(label)
        return f"top_k={self.top_k}  genes=[{', '.join(parts)}]"
