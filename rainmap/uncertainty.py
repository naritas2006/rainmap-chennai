"""Input uncertainty: imagery-derived roof area and rainfall year.

A scenario is (rainfall year, roof-area draw). Roof area per building is the
footprint area times a lognormal error whose spread grows as the footprint
confidence score falls, and a low-confidence footprint may not be a building at all.
"""
from __future__ import annotations

import numpy as np

from .config import Uncertainty


def roof_draws(area: np.ndarray, conf: np.ndarray, unc: Uncertainty, n_draws: int,
               rng: np.random.Generator) -> np.ndarray:
    """Effective roof areas, shape (n_draws, n_buildings)."""
    sigma = unc.area_sigma_min + unc.area_sigma_slope * (1 - conf)
    factor = rng.lognormal(unc.area_bias, sigma, size=(n_draws, len(area)))
    exists = rng.random((n_draws, len(area))) < np.clip(conf + unc.exist_offset, 0, 1)
    return area * factor * exists


def scenarios(years, n_draws: int, n_keep: int | None, rng: np.random.Generator):
    """All (year, draw) pairs, optionally subsampled to n_keep for the MILP."""
    pairs = [(y, d) for y in years for d in range(n_draws)]
    if n_keep and n_keep < len(pairs):
        idx = rng.choice(len(pairs), n_keep, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs
