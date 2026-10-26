"""Comparison designs: private tanks, naive proximity grouping, design-storm sizing."""
from __future__ import annotations

import numpy as np

from .hydrology import effective_rain_m
from .instance import Instance
from .optimize import Design, ScenarioSet, individual_yield, solve


def individual(inst: Instance, sc: ScenarioSet) -> Design:
    """Best private tank per building under the same risk-aware objective."""
    return solve(inst, sc, method="individual", allow_shared=False)


def proximity(inst: Instance, sc: ScenarioSet) -> Design:
    """Every building joins the straight-line nearest candidate hub it can be piped to."""
    xy = inst.xy
    hub_xy = xy[inst.hubs]
    d = np.hypot(xy[:, None, 0] - hub_xy[None, :, 0], xy[:, None, 1] - hub_xy[None, :, 1])
    d[~np.isfinite(inst.link_cost)] = np.inf
    assign = np.where(np.isfinite(d.min(1)), d.argmin(1), -1)
    assign[inst.hubs] = np.arange(len(inst.hubs))
    return solve(inst, sc, method="proximity", fixed_assign=close_singletons(inst, assign))


def close_singletons(inst: Instance, assign: np.ndarray) -> np.ndarray:
    """A hub serving only its own roof is just a private tank with a pump; drop it."""
    assign = assign.copy()
    for h, k in enumerate(inst.hubs):
        members = np.nonzero(assign == h)[0]
        if len(members) <= 1 or assign[k] != h:
            assign[members] = -1
    return assign


def design_storm(inst: Instance, sc: ScenarioSet) -> Design:
    """Conventional sizing: private tank holds one median annual-maximum day of runoff."""
    cost = inst.cfg.costs
    ff = inst.cfg.hydrology.first_flush_mm
    depth = np.median([effective_rain_m(r, ff).max() for r in inst.rain_train.values()])
    need = inst.buildings.area_m2.values * inst.runoff * depth
    sizes = np.array([s for s in cost.individual_sizes_m3 if s > 0])
    space = cost.private_m3_per_roof_m2 * inst.buildings.area_m2.values
    fits = (sizes[None, :] <= space[:, None]).sum(1) - 1              # -1: no room at all
    pick = np.minimum(np.searchsorted(sizes, need), np.maximum(fits, 0))
    ind_size = np.where(fits >= 0, sizes[pick], 0.0)
    promise = (individual_yield(sc, sizes)[np.arange(inst.n), pick] * sc.p).sum(-1)
    promise[fits < 0] = 0.0
    H = len(inst.hubs)
    return Design("design_storm", np.full(inst.n, -1), ind_size, np.zeros(H, bool),
                  np.zeros(H), np.zeros(H), promise, np.nan, 0.0, "rule")
