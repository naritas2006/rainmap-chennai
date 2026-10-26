"""Tank water balance.

Each step: water available = storage + inflow; deliver min(demand, available);
keep the rest up to capacity and spill the excess. With no losses this greedy rule
maximises total delivery, so it gives the same answer as the MILP's optimal
operation, and the planning model and the simulator are consistent.
"""
from __future__ import annotations

import numpy as np


def effective_rain_m(rain_mm: np.ndarray, first_flush_mm: float) -> np.ndarray:
    """Depth of rain that reaches the tank per unit roof area, in metres."""
    return np.clip(rain_mm - first_flush_mm, 0.0, None) / 1000.0


def to_periods(daily: np.ndarray, periods: int) -> np.ndarray:
    """Sum the last axis (days) into `periods` near-equal chunks."""
    edges = np.linspace(0, daily.shape[-1], periods + 1).astype(int)
    return np.add.reduceat(daily, edges[:-1], axis=-1)


def simulate(inflow: np.ndarray, demand: np.ndarray, capacity: np.ndarray,
             warmup: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over leading axes; the last axis is time.

    inflow, demand: (..., T); capacity broadcastable to (...).
    Returns (delivered per step (..., T), fraction of steps with demand fully met (...)).
    A warm-up pass over the same series sets the starting storage (cyclic year).
    """
    inflow, demand = np.broadcast_arrays(inflow, demand)
    cap = np.broadcast_to(np.asarray(capacity, float), inflow.shape[:-1])
    store = np.zeros(inflow.shape[:-1])
    out = np.empty(inflow.shape)
    for p in range(2 if warmup else 1):
        for t in range(inflow.shape[-1]):
            avail = store + inflow[..., t]
            d = np.minimum(demand[..., t], avail)
            store = np.minimum(avail - d, cap)
            if p == (1 if warmup else 0):
                out[..., t] = d
    met = (out >= demand - 1e-9).mean(axis=-1)
    return out, met
