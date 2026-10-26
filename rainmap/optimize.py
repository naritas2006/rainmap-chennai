"""Risk-aware shared-tank design as a capacitated facility-location MILP (HiGHS).

Each building either joins one shared hub tank (piped along streets), gets its
own tank of a standard size, or gets nothing. Hub tanks have a continuous size.
Water balance is tracked per hub, scenario and week. The objective is

    value of water * [(1 - lambda) * E[delivered] + lambda * CVaR_alpha(delivered)]
    - annualised cost (tanks, pumps, pipes)

where the CVaR term is the mean delivery over the worst alpha share of scenarios,
so designs that only pay off in wet years with generous roof estimates are penalised.

Individual tanks are precomputed per (building, size, scenario) with the same
greedy balance, so "everyone on their own" is always a feasible answer and
sharing has to earn its place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .hydrology import effective_rain_m, simulate, to_periods
from .instance import Instance
from .uncertainty import roof_draws, scenarios


@dataclass
class ScenarioSet:
    pairs: list                 # (year, roof draw)
    rain: np.ndarray            # (S, T) effective rain depth per period, m
    roof: np.ndarray            # (S, n) roof area x runoff coefficient, m2
    demand: np.ndarray          # (n, T) m3 per period
    p: np.ndarray               # (S,) probabilities


def scenario_set(inst: Instance, rain_by_year: dict, periods: int, n_draws: int,
                 n_keep: int | None, seed: int) -> ScenarioSet:
    cfg = inst.cfg
    rng = np.random.default_rng(seed)
    b = inst.buildings
    draws = roof_draws(b.area_m2.values, b.confidence.values, cfg.uncertainty, n_draws, rng)
    pairs = scenarios(sorted(rain_by_year), n_draws, n_keep, rng)
    ff = cfg.hydrology.first_flush_mm
    rain = np.stack([to_periods(effective_rain_m(rain_by_year[y], ff), periods) for y, _ in pairs])
    runoff = np.clip(inst.runoff + inst.runoff_sd * rng.standard_normal((n_draws, inst.n)), 0.2, 0.95)
    roof = np.stack([draws[d] * runoff[d] for _, d in pairs])
    days = to_periods(np.ones(365), periods)
    demand = inst.demand_day[:, None] * days[None, :]
    return ScenarioSet(pairs, rain, roof, demand, np.full(len(pairs), 1 / len(pairs)))


def individual_yield(sc: ScenarioSet, sizes) -> np.ndarray:
    """Annual delivery of a private tank, shape (n, M, S)."""
    inflow = sc.rain[:, None, :] * sc.roof[:, :, None]            # (S, n, T)
    out = [simulate(inflow, sc.demand[None], m)[0].sum(-1) for m in sizes]
    return np.stack(out).transpose(2, 0, 1)


@dataclass
class Design:
    method: str
    assign: np.ndarray          # (n,) hub index or -1
    ind_size: np.ndarray        # (n,) private tank m3, 0 = none
    hub_open: np.ndarray        # (H,) bool
    cap: np.ndarray             # (H,) hub tank m3
    promise_hub: np.ndarray     # (H,) planned m3/yr
    promise_ind: np.ndarray     # (n,) planned m3/yr
    objective: float
    runtime_s: float
    status: str

    @property
    def open_hubs(self) -> np.ndarray:
        return np.nonzero(self.hub_open)[0]


class _Rows:
    def __init__(self):
        self.r, self.c, self.v, self.lb, self.ub, self.n = [], [], [], [], [], 0

    def block(self, nrows, lb, ub):
        start = self.n
        self.n += nrows
        self.lb.append(np.broadcast_to(np.asarray(lb, float), (nrows,)))
        self.ub.append(np.broadcast_to(np.asarray(ub, float), (nrows,)))
        return start

    def put(self, rows, cols, vals):
        rows, cols = np.ravel(rows), np.ravel(cols)
        self.r.append(rows)
        self.c.append(cols)
        self.v.append(np.broadcast_to(np.asarray(vals, float), rows.shape).ravel()
                      if np.ndim(vals) == 0 else np.ravel(vals))

    def constraint(self, ncols):
        A = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                       shape=(self.n, ncols)).tocsr()
        return LinearConstraint(A, np.concatenate(self.lb), np.concatenate(self.ub))


def solve(inst: Instance, sc: ScenarioSet, method: str = "milp",
          fixed_assign: np.ndarray | None = None, allow_shared: bool = True,
          allow_individual: bool = True, time_limit: float | None = None) -> Design:
    cfg, cost, opt = inst.cfg, inst.cfg.costs, inst.cfg.opt
    A = cost.annualise
    V, lam, alpha = cost.water_value_per_m3, opt.risk_lambda, opt.cvar_alpha
    n, H = inst.n, len(inst.hubs)
    S, T = sc.rain.shape
    li, lh = inst.links()
    nL = len(li)
    sizes = np.array([s for s in cost.individual_sizes_m3 if s > 0])
    M = len(sizes)
    Dind = individual_yield(sc, sizes)                              # (n, M, S)
    tmax = cost.tank_max_m3

    # Variable layout.
    ox = 0
    oy = ox + nL
    oz = oy + H
    ocap = oz + n * M
    odem = ocap + H
    oroof = odem + H * T
    ost = oroof + H * S
    oyd = ost + H * S * T
    oeta = oyd + H * S * T
    ou = oeta + 1
    N = ou + S
    zi = oz + np.arange(n * M).reshape(n, M)
    dem = odem + np.arange(H * T).reshape(H, T)
    roof = oroof + np.arange(H * S).reshape(H, S)
    st = ost + np.arange(H * S * T).reshape(H, S, T)
    yd = oyd + np.arange(H * S * T).reshape(H, S, T)
    uu = ou + np.arange(S)

    R = _Rows()
    # One option per building.
    r0 = R.block(n, -np.inf, 1)
    R.put(r0 + li, ox + np.arange(nL), 1)
    R.put(r0 + np.repeat(np.arange(n), M), zi, 1)
    # Link only to an open hub; an open hub serves its own roof.
    r0 = R.block(nL, -np.inf, 0)
    R.put(r0 + np.arange(nL), ox + np.arange(nL), 1)
    R.put(r0 + np.arange(nL), oy + lh, -1)
    self_link = np.array([np.nonzero((li == k) & (lh == h))[0][0] for h, k in enumerate(inst.hubs)])
    r0 = R.block(H, -np.inf, 0)
    R.put(r0 + np.arange(H), oy + np.arange(H), 1)
    R.put(r0 + np.arange(H), ox + self_link, -1)
    # Hub capacity only if open.
    r0 = R.block(H, -np.inf, 0)
    R.put(r0 + np.arange(H), ocap + np.arange(H), 1)
    R.put(r0 + np.arange(H), oy + np.arange(H), -tmax)
    # Cluster demand and roof area are sums over linked buildings.
    r0 = R.block(H * T, 0, 0)
    R.put(r0 + np.arange(H * T), dem.ravel(), 1)
    R.put(r0 + lh[:, None] * T + np.arange(T)[None], np.repeat(ox + np.arange(nL), T),
          -sc.demand[li].ravel())
    r0 = R.block(H * S, 0, 0)
    R.put(r0 + np.arange(H * S), roof.ravel(), 1)
    R.put(r0 + lh[:, None] * S + np.arange(S)[None], np.repeat(ox + np.arange(nL), S),
          -sc.roof[:, li].T.ravel())
    # Storage balance (cyclic year): st_t <= st_{t-1} + inflow_t - delivered_t.
    rows = np.arange(H * S * T)
    r0 = R.block(H * S * T, -np.inf, 0)
    R.put(r0 + rows, st.ravel(), 1)
    R.put(r0 + rows, np.roll(st, 1, axis=2).ravel(), -1)
    R.put(r0 + rows, yd.ravel(), 1)
    R.put(r0 + rows, np.repeat(roof.ravel(), T), -np.tile(sc.rain.ravel(), H))
    # Deliver no more than demand; store no more than capacity.
    r0 = R.block(H * S * T, -np.inf, 0)
    R.put(r0 + rows, yd.ravel(), 1)
    R.put(r0 + rows, np.broadcast_to(dem[:, None, :], (H, S, T)).ravel(), -1)
    r0 = R.block(H * S * T, -np.inf, 0)
    R.put(r0 + rows, st.ravel(), 1)
    R.put(r0 + rows, np.repeat(ocap + np.arange(H), S * T), -1)
    # CVaR: u_s >= eta - delivered_s.
    r0 = R.block(S, -np.inf, 0)
    R.put(r0 + np.arange(S), np.full(S, oeta), 1)
    R.put(r0 + np.arange(S), uu, -1)
    R.put(r0 + np.broadcast_to(np.arange(S)[None, :, None], (H, S, T)), yd, -1)
    R.put(r0 + np.broadcast_to(np.arange(S)[None, None, :], (n, M, S)),
          np.broadcast_to(zi[:, :, None], (n, M, S)), -Dind)
    cons = R.constraint(N)

    c = np.zeros(N)
    c[ox:oy] = inst.link_cost[li, lh]
    c[oy:oz] = (cost.tank_fixed + cost.hub_pump) * A
    c[zi] = (cost.tank_fixed + cost.tank_per_m3 * sizes)[None] * A \
        - V * (1 - lam) * (Dind * sc.p).sum(-1)
    c[ocap:odem] = cost.tank_per_m3 * A
    c[yd] = -V * (1 - lam) * sc.p[None, :, None]
    c[oeta] = -V * lam
    c[uu] = V * lam * sc.p / alpha

    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    ub[ox:ocap] = 1
    ub[ocap:odem] = tmax
    ub[st] = tmax
    lb[oeta] = -np.inf
    space = cost.private_m3_per_roof_m2 * inst.buildings.area_m2.values
    ub[zi[sizes[None, :] > space[:, None]]] = 0
    if not allow_shared:
        ub[ox:oz] = 0
    if not allow_individual:
        ub[zi] = 0
    if fixed_assign is not None:
        on = (fixed_assign[li] == lh).astype(float)
        lb[ox:oy] = ub[ox:oy] = on
        hub_on = np.zeros(H)
        hub_on[lh[on > 0]] = 1
        lb[oy:oz] = ub[oy:oz] = hub_on
        ub[zi[fixed_assign >= 0]] = 0
    integrality = np.zeros(N)
    integrality[ox:ocap] = 1

    t0 = time.perf_counter()
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit or opt.time_limit_s,
                        "mip_rel_gap": opt.mip_gap, "disp": False})
    runtime = time.perf_counter() - t0
    if res.x is None:
        raise RuntimeError(f"{method}: solver returned no solution ({res.message})")
    x = res.x

    assign = np.full(n, -1)
    on = x[ox:oy] > 0.5
    assign[li[on]] = lh[on]
    zsel = x[zi] > 0.5
    ind_size = np.where(zsel.any(1), sizes[zsel.argmax(1)], 0.0)
    ind_size[assign >= 0] = 0.0
    hub_open = x[oy:oz] > 0.5
    cap = np.where(hub_open, x[ocap:odem], 0.0)
    promise_hub = (x[yd].sum(-1) * sc.p).sum(-1)
    promise_ind = np.where(zsel.any(1), (Dind[np.arange(n), zsel.argmax(1)] * sc.p).sum(-1), 0.0)
    promise_ind[assign >= 0] = 0.0
    status = "optimal" if res.status == 0 else res.message
    return Design(method, assign, ind_size, hub_open, cap, promise_hub, promise_ind,
                  -float(res.fun), runtime, status)
