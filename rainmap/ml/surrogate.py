"""Learned yield surrogate that shrinks the MILP, with exact repair for feasibility.

Why this works: for one rainfall year, the most water a tank can deliver is the
optimal value of a linear program whose right-hand side is linear in pooled roof
area R, daily demand D and storage S. An LP's optimal value is concave in its
right-hand side, and doubling R, D and S doubles the delivery, so

    Y(R, D, S) = D * g(R / D, S / D),   g concave and non-decreasing.

Any such function is a minimum of planes through the origin, so we learn

    Y(R, D, S) ~ min_k  a_k R + b_k D + c_k S        (a, b, c >= 0)

per rainfall year from exact daily simulations. In the MILP each plane is one
linear constraint per hub and scenario, replacing the hub x scenario x day water
balance (hundreds of thousands of variables) with a few thousand rows.

Feasibility is kept by construction: the surrogate only proposes which
buildings share which hub; the clustering is then re-sized with the exact daily
model (``optimize.solve(fixed_assign=...)``), so every reported design is
feasible and costed exactly like the full MILP's.
"""
from __future__ import annotations

import time

import numpy as np
import pyomo.environ as pyo
import torch

from ..hydrology import simulate, to_periods
from ..instance import Instance
from ..optimize import Design, ScenarioSet, individual_yield, solve


class YieldSurrogate:
    """Min-of-planes yield model, one set of K planes per rainfall series."""

    def __init__(self, planes: np.ndarray, fit_error: float):
        self.planes = planes            # (S, K, 3): a (per m2 roof), b (per m3/day demand), c (per m3 storage)
        self.fit_error = fit_error      # mean absolute error, fraction of annual demand

    @classmethod
    def fit(cls, rain: np.ndarray, k: int = 8, n_points: int = 600, steps: int = 800,
            seed: int = 0) -> "YieldSurrogate":
        """rain: (S, T) effective daily rain depth (m) for each scenario."""
        rng = np.random.default_rng(seed)
        T = rain.shape[1]
        # Dimensionless design points: roof per unit daily demand, storage in days of demand.
        r = np.exp(rng.uniform(np.log(5), np.log(3000), n_points))
        s = np.where(rng.random(n_points) < 0.1, 0.0, np.exp(rng.uniform(np.log(0.2), np.log(400), n_points)))
        uniq, inv = np.unique(rain, axis=0, return_inverse=True)
        days = to_periods(np.ones(365), T)                  # steps may be daily or multi-day
        delivered, _ = simulate(uniq[:, None, :] * r[None, :, None], days[None, None, :], s[None, :])
        frac = delivered.sum(-1) / days.sum()               # (U, n_points) share of annual demand met
        coef, err = _fit_planes(r, s, frac, k, steps, seed)
        # frac = min(a r + b + c s) with r = R/D, s = S/D (D per day)  =>  Y = 365 min(a R + b D + c S)
        return cls(coef[np.ravel(inv)] * days.sum(), float(err))

    def predict(self, R, D, S) -> np.ndarray:
        """Annual delivery, shape (S_scen, ...) for broadcastable R, D, S."""
        R, D, S = (np.asarray(v, float) for v in (R, D, S))
        a, b, c = (self.planes[..., j] for j in range(3))   # (S_scen, K)
        shape = (slice(None), slice(None)) + (None,) * np.broadcast(R, D, S).ndim
        return (a[shape] * R + b[shape] * D + c[shape] * S).min(1)


def _fit_planes(r, s, frac, k, steps, seed):
    """Fit g_u(r, s) = min_k a_uk r + b_uk + c_uk s for every series u at once (smooth-min annealing)."""
    torch.manual_seed(seed)
    U = frac.shape[0]
    rs, ss = float(np.median(r)), float(np.median(s[s > 0]))
    rt = torch.tensor(r / rs, dtype=torch.float32)
    st = torch.tensor(s / ss, dtype=torch.float32)
    yt = torch.tensor(frac, dtype=torch.float32)
    raw = (torch.randn(U, k - 1, 3) * 0.5 - 1.0).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=0.05)
    cap = torch.tensor([0.0, 1.0, 0.0]).expand(U, 1, 3)    # never more than demand

    def planes():
        return torch.cat([torch.nn.functional.softplus(raw), cap], 1)       # (U, K, 3)

    for i in range(steps):
        c = planes()
        val = c[:, None, :, 0] * rt[None, :, None] + c[:, None, :, 1] + c[:, None, :, 2] * st[None, :, None]
        tau = max(0.05 * (1 - i / steps), 1e-3)
        loss = ((-tau * torch.logsumexp(-val / tau, -1) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        coef = planes().numpy() / np.array([rs, 1.0, ss])
    pred = (coef[:, None, :, 0] * r[None, :, None] + coef[:, None, :, 1]
            + coef[:, None, :, 2] * s[None, :, None]).min(-1)
    return coef, float(np.abs(pred - frac).mean())


def solve_surrogate(inst: Instance, sc: ScenarioSet, sur: YieldSurrogate,
                    time_limit: float | None = None) -> tuple[np.ndarray, float, float]:
    """Pyomo + HiGHS facility-location MILP with surrogate yields.

    Returns (assignment, surrogate objective, solve seconds).
    """
    cfg, cost, opt = inst.cfg, inst.cfg.costs, inst.cfg.opt
    A, V, lam, alpha = cost.annualise, cost.water_value_per_m3, opt.risk_lambda, opt.cvar_alpha
    n, H, S = inst.n, len(inst.hubs), len(sc.p)
    li, lh = inst.links()
    sizes = np.array([v for v in cost.individual_sizes_m3 if v > 0])
    Dind = individual_yield(sc, sizes)                              # (n, M, S)
    space = cost.private_m3_per_roof_m2 * inst.buildings.area_m2.values
    allowed = [(i, m) for i in range(n) for m in range(len(sizes)) if sizes[m] <= space[i]]
    links_of_hub = {h: np.nonzero(lh == h)[0].tolist() for h in range(H)}
    links_of_bld = {i: np.nonzero(li == i)[0].tolist() for i in range(n)}
    self_link = {h: int(np.nonzero((li == k) & (lh == h))[0][0]) for h, k in enumerate(inst.hubs)}
    K = sur.planes.shape[1]

    m = pyo.ConcreteModel()
    m.L, m.H, m.S, m.K = (pyo.RangeSet(0, v - 1) for v in (len(li), H, S, K))
    m.Z = pyo.Set(initialize=allowed, dimen=2)
    m.x = pyo.Var(m.L, domain=pyo.Binary)
    m.y = pyo.Var(m.H, domain=pyo.Binary)
    m.z = pyo.Var(m.Z, domain=pyo.Binary)
    m.cap = pyo.Var(m.H, bounds=(0, cost.tank_max_m3))
    m.Y = pyo.Var(m.H, m.S, domain=pyo.NonNegativeReals)            # hub delivery per scenario
    m.eta = pyo.Var()
    m.u = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    z_of_bld = {i: [j for (ii, j) in allowed if ii == i] for i in range(n)}

    def one(m, i):
        terms = [m.x[l] for l in links_of_bld[i]] + [m.z[i, j] for j in z_of_bld[i]]
        return sum(terms) <= 1 if len(terms) > 1 else pyo.Constraint.Skip
    m.one = pyo.Constraint(range(n), rule=one)
    m.open = pyo.Constraint(m.L, rule=lambda m, l: m.x[l] <= m.y[int(lh[l])])
    m.own = pyo.Constraint(m.H, rule=lambda m, h: m.y[h] <= m.x[self_link[h]])
    m.capon = pyo.Constraint(m.H, rule=lambda m, h: m.cap[h] <= cost.tank_max_m3 * m.y[h])
    demand_day = inst.demand_day

    def plane(m, h, s, k):
        a, b, c = sur.planes[s, k]
        R = sum(sc.roof[s, li[l]] * m.x[l] for l in links_of_hub[h])
        D = sum(demand_day[li[l]] * m.x[l] for l in links_of_hub[h])
        return m.Y[h, s] <= a * R + b * D + c * m.cap[h]
    m.plane = pyo.Constraint(m.H, m.S, m.K, rule=plane)

    def delivered(m, s):
        return sum(m.Y[h, s] for h in m.H) + sum(Dind[i, j, s] * m.z[i, j] for (i, j) in allowed)
    m.cvar = pyo.Constraint(m.S, rule=lambda m, s: m.u[s] >= m.eta - delivered(m, s))

    cost_expr = (sum(inst.link_cost[li[l], lh[l]] * m.x[l] for l in m.L)
                 + sum((cost.tank_fixed + cost.hub_pump) * A * m.y[h] + cost.tank_per_m3 * A * m.cap[h]
                       for h in m.H)
                 + sum((cost.tank_fixed + cost.tank_per_m3 * sizes[j]) * A * m.z[i, j] for (i, j) in allowed))
    water = (1 - lam) * sum(sc.p[s] * delivered(m, s) for s in m.S) \
        + lam * (m.eta - sum(sc.p[s] * m.u[s] for s in m.S) / alpha)
    m.obj = pyo.Objective(expr=V * water - cost_expr, sense=pyo.maximize)

    solver = pyo.SolverFactory("appsi_highs")
    solver.config.time_limit = time_limit or opt.time_limit_s
    solver.config.mip_gap = opt.mip_gap
    solver.config.load_solution = True
    t0 = time.perf_counter()
    solver.solve(m)
    secs = time.perf_counter() - t0

    assign = np.full(n, -1)
    for l in m.L:
        if pyo.value(m.x[l]) > 0.5:
            assign[li[l]] = lh[l]
    return assign, float(pyo.value(m.obj)), secs


def accelerated_design(inst: Instance, sc: ScenarioSet, sur: YieldSurrogate | None = None) -> Design:
    """Surrogate MILP proposes clusters; exact daily model sizes them (always feasible)."""
    from ..baselines import close_singletons
    t0 = time.perf_counter()
    sur = sur or YieldSurrogate.fit(sc.rain, seed=inst.cfg.seed)
    t_fit = time.perf_counter() - t0
    assign, sur_obj, t_milp = solve_surrogate(inst, sc, sur)
    d = solve(inst, sc, method="surrogate", fixed_assign=close_singletons(inst, assign))
    d.status = (f"surrogate fit {t_fit:.1f}s + MILP {t_milp:.1f}s + exact sizing {d.runtime_s:.1f}s; "
                f"surrogate obj {sur_obj:,.0f}, fit MAE {sur.fit_error:.3f}")
    d.runtime_s += t_fit + t_milp
    return d
