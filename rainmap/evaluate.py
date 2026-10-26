"""Judge a design by daily continuous simulation on held-out rainfall years.

The optimiser plans on weekly totals of training years; here every unit (shared
cluster or private tank) is simulated day by day on years it never saw, with
fresh roof-area draws. A unit "fails" a scenario-year if it delivers less than
`failure_threshold` of what the plan promised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hydrology import simulate
from .instance import Instance
from .optimize import Design, ScenarioSet


def trench_length(inst: Instance, members: np.ndarray, h: int) -> float:
    """Street length actually dug when member routes share trenches."""
    edges = set()
    for i in members:
        p = inst.paths.get((i, h), [])
        edges.update(tuple(sorted(e)) for e in zip(p[:-1], p[1:]))
    G = inst.G
    street = sum(min(d["length"] for d in G.get_edge_data(u, v, default=G.get_edge_data(v, u)).values())
                 for u, v in edges)
    return street + inst.access[members].sum()


def unit_table(inst: Instance, design: Design, sc: ScenarioSet) -> pd.DataFrame:
    cost, opt = inst.cfg.costs, inst.cfg.opt
    A = cost.annualise
    units = []
    for h in design.open_hubs:
        m = np.nonzero(design.assign == h)[0]
        capex = cost.tank_fixed + cost.hub_pump + cost.tank_per_m3 * design.cap[h] \
            + (inst.link_cost[m, h].sum() / A)
        units.append(dict(unit=f"C{h}", kind="shared", hub_building=int(inst.hubs[h]),
                          members=m, storage_m3=design.cap[h], promise=design.promise_hub[h],
                          capex=capex,
                          pipe_m=float(inst.dist[m, h].sum() * cost.lines_per_link),
                          trench_m=trench_length(inst, m, h),
                          lifts=int((~inst.gravity_ok[m, h]).sum())))
    for i in np.nonzero(design.ind_size > 0)[0]:
        units.append(dict(unit=f"B{i}", kind="private", hub_building=int(i), members=np.array([i]),
                          storage_m3=design.ind_size[i], promise=design.promise_ind[i],
                          capex=cost.tank_fixed + cost.tank_per_m3 * design.ind_size[i],
                          pipe_m=0.0, trench_m=0.0, lifts=0))
    if not units:
        return pd.DataFrame()
    df = pd.DataFrame(units)

    inflow = np.stack([sc.rain * sc.roof[:, m].sum(1)[:, None] for m in df.members])   # (U, S, T)
    demand = np.stack([sc.demand[m].sum(0) for m in df.members])                      # (U, T)
    delivered, met = simulate(inflow, demand[:, None, :], df.storage_m3.values[:, None])
    annual = delivered.sum(-1)                                                        # (U, S)

    df["n_buildings"] = df.members.apply(len)
    df["roof_m2"] = df.members.apply(lambda m: inst.buildings.area_m2.values[m].sum())
    df["demand_m3_yr"] = demand.sum(-1)
    df["delivered_m3_yr"] = annual.mean(1)
    df["delivered_p10"] = np.percentile(annual, 10, axis=1)
    df["supply_ratio"] = df.delivered_m3_yr / df.demand_m3_yr
    df["reliability_days"] = met.mean(1)
    df["failure_prob"] = (annual < opt.failure_threshold * df.promise.values[:, None]).mean(1)
    df["annual_cost"] = df.capex * A
    df["litres_per_rupee"] = df.delivered_m3_yr * 1000 / df.annual_cost
    df["net_value"] = df.delivered_m3_yr * cost.water_value_per_m3 - df.annual_cost
    df.attrs["annual_by_scenario"] = annual
    return df


def summarise(inst: Instance, design: Design, units: pd.DataFrame) -> dict:
    opt = inst.cfg.opt
    total_demand = float(inst.demand_day.sum() * 365)
    if units.empty:
        return dict(method=design.method, delivered_m3_yr=0.0, annual_cost=0.0,
                    litres_per_rupee=0.0, net_value=0.0, supply_ratio=0.0,
                    reliability_days=0.0, failure_prob=1.0, n_shared=0, n_private=0,
                    buildings_served=0, buildings_shared=0, pipe_m=0.0, trench_m=0.0,
                    storage_m3=0.0, promised_m3_yr=0.0, plan_gap=0.0,
                    runtime_s=design.runtime_s, objective=design.objective, status=design.status)
    annual = units.attrs["annual_by_scenario"].sum(0)                                 # (S,)
    promised = float(units.promise.sum())
    w = units.demand_m3_yr / units.demand_m3_yr.sum()
    delivered = float(annual.mean())
    cost = float(units.annual_cost.sum())
    return dict(
        method=design.method,
        delivered_m3_yr=delivered,
        delivered_p10=float(np.percentile(annual, 10)),
        annual_cost=cost,
        capex=float(units.capex.sum()),
        litres_per_rupee=delivered * 1000 / cost if cost else 0.0,
        net_value=float(units.net_value.sum()),
        supply_ratio=delivered / total_demand,
        reliability_days=float((units.reliability_days * w).sum()),
        failure_prob=float((annual < opt.failure_threshold * promised).mean()),
        n_shared=int((units.kind == "shared").sum()),
        n_private=int((units.kind == "private").sum()),
        buildings_served=int(units.n_buildings.sum()),
        buildings_shared=int(units.loc[units.kind == "shared", "n_buildings"].sum()),
        pipe_m=float(units.pipe_m.sum()),
        trench_m=float(units.trench_m.sum()),
        storage_m3=float(units.storage_m3.sum()),
        promised_m3_yr=promised,
        plan_gap=(delivered - promised) / promised if promised else 0.0,
        runtime_s=design.runtime_s,
        objective=design.objective,
        status=design.status,
    )


def sharing_gain(shared_units: pd.DataFrame, private_units: pd.DataFrame) -> pd.DataFrame:
    """Does each cluster beat its members going it alone (private baseline, same test years)?"""
    priv = {int(r.hub_building): r for r in private_units.itertuples()} if not private_units.empty else {}
    out = shared_units.copy()
    alone_value, alone_water, alone_rel = [], [], []
    for m in out.members:
        rows = [priv[i] for i in m if i in priv]
        alone_value.append(sum(r.net_value for r in rows))
        alone_water.append(sum(r.delivered_m3_yr for r in rows))
        dem = sum(r.demand_m3_yr for r in rows)
        alone_rel.append(sum(r.reliability_days * r.demand_m3_yr for r in rows) / dem if dem else 0.0)
    out["alone_net_value"] = alone_value
    out["alone_delivered"] = alone_water
    out["sharing_gain_value"] = out.net_value - out.alone_net_value
    out["sharing_gain_reliability"] = out.reliability_days - np.array(alone_rel)
    out["sharing_pays"] = out.sharing_gain_value > 0
    return out


def rank(units: pd.DataFrame) -> pd.DataFrame:
    """Survey shortlist: cost-effective, reliable clusters first."""
    df = units.copy()
    df["survey_score"] = df.litres_per_rupee * (1 - df.failure_prob) * df.reliability_days
    return df.sort_values("survey_score", ascending=False).reset_index(drop=True)
