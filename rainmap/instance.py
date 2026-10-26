"""A neighbourhood ready for optimisation: buildings, candidate hubs, routed links."""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from scipy.cluster.vq import kmeans2

from .config import Config
from .data.streets import route_matrix, snap


@dataclass
class Instance:
    name: str
    buildings: gpd.GeoDataFrame
    G: object
    hubs: np.ndarray            # building index of each candidate hub
    dist: np.ndarray            # (n, H) street-routed metres, inf = no link
    paths: dict                 # (i, h) -> street node list
    gravity_ok: np.ndarray      # (n, H) roof-to-tank gravity flow possible
    access: np.ndarray          # (n,) building-to-street leg, m
    link_cost: np.ndarray       # (n, H) annualised Rs, inf = no link
    runoff: np.ndarray          # expected runoff coefficient
    runoff_sd: np.ndarray       # its uncertainty (roof material)
    demand_day: np.ndarray      # m3/day per building
    elev: np.ndarray
    rain_train: dict
    rain_test: dict
    cfg: Config
    sources: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.buildings)

    @property
    def xy(self) -> np.ndarray:
        c = self.buildings.centroid
        return np.c_[c.x.values, c.y.values]

    def links(self) -> tuple[np.ndarray, np.ndarray]:
        """(building index, hub index) pairs that can be piped."""
        i, h = np.nonzero(np.isfinite(self.link_cost))
        return i, h


def pick_hubs(xy: np.ndarray, area: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Spread candidate hubs: k-means centres, snapped to the largest nearby roof."""
    k = int(np.clip(k, 1, len(xy)))
    centres, _ = kmeans2(xy, k, seed=seed, minit="++")
    hubs = []
    for c in centres:
        d = np.hypot(*(xy - c).T)
        near = np.argsort(d)[:3]
        best = near[np.argmax(area[near])]
        if best not in hubs:
            hubs.append(best)
    return np.array(sorted(hubs))


def build(name: str, buildings: gpd.GeoDataFrame, G, rain_train: dict, rain_test: dict,
          cfg: Config, sources: dict | None = None) -> Instance:
    hyd, cost = cfg.hydrology, cfg.costs
    c = buildings.centroid
    xy = np.c_[c.x.values, c.y.values]
    area = buildings.area_m2.values
    nodes, access = snap(G, xy[:, 0], xy[:, 1])
    elev = np.array([G.nodes[v].get("elevation", 0.0) for v in nodes], float)

    hubs = pick_hubs(xy, area, round(cfg.opt.hub_fraction * len(xy)), cfg.seed)
    dist, paths = route_matrix(G, nodes, access, hubs, cost.max_link_m)

    # Roof outlet must sit above the hub tank inlet after the pipe's fall.
    head = elev[:, None] + cost.roof_head_m - elev[hubs][None, :] - cost.pipe_slope * dist
    gravity_ok = head >= 0
    per_m = cost.trench_per_m + cost.pipe_per_m * cost.lines_per_link
    capex = dist * per_m + np.where(gravity_ok, 0, cost.lift_pump)
    link_cost = np.where(np.isfinite(dist), capex * cost.annualise, np.inf)
    for h, k in enumerate(hubs):
        link_cost[k, h] = 0.0

    roof = buildings["roof_type"] if "roof_type" in buildings else [None] * len(xy)
    prior = np.array([hyd.runoff_by_roof.get(r, hyd.default_runoff_coeff) for r in roof])
    from .ml.roof_cnn import building_runoff
    runoff, runoff_sd, roof_status = building_runoff(buildings, name, prior)
    sources = {**(sources or {}), "roof_material": roof_status}
    people = area * hyd.floors_default / hyd.m2_per_person
    demand_day = people * hyd.demand_lpcd / 1000.0

    return Instance(name, buildings.reset_index(drop=True), G, hubs, dist, paths, gravity_ok,
                    access, link_cost, runoff, runoff_sd, demand_day, elev, rain_train, rain_test, cfg,
                    sources)
