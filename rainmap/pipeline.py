"""Run RainMap for the configured Chennai neighbourhoods.

    python -m rainmap.pipeline                       # all neighbourhoods
    python -m rainmap.pipeline --only mylapore --max-buildings 100
    python -m rainmap.pipeline --footprints data/raw/IND.parquet --imd data/raw/imd --dem data/raw/dem.tif

For each neighbourhood: load footprints, streets and rainfall (real where
available, cached; otherwise a clearly labelled synthetic stand-in), build the
instance, solve every design, evaluate on held-out years, and write map-ready
outputs to outputs/<neighbourhood>/ for the dashboard.
"""
from __future__ import annotations

import argparse
import copy
import json
import traceback
import zlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from . import baselines, evaluate, instance, optimize
from .config import NEIGHBOURHOODS, WGS84, Config
from .data import buildings as bld
from .data import rainfall, streets, synthetic
from .ml.rainfall_model import RainfallModel, risk_report, write_report
from .ml.surrogate import YieldSurrogate, accelerated_design

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs"
OB = ROOT / "data" / "raw" / "open_buildings_chennai.parquet"
SENS_SCENARIOS, SENS_TIME_LIMIT = 6, 120


def load_area(nb, cfg: Config, args) -> tuple[gpd.GeoDataFrame, object, dict]:
    sources = {}
    if not args.offline:
        try:
            cache_b = CACHE / f"{nb.name}_buildings.parquet"
            if args.footprints:
                b = bld.from_parquet(Path(args.footprints), nb.lat, nb.lon, nb.radius_m)
                sources["buildings"] = "Open Buildings GeoParquet"
            elif cache_b.exists():
                b = gpd.read_parquet(cache_b)
                sources["buildings"] = "OpenStreetMap (cached)"
            else:
                b = bld.from_osm(nb.lat, nb.lon, nb.radius_m, cfg.uncertainty.osm_confidence)
                CACHE.mkdir(parents=True, exist_ok=True)
                b.to_parquet(cache_b)
                sources["buildings"] = "OpenStreetMap"
            G = streets.fetch_graph(nb.lat, nb.lon, nb.radius_m, CACHE / f"{nb.name}_streets.graphml")
            G = streets.add_elevation(G, args.dem)
            sources["streets"] = "OpenStreetMap via OSMnx"
            sources["terrain"] = "DEM" if args.dem else "flat (no DEM given)"
            if len(b) >= 10:
                return bld.nearest_n(b, nb.lat, nb.lon, args.max_buildings), G, sources
            print(f"  only {len(b)} footprints found, using synthetic stand-in")
        except Exception as e:                                       # network, Overpass, files
            print(f"  real data unavailable ({type(e).__name__}: {str(e)[:120]}), using synthetic")
    seed = zlib.crc32(nb.name.encode()) % 10_000
    b, G = synthetic.neighbourhood(min(args.max_buildings, 120), nb.lat, nb.lon, seed=seed)
    return b, G, {"buildings": "SYNTHETIC", "streets": "SYNTHETIC", "terrain": "SYNTHETIC"}


def routes_gdf(inst, design, units) -> gpd.GeoDataFrame:
    xy = inst.xy
    rows = []
    for h in design.open_hubs:
        k = inst.hubs[h]
        for i in np.nonzero(design.assign == h)[0]:
            if i == k:
                continue
            path = inst.paths.get((i, h), [])
            pts = [tuple(xy[i])] + [(inst.G.nodes[p]["x"], inst.G.nodes[p]["y"]) for p in path] \
                + [tuple(xy[k])]
            rows.append(dict(unit=f"C{h}", building=int(i), length_m=float(inst.dist[i, h]),
                             gravity=bool(inst.gravity_ok[i, h]), geometry=LineString(pts)))
    if not rows:
        return gpd.GeoDataFrame(columns=["unit", "building", "length_m", "gravity", "geometry"],
                                geometry="geometry", crs=inst.buildings.crs).to_crs(WGS84)
    return gpd.GeoDataFrame(rows, crs=inst.buildings.crs).to_crs(WGS84)


def buildings_gdf(inst, design, units) -> gpd.GeoDataFrame:
    g = inst.buildings[["building_id", "area_m2", "confidence", "source", "geometry"]].copy()
    g["demand_m3_yr"] = inst.demand_day * 365
    g["unit"], g["role"] = "none", "none"
    cols = ["storage_m3", "reliability_days", "failure_prob", "litres_per_rupee", "survey_rank"]
    for c in cols:
        g[c] = np.nan
    for r in units.itertuples():
        for i in r.members:
            g.loc[i, "unit"] = r.unit
            g.loc[i, "role"] = "private" if r.kind == "private" else (
                "hub" if i == r.hub_building else "member")
            for c in cols:
                g.loc[i, c] = getattr(r, c, np.nan)
    return g.to_crs(WGS84)


def run_one(nb, cfg: Config, args, gnn_model, scenario_rain: dict | None) -> dict:
    print(f"\n== {nb.name} ==")
    b, G, sources = load_area(nb, cfg, args)
    series, rain_src = rainfall.load_rainfall(list(cfg.train_years) + list(cfg.test_years),
                                              args.imd, seed=cfg.seed)
    sources["rainfall"] = rain_src
    rain = rainfall.by_year(series)
    train = {y: rain[y] for y in cfg.train_years}
    test = {y: rain[y] for y in cfg.test_years}
    inst = instance.build(nb.name, b, G, train, test, cfg, sources)
    print(f"  {inst.n} buildings, {len(inst.hubs)} candidate hubs, {len(inst.links()[0])} links")

    # Design against generated rainfall years; judge on real held-out years.
    sc = optimize.scenario_set(inst, scenario_rain or train, cfg.opt.periods,
                               cfg.uncertainty.n_area_draws, cfg.opt.n_scenarios, cfg.seed)
    ev = optimize.scenario_set(inst, test, 365, args.eval_draws, None, cfg.seed + 1)

    surrogate = YieldSurrogate.fit(sc.rain, seed=cfg.seed)
    designs = {"individual": baselines.individual(inst, sc),
               "design_storm": baselines.design_storm(inst, sc),
               "proximity": baselines.proximity(inst, sc),
               "surrogate": accelerated_design(inst, sc, surrogate)}
    if not args.skip_full_milp:
        designs["full_milp"] = optimize.solve(inst, sc, method="full_milp")
    if gnn_model is not None:
        from . import gnn
        designs["gnn"] = gnn.gnn_design(gnn_model, inst, sc)
    # Recommended design: best exact objective among the optimising methods.
    rec = max([k for k in ("surrogate", "full_milp", "gnn") if k in designs],
              key=lambda k: designs[k].objective)
    designs["optimised"] = designs[rec]

    out = OUT / nb.name
    out.mkdir(parents=True, exist_ok=True)
    summaries, unit_tables = {}, {}
    for name, d in designs.items():
        if name == "optimised":
            summaries[name], unit_tables[name] = summaries[rec], unit_tables[rec]
        else:
            u = evaluate.unit_table(inst, d, ev)
            summaries[name] = evaluate.summarise(inst, d, u)
            unit_tables[name] = u
        if name == "optimised":
            continue
        print(f"  {name:13s} {summaries[name]['delivered_m3_yr']:8.0f} m3/yr  "
              f"{summaries[name]['litres_per_rupee']:5.1f} L/Rs  net Rs {summaries[name]['net_value']:>10,.0f}  "
              f"shared {summaries[name]['buildings_shared']:3d}  ({d.runtime_s:.1f}s)")

    best = unit_tables["optimised"]
    if not best.empty:
        best = evaluate.rank(best)
        best["survey_rank"] = np.arange(1, len(best) + 1)
        shared = best[best.kind == "shared"]
        if not shared.empty:
            g = evaluate.sharing_gain(shared, unit_tables["individual"])
            best = best.merge(g[["unit", "alone_net_value", "sharing_gain_value",
                                 "sharing_gain_reliability", "sharing_pays"]], on="unit", how="left")
    for name, u in unit_tables.items():
        if not u.empty:
            u.drop(columns="members").assign(members=u.members.apply(lambda m: " ".join(map(str, m)))) \
                .to_csv(out / f"units_{name}.csv", index=False)
    if not best.empty:
        best.assign(members=best.members.apply(lambda m: " ".join(map(str, m)))) \
            .to_csv(out / "clusters_ranked.csv", index=False)
    buildings_gdf(inst, designs["optimised"], best if not best.empty else unit_tables["optimised"]) \
        .to_file(out / "buildings.geojson", driver="GeoJSON")
    routes_gdf(inst, designs["optimised"], best).to_file(out / "routes.geojson", driver="GeoJSON")

    # Where does sharing start to pay? Re-solve at other water values with the surrogate
    # (exact daily sizing, so counts are directly comparable with the main design).
    sens = []
    for v in args.water_values:
        c2 = copy.deepcopy(cfg)
        c2.costs.water_value_per_m3 = v
        inst2 = copy.copy(inst)
        inst2.cfg = c2
        d2 = accelerated_design(inst2, sc, surrogate)
        sens.append(dict(water_value=v, hubs=int(d2.hub_open.sum()),
                         buildings_shared=int((d2.assign >= 0).sum()),
                         private=int((d2.ind_size > 0).sum()), objective=d2.objective))
    centre = inst.buildings.to_crs(WGS84).union_all().centroid
    summary = dict(name=nb.name, held_out=nb.held_out, n_buildings=inst.n,
                   centre=[centre.y, centre.x], sources=inst.sources, designs=summaries,
                   recommended=rec, surrogate_fit_mae=surrogate.fit_error,
                   sensitivity=sens, config=cfg.to_dict())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def rainfall_scenarios(cfg: Config, args, n_years: int = 200) -> dict:
    """Fit (or load) the neural rainfall model on training years; write the risk report."""
    series, src = rainfall.load_rainfall(list(cfg.train_years) + list(cfg.test_years),
                                         args.imd, seed=cfg.seed)
    rain = rainfall.by_year(series)
    train = {y: rain[y] for y in cfg.train_years}
    test = {y: rain[y] for y in cfg.test_years}
    path = OUT / "rainfall_model.pt"
    if path.exists():
        model = RainfallModel.load(path)
    else:
        print("fitting seasonal rainfall model ...")
        model = RainfallModel.fit(train, seed=cfg.seed)
        model.save(path)
    report = risk_report(model, train, test, seed=cfg.seed)
    report["source"] = src
    write_report(report, OUT / "rainfall_risk.json")
    print(f"rainfall model: test NLL {report['model_nll_test']:.3f} vs climatology "
          f"{report['climatology_nll_test']:.3f}; P(annual<1000mm) {report['p_annual_below_1000mm']:.0%}")
    sims = model.sample(n_years, seed=cfg.seed + 100)
    return {100_000 + i: sims[i] for i in range(n_years)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-buildings", type=int, default=250)
    ap.add_argument("--footprints", default=str(OB) if OB.exists() else None,
                    help="Open Buildings GeoParquet (default: output of fetch_open_buildings)")
    ap.add_argument("--imd", default=str(ROOT / "data" / "raw" / "imd"))
    ap.add_argument("--dem", help="DEM GeoTIFF for terrain")
    ap.add_argument("--offline", action="store_true", help="skip downloads, use synthetic areas")
    ap.add_argument("--eval-draws", type=int, default=20)
    ap.add_argument("--water-values", type=float, nargs="*", default=[120, 250, 400])
    ap.add_argument("--skip-full-milp", action="store_true", help="surrogate only (much faster)")
    args = ap.parse_args()
    cfg = Config()

    gnn_model = None
    if (OUT / "gnn.pt").exists():
        import torch

        from .gnn import RainGNN
        gnn_model = RainGNN()
        gnn_model.load_state_dict(torch.load(OUT / "gnn.pt"))

    OUT.mkdir(exist_ok=True)
    scenario_rain = rainfall_scenarios(cfg, args) if cfg.scenario_source == "generator" else None
    index_path = OUT / "index.json"
    index = {e["name"]: e for e in json.loads(index_path.read_text())} if index_path.exists() else {}
    for nb in NEIGHBOURHOODS:
        if args.only and nb.name not in args.only:
            continue
        try:
            s = run_one(nb, cfg, args, gnn_model, scenario_rain)
            index[nb.name] = dict(name=s["name"], held_out=s["held_out"], centre=s["centre"],
                                  sources=s["sources"])
        except Exception:
            traceback.print_exc()
    index_path.write_text(json.dumps(list(index.values()), indent=2))
    from . import report
    print(f"wrote {report.write()}")


if __name__ == "__main__":
    main()
