"""Generate solver-labelled instances, train the GNN, benchmark it against the MILP.

    python -m rainmap.train_gnn --train 200 --val 30 --test 20

Training instances are small synthetic neighbourhoods (20-60 buildings) with
randomised costs and plot space, so the labels include both "share" and "go it
alone" answers. The test set is larger (100-150 buildings) and never seen, to
measure transfer. Results go to outputs/gnn_benchmark.csv.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines, gnn, instance, optimize
from .ml.surrogate import accelerated_design
from .config import Config
from .data import rainfall, synthetic

ROOT = Path(__file__).resolve().parent.parent


def random_cfg(rng: np.random.Generator) -> Config:
    cfg = Config()
    cfg.costs.water_value_per_m3 = float(rng.uniform(150, 600))
    cfg.costs.trench_per_m = float(rng.uniform(100, 450))
    cfg.costs.private_m3_per_roof_m2 = float(rng.choice([0.0, 0.02, 0.04, 0.06, 0.08]))
    cfg.opt.periods = 73            # 5-day steps keep label generation fast
    cfg.opt.n_scenarios = 8
    cfg.opt.time_limit_s = 60
    return cfg


def make(seed: int, n_range: tuple, imd: str | None):
    rng = np.random.default_rng(seed)
    cfg = random_cfg(rng)
    series, _ = rainfall.load_rainfall(cfg.train_years, imd, seed=0)
    rain = rainfall.by_year(series)
    b, G = synthetic.neighbourhood(int(rng.integers(*n_range)), seed=seed)
    inst = instance.build(f"syn{seed}", b, G, rain, {}, cfg)
    sc = optimize.scenario_set(inst, rain, cfg.opt.periods, cfg.uncertainty.n_area_draws,
                               cfg.opt.n_scenarios, seed)
    return inst, sc


def label(args):
    torch.set_num_threads(1)          # one thread per worker; otherwise workers oversubscribe the CPU
    seed, n_range, imd = args
    inst, sc = make(seed, n_range, imd)
    d = optimize.solve(inst, sc)
    return seed, gnn.graph_tensors(inst, sc), gnn.targets(inst, d), d.objective, d.runtime_s, \
        int(d.hub_open.sum())


def generate(seeds, n_range, imd, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(label, [(s, n_range, imd) for s in seeds]))


def benchmark(model, seeds, n_range, imd) -> pd.DataFrame:
    rows = []
    for s in seeds:
        inst, sc = make(s, n_range, imd)
        exact = optimize.solve(inst, sc)
        learned = gnn.gnn_design(model, inst, sc)
        sur = accelerated_design(inst, sc)
        prox = baselines.proximity(inst, sc)
        alone = baselines.individual(inst, sc)
        base = alone.objective
        rows.append(dict(seed=s, n=inst.n, water_value=inst.cfg.costs.water_value_per_m3,
                         milp_obj=exact.objective, gnn_obj=learned.objective,
                         proximity_obj=prox.objective, individual_obj=base,
                         surrogate_obj=sur.objective, surrogate_s=sur.runtime_s,
                         milp_s=exact.runtime_s, gnn_s=learned.runtime_s,
                         milp_hubs=int(exact.hub_open.sum()), gnn_hubs=int(learned.hub_open.sum()),
                         # Share of the solver's improvement over "everyone alone" the GNN recovers.
                         gnn_gain_recovered=(learned.objective - base) / (exact.objective - base)
                         if exact.objective - base > 1 else np.nan,
                         gnn_gap=(exact.objective - learned.objective) / abs(exact.objective),
                         surrogate_gap=(exact.objective - sur.objective) / abs(exact.objective)))
        print(rows[-1])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--imd", default=str(ROOT / "data" / "raw" / "imd"))
    ap.add_argument("--benchmark-only", action="store_true", help="reuse outputs/gnn.pt")
    a = ap.parse_args()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    if a.benchmark_only:
        model = gnn.RainGNN()
        model.load_state_dict(torch.load(out / "gnn.pt"))
        return report(model, a, out)
    cache = ROOT / "data" / "gnn_dataset.pt"
    cache.parent.mkdir(exist_ok=True)

    if cache.exists():
        data = torch.load(cache, weights_only=False)
    else:
        print("labelling training instances with the MILP ...")
        data = generate(range(1000, 1000 + a.train + a.val), (20, 61), a.imd, a.workers)
        torch.save(data, cache)
    share = np.mean([r[5] > 0 for r in data])
    print(f"{len(data)} labelled instances, {share:.0%} with at least one shared hub, "
          f"mean solve {np.mean([r[4] for r in data]):.1f}s")
    pairs = [(r[1], r[2]) for r in data]
    model = gnn.train(pairs[:a.train], pairs[a.train:], epochs=a.epochs)
    torch.save(model.state_dict(), out / "gnn.pt")
    report(model, a, out)


def report(model, a, out):
    print("benchmark on larger unseen instances ...")
    df = benchmark(model, range(5000, 5000 + a.test), (100, 151), a.imd)
    df.to_csv(out / "gnn_benchmark.csv", index=False)
    summary = dict(n_test=len(df), mean_gap=float(df.gnn_gap.mean()),
                   median_gap=float(df.gnn_gap.median()),
                   gain_recovered=float(df.gnn_gain_recovered.mean()),
                   speedup=float((df.milp_s / df.gnn_s).median()),
                   gnn_beats_proximity=float((df.gnn_obj >= df.proximity_obj).mean()),
                   surrogate_mean_gap=float(df.surrogate_gap.mean()),
                   surrogate_median_gap=float(df.surrogate_gap.median()),
                   surrogate_speedup=float((df.milp_s / df.surrogate_s).median()))
    (out / "gnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
