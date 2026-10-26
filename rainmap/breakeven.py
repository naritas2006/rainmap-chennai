"""Where does sharing start to pay? Sweep the value of water per neighbourhood.

    python -m rainmap.breakeven                           # all neighbourhoods
    python -m rainmap.breakeven --only mylapore --values 80 120 200

At each water value the private-only design and the surrogate design are both
re-solved and evaluated on held-out years, so the sharing gain is measured the
same way as in the main results. Writes outputs/breakeven.csv; RESULTS.md picks
it up on the next report.
"""
from __future__ import annotations

import argparse
import copy

import pandas as pd

from . import baselines, evaluate, report
from .config import NEIGHBOURHOODS, Config
from .ml.surrogate import accelerated_design
from .pipeline import OUT, ROOT, OB, prepare, rainfall_scenarios

VALUES = [40, 60, 80, 100, 120, 150, 200, 250, 300, 400]


def sweep(nb, cfg: Config, args, scenario_rain) -> list[dict]:
    print(f"\n== {nb.name} ==")
    inst, sc, ev, surrogate = prepare(nb, cfg, args, scenario_rain)
    rows = []
    for v in args.values:
        c2 = copy.deepcopy(cfg)
        c2.costs.water_value_per_m3 = v
        inst2 = copy.copy(inst)
        inst2.cfg = c2
        d_ind = baselines.individual(inst2, sc)
        d_sur = accelerated_design(inst2, sc, surrogate)
        # Private tanks are a feasible answer to the sharing problem, so never return worse.
        fallback = d_ind.objective >= d_sur.objective
        ind = evaluate.summarise(inst2, d_ind, evaluate.unit_table(inst2, d_ind, ev))
        opt = ind if fallback else evaluate.summarise(inst2, d_sur, evaluate.unit_table(inst2, d_sur, ev))
        rows.append(dict(neighbourhood=nb.name, water_value=v, fallback=fallback,
                         surrogate_buildings_shared=int((d_sur.assign >= 0).sum()),
                         objective_individual=d_ind.objective, objective_optimised=max(d_ind.objective, d_sur.objective),
                         failure_prob_individual=ind["failure_prob"], failure_prob_optimised=opt["failure_prob"],
                         p10_individual=ind["delivered_p10"], p10_optimised=opt["delivered_p10"],
                         buildings_shared=opt["buildings_shared"], shared_hubs=opt["n_shared"],
                         private_tanks=opt["n_private"], trench_m=opt["trench_m"],
                         net_individual=ind["net_value"], net_optimised=opt["net_value"],
                         sharing_gain=opt["net_value"] - ind["net_value"],
                         delivered_individual=ind["delivered_m3_yr"],
                         delivered_optimised=opt["delivered_m3_yr"]))
        r = rows[-1]
        print(f"  Rs {v:4.0f}/m3  shared {r['buildings_shared']:3d}  hubs {r['shared_hubs']:2d}  "
              f"net private Rs {r['net_individual']:>11,.0f}  optimised Rs {r['net_optimised']:>11,.0f}  "
              f"gain Rs {r['sharing_gain']:>9,.0f}" + ("  (surrogate worse; private kept)" if fallback else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--values", type=float, nargs="*", default=VALUES)
    ap.add_argument("--max-buildings", type=int, default=250)
    ap.add_argument("--footprints", default=str(OB) if OB.exists() else None)
    ap.add_argument("--imd", default=str(ROOT / "data" / "raw" / "imd"))
    ap.add_argument("--dem")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--eval-draws", type=int, default=20)
    args = ap.parse_args()
    cfg = Config()

    scenario_rain = rainfall_scenarios(cfg, args) if cfg.scenario_source == "generator" else None
    path = OUT / "breakeven.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["neighbourhood"])
    rows = []
    for nb in NEIGHBOURHOODS:
        if args.only and nb.name not in args.only:
            continue
        rows += sweep(nb, cfg, args, scenario_rain)
    new = pd.DataFrame(rows)
    out = pd.concat([old[~old.neighbourhood.isin(new.neighbourhood)], new], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    print(f"wrote {report.write()}")


if __name__ == "__main__":
    main()
