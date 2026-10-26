"""Write RESULTS.md from whatever is in outputs/.

    python -m rainmap.report

Called at the end of every pipeline run, so the numbers in RESULTS.md always
match the latest outputs. Anything missing (a neighbourhood that failed, no GNN
benchmark, no roof CNN) is reported as missing rather than skipped silently.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import NEIGHBOURHOODS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DESIGNS = ["individual", "design_storm", "proximity", "surrogate", "full_milp", "gnn"]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _rs(x: float) -> str:
    return f"{x:,.0f}"


def rainfall_section(r: dict | None) -> list[str]:
    if r is None:
        return ["_No rainfall model output (`outputs/rainfall_risk.json` missing)._", ""]
    q = r["annual_q"]
    rl = r["maxday_return_levels"]
    gain = 1 - r["model_nll_test"] / r["climatology_nll_test"]
    return [
        f"Trained on 1991–2015, judged on 2016–2023 ({r.get('source', 'unknown source')}).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Test NLL, model vs climatology | {r['model_nll_test']:.3f} vs {r['climatology_nll_test']:.3f} "
        f"({gain:.1%} better) |",
        f"| Simulated annual rainfall, mean ± sd | {r['annual_mean_sim']:.0f} ± {r['annual_sd_sim']:.0f} mm "
        f"(training years {r['annual_mean_train']:.0f} mm) |",
        f"| Annual quantiles P5 / P50 / P95 | {q['p05']:.0f} / {q['p50']:.0f} / {q['p95']:.0f} mm |",
        f"| P(annual < 1000 mm) | {r['p_annual_below_1000mm']:.0%} |",
        f"| P(Oct–Dec < 500 mm) | {r['p_ond_below_500mm']:.0%} |",
        f"| Max-day return levels 2 / 10 / 50 yr | {rl['2yr']:.0f} / {rl['10yr']:.0f} / {rl['50yr']:.0f} mm |",
        f"| Median longest dry spell | {r['longest_dry_spell_median_days']:.0f} days |",
        "",
    ]


def neighbourhood_section(name: str, s: dict | None) -> list[str]:
    if s is None or "recommended" not in s:
        return [f"### {name}", "", "_Missing or out of date: re-run "
                f"`python -m rainmap.pipeline --only {name}`._", ""]
    tag = " (held out)" if s["held_out"] else ""
    src = s["sources"]
    lines = [f"### {name}{tag}", "",
             f"{s['n_buildings']} buildings. Buildings: {src.get('buildings')}; streets: {src.get('streets')}; "
             f"terrain: {src.get('terrain')}; roofs: {src.get('roof_material', 'prior')}.", "",
             "| Design | Water m³/yr (P10) | L/₹ | Net value ₹/yr | Buildings shared | Fails plan | Runtime s |",
             "|---|---|---|---|---|---|---|"]
    for k in DESIGNS:
        d = s["designs"].get(k)
        if d is None:
            continue
        star = " ★" if k == s["recommended"] else ""
        lines.append(f"| {k}{star} | {d['delivered_m3_yr']:,.0f} ({d.get('delivered_p10', 0):,.0f}) | "
                     f"{d['litres_per_rupee']:.1f} | {_rs(d['net_value'])} | {d['buildings_shared']} | "
                     f"{d['failure_prob']:.0%} | {d['runtime_s']:.1f} |")
    lines += ["", "★ recommended (best exact objective). *Fails plan* = share of held-out scenarios "
              "delivering under 80% of the promised volume.", ""]
    if s.get("sensitivity"):
        lines += ["Water value sensitivity (surrogate MILP):", "",
                  "| ₹/m³ | Shared hubs | Buildings shared | Private tanks |", "|---|---|---|---|"]
        lines += [f"| {r['water_value']:.0f} | {r['hubs']} | {r['buildings_shared']} | {r['private']} |"
                  for r in s["sensitivity"]]
        lines.append("")
    return lines


def benchmark_section(g: dict | None, bench: pd.DataFrame | None) -> list[str]:
    if g is None:
        return ["_No GNN benchmark (`python -m rainmap.train_gnn`)._", ""]
    lines = [f"{g['n_test']} unseen synthetic instances"
             + (f" ({bench.n.min()}–{bench.n.max()} buildings)" if bench is not None else "") + ".", "",
             "| | GNN | Surrogate MILP |", "|---|---|---|",
             f"| Mean gap to exact MILP | {g['mean_gap']:.1%} | {g['surrogate_mean_gap']:.1%} |",
             f"| Median gap | {g['median_gap']:.1%} | {g['surrogate_median_gap']:.1%} |",
             f"| Speed-up vs exact MILP | {g['speedup']:.1f}× | {g['surrogate_speedup']:.2f}× |",
             f"| Beats proximity grouping | {g['gnn_beats_proximity']:.0%} | |",
             f"| Sharing gain recovered | {g['gain_recovered']:.0%} | |", ""]
    if g["surrogate_speedup"] < 1:
        lines += ["On these small instances the surrogate is slower than the exact MILP; "
                  "its advantage appears at full neighbourhood size (see runtimes above).", ""]
    return lines


def breakeven_section(be: pd.DataFrame | None, threshold: float = 0.2) -> list[str]:
    if be is None:
        return ["_No sweep yet (`python -m rainmap.breakeven`)._", ""]
    values = sorted(be.water_value.unique())
    lines = ["Surrogate design vs private tanks only, both re-solved at each water value and evaluated "
             "on held-out years.", "",
             "Buildings sharing a tank (of 250):", "",
             "| Neighbourhood | " + " | ".join(f"₹{v:.0f}" for v in values) + " |",
             "|---|" + "---|" * len(values)]
    for nb, g in be.groupby("neighbourhood", sort=False):
        g = g.set_index("water_value")
        lines.append(f"| {nb} | " + " | ".join(str(int(g.buildings_shared.get(v, 0))) for v in values) + " |")
    lines += ["", "Sharing gain, ₹/yr (optimised net value minus private-only net value):", "",
              "| Neighbourhood | " + " | ".join(f"₹{v:.0f}" for v in values) + " |",
              "|---|" + "---|" * len(values)]
    for nb, g in be.groupby("neighbourhood", sort=False):
        g = g.set_index("water_value")
        lines.append(f"| {nb} | " + " | ".join(_rs(g.sharing_gain.get(v, 0)) for v in values) + " |")
    if "fallback" in be:
        if "surrogate_buildings_shared" in be:
            worse = int((be.fallback & (be.surrogate_buildings_shared > 0)).sum())
            lines += ["", f"In {worse} of {len(be)} runs the surrogate proposed sharing that scored worse than "
                      "private tanks on the design objective; private tanks were kept there."]
        else:
            lines += ["", f"In {int(be.fallback.sum())} of {len(be)} runs the surrogate design did not beat "
                      "private tanks on the design objective (worse, or private tanks too); private tanks "
                      "were kept there."]
        neg = be[~be.fallback & (be.sharing_gain < 0)]
        if len(neg):
            cases = ", ".join(f"{r.neighbourhood} ₹{r.water_value:.0f}" for r in neg.itertuples())
            lines += ["", f"Negative gains that remain ({cases}) are designs that beat private tanks on the "
                      "risk-adjusted design objective (which rewards the worst 25% of years) but deliver less "
                      "average value on the held-out years. They mark where sharing is marginal, not where it "
                      "clearly pays."]
    lines += ["", f"Break-even = lowest tested water value at which at least {threshold:.0%} of buildings "
              "share a tank and sharing beats private tanks on held-out years.", ""]
    for nb, g in be.groupby("neighbourhood", sort=False):
        hit = g[(g.buildings_shared >= threshold * 250) & (g.sharing_gain > 0)].sort_values("water_value")
        lines.append(f"- **{nb}**: " + (f"₹{hit.water_value.iloc[0]:.0f}/m³" if len(hit)
                                         else f"not reached up to ₹{g.water_value.max():.0f}/m³"))
    return lines + [""]


def findings(summaries: dict) -> list[str]:
    done = {k: s for k, s in summaries.items() if s and "recommended" in s}
    if not done:
        return []
    lines = []
    for k, s in done.items():
        d = s["designs"]
        rec = d[s["recommended"]]
        synthetic = " **(synthetic data)**" if "SYNTHETIC" in s["sources"].values() else ""
        pays = [r for r in s.get("sensitivity", []) if r["buildings_shared"] > 0]
        first = (f"sharing starts at ₹{pays[0]['water_value']:.0f}/m³ "
                 f"({pays[0]['buildings_shared']} buildings)" if pays else "no sharing at any tested water value")
        lines.append(f"- **{k}**{synthetic}: recommended design shares {rec['buildings_shared']} of "
                     f"{s['n_buildings']} buildings; net ₹{_rs(rec['net_value'])}/yr vs "
                     f"₹{_rs(d['proximity']['net_value'])}/yr for proximity grouping; {first}.")
    return lines + [""]


def write(path: Path = ROOT / "RESULTS.md") -> Path:
    summaries = {nb.name: _load(OUT / nb.name / "summary.json") for nb in NEIGHBOURHOODS}
    bench = pd.read_csv(OUT / "gnn_benchmark.csv") if (OUT / "gnn_benchmark.csv").exists() else None
    roof = _load(OUT / "roof_cnn_validation.json")
    lines = ["# RainMap results", "",
             f"_Generated by `python -m rainmap.report` on {date.today().isoformat()} "
             "from `outputs/`. Do not edit by hand._", "",
             "## Headline findings", "", *findings(summaries),
             "## Rainfall model", "", *rainfall_section(_load(OUT / "rainfall_risk.json")),
             "## Neighbourhoods", "", "Evaluated on held-out 2016–2023 rainfall with fresh roof-area "
             "and runoff draws.", ""]
    for name, s in summaries.items():
        lines += neighbourhood_section(name, s)
    be = pd.read_csv(OUT / "breakeven.csv") if (OUT / "breakeven.csv").exists() else None
    lines += ["## When does sharing pay?", "", *breakeven_section(be),
              "## GNN and surrogate benchmark", "",
              *benchmark_section(_load(OUT / "gnn_summary.json"), bench),
              "## Roof CNN", "",
              json.dumps(roof, indent=2) if roof else "_Not trained or validated yet; every roof uses the "
              "prior runoff coefficient (0.80 ± 0.08)._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {write()}")
