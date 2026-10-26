import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import { HBar, LineChart, MonthlyBand } from "./Charts.jsx";
import MapView from "./MapView.jsx";

const METHOD = {
  individual: "Private tanks (optimal size)",
  design_storm: "Private tanks (design-storm rule)",
  proximity: "Proximity grouping",
  surrogate: "Surrogate MILP + exact sizing",
  full_milp: "Full MILP (HiGHS)",
  gnn: "GNN proposal + exact sizing",
};
const fmt = {
  int: (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-IN")),
  one: (v) => (v == null ? "–" : Number(v).toFixed(1)),
  pct: (v) => (v == null ? "–" : `${Math.round(v * 100)}%`),
  signed: (v) => (v == null ? "–" : `${v >= 0 ? "+" : ""}${Math.round(v * 100)}%`),
  rs: (v) => (v == null ? "–" : `${v < 0 ? "−" : ""}₹${Math.abs(Math.round(v)).toLocaleString("en-IN")}`),
};
const title = (n) => n.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

function Tile({ label, value, sub }) {
  return (
    <div className="card tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

function Sidebar({ index, name, setName, summary }) {
  const c = summary?.config;
  return (
    <aside>
      <h1>RainMap Chennai</h1>
      <div className="muted small">Shared rooftop harvesting screening</div>
      <h3>Neighbourhood</h3>
      <select value={name || ""} onChange={(e) => setName(e.target.value)}>
        {index.map((e) => (
          <option key={e.name} value={e.name}>{title(e.name)}{e.held_out ? " (held out)" : ""}</option>
        ))}
      </select>
      {summary && (
        <>
          <h3>Data sources</h3>
          <ul className="plain small">
            {Object.entries(summary.sources).map(([k, v]) => (
              <li key={k}><b>{k.replace(/_/g, " ")}</b>: {String(v).includes("SYNTHETIC") || String(v).includes("UNVALIDATED") ? "⚠ " : ""}{v}</li>
            ))}
          </ul>
          <h3>Key assumptions</h3>
          <ul className="plain small muted">
            <li>Water value ₹{c.costs.water_value_per_m3}/m³</li>
            <li>Trench ₹{c.costs.trench_per_m}/m + pipe ₹{c.costs.pipe_per_m}/m × {c.costs.lines_per_link}</li>
            <li>Private sump space {c.costs.private_m3_per_roof_m2} m³ per m² footprint</li>
            <li>Demand {c.hydrology.demand_lpcd} L/person/day (non-potable)</li>
            <li>Risk weight λ={c.opt.risk_lambda}, CVaR worst {fmt.pct(c.opt.cvar_alpha)}</li>
            <li>Design scenarios: {c.scenario_source} rainfall; judged on {c.test_years[0]}–{c.test_years.at(-1)}</li>
          </ul>
        </>
      )}
    </aside>
  );
}

function DesignTable({ summary }) {
  const rows = Object.entries(summary.designs).filter(([k]) => k !== "optimised");
  return (
    <div className="card table-wrap">
      <table>
        <thead>
          <tr>
            <th>Design</th><th>Water m³/yr</th><th>Annual cost</th><th>L per ₹</th><th>Net value/yr</th>
            <th>Days met</th><th>P(fail)</th><th>Shared bldgs</th><th>Trench m</th><th>Realised vs plan</th><th>Solve s</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, d]) => (
            <tr key={k} className={k === summary.recommended ? "rec" : ""}>
              <td>{METHOD[k] || k}{k === summary.recommended ? " ★" : ""}</td>
              <td>{fmt.int(d.delivered_m3_yr)}</td><td>{fmt.rs(d.annual_cost)}</td>
              <td>{fmt.one(d.litres_per_rupee)}</td><td>{fmt.rs(d.net_value)}</td>
              <td>{fmt.pct(d.reliability_days)}</td><td>{fmt.pct(d.failure_prob)}</td>
              <td>{d.buildings_shared}</td><td>{fmt.int(d.trench_m)}</td>
              <td>{fmt.signed(d.plan_gap)}</td><td>{fmt.one(d.runtime_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="small muted" style={{ marginTop: 8 }}>
        ★ recommended design (best exact objective). All designs judged by daily simulation on held-out
        rainfall years with fresh roof-area and roof-material draws.
      </div>
    </div>
  );
}

function Shortlist({ clusters }) {
  if (!clusters.length) return <div className="card muted">No units recommended.</div>;
  const losers = clusters.filter((c) => c.sharing_pays === false).length;
  return (
    <div className="card table-wrap">
      <table>
        <thead>
          <tr><th>Rank</th><th>Unit</th><th>Type</th><th>Buildings</th><th>Tank m³</th><th>Water m³/yr</th>
            <th>P10 m³/yr</th><th>L per ₹</th><th>Days met</th><th>P(fail)</th><th>Trench m</th><th>Sharing gain ₹/yr</th></tr>
        </thead>
        <tbody>
          {clusters.slice(0, 25).map((c) => (
            <tr key={c.unit}>
              <td>{c.survey_rank}</td><td>{c.unit}</td><td>{c.kind}</td><td>{c.n_buildings}</td>
              <td>{fmt.one(c.storage_m3)}</td><td>{fmt.int(c.delivered_m3_yr)}</td><td>{fmt.int(c.delivered_p10)}</td>
              <td>{fmt.one(c.litres_per_rupee)}</td><td>{fmt.pct(c.reliability_days)}</td><td>{fmt.pct(c.failure_prob)}</td>
              <td>{fmt.int(c.trench_m)}</td>
              <td>{c.kind === "shared" ? `${fmt.rs(c.sharing_gain_value)}${c.sharing_pays === false ? " ✕" : ""}` : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {losers > 0 && (
        <div className="warn small">✕ {losers} shared cluster(s) did worse on held-out years than their members with private tanks.</div>
      )}
    </div>
  );
}

function RainfallPanel({ r }) {
  if (!r) return null;
  return (
    <>
      <h2>Seasonal rainfall model</h2>
      <div className="tiles">
        <Tile label="Test log-likelihood gain vs climatology"
              value={fmt.pct((r.climatology_nll_test - r.model_nll_test) / r.climatology_nll_test)}
              sub={`NLL/day ${r.model_nll_test.toFixed(3)} vs ${r.climatology_nll_test.toFixed(3)}`} />
        <Tile label="P(annual rain < 1000 mm)" value={fmt.pct(r.p_annual_below_1000mm)}
              sub={`median year ${fmt.int(r.annual_q.p50)} mm`} />
        <Tile label="1-day max, 10-year return" value={`${fmt.int(r.maxday_return_levels["10yr"])} mm`}
              sub={`50-year: ${fmt.int(r.maxday_return_levels["50yr"])} mm`} />
        <Tile label="Held-out years inside 90% band" value={fmt.pct(r.test_annual_inside_90pct)}
              sub={`source: ${r.source}`} />
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <MonthlyBand band={r.monthly_band} />
      </div>
    </>
  );
}

function MLPanel({ ml }) {
  if (!ml) return null;
  const g = ml.gnn;
  const cnn = ml.roof_cnn;
  return (
    <>
      <h2>Learned solvers vs exact MILP</h2>
      {g ? (
        <div className="tiles">
          <Tile label="Surrogate: median objective gap" value={fmt.pct(g.surrogate_median_gap)}
                sub={`median ${fmt.one(g.surrogate_speedup)}× faster than full MILP`} />
          <Tile label="GNN: median objective gap" value={fmt.pct(g.median_gap)}
                sub={`median ${fmt.one(g.speedup)}× faster; recovers ${fmt.pct(g.gain_recovered)} of MILP gain`} />
          <Tile label="GNN ≥ proximity grouping" value={fmt.pct(g.gnn_beats_proximity)} sub={`${g.n_test} unseen larger instances`} />
        </div>
      ) : <div className="card muted">Run <code>python -m rainmap.train_gnn</code> to benchmark the learned solvers.</div>}
      <h2>Roof-material CNN</h2>
      <div className="card small">
        {!cnn.trained && <>Not trained yet. Download RoofNet (Kaggle, CC BY-NC 4.0) and run <code>python -m rainmap.ml.roof_cnn train</code>. Runoff uses the prior coefficient with ±0.08 uncertainty until then.</>}
        {cnn.trained && !cnn.validation && <>Trained, <b>not yet validated on Chennai</b>: predictions are blended 50/50 with the prior. Label chips with <code>roof_cnn label-sheet</code> and run <code>roof_cnn validate</code>.</>}
        {cnn.validation && <>Chennai validation on {cnn.validation.n} hand-labelled roofs: accuracy {fmt.pct(cnn.validation.accuracy)}, runoff MAE {cnn.validation.runoff_mae.toFixed(3)} (prior {cnn.validation.prior_runoff_mae.toFixed(3)}).</>}
      </div>
    </>
  );
}

export default function App() {
  const [index, setIndex] = useState([]);
  const [name, setName] = useState(null);
  const [summary, setSummary] = useState(null);
  const [geo, setGeo] = useState({});
  const [clusters, setClusters] = useState([]);
  const [rain, setRain] = useState(null);
  const [ml, setMl] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.neighbourhoods().then((ix) => { setIndex(ix); setName(ix[0]?.name); }).catch((e) => setError(e.message));
    api.rainfall().then(setRain).catch(() => {});
    api.ml().then(setMl).catch(() => {});
  }, []);
  useEffect(() => {
    if (!name) return;
    setSummary(null); setGeo({});
    Promise.all([api.summary(name), api.buildings(name), api.routes(name), api.clusters(name)])
      .then(([s, b, r, c]) => { setSummary(s); setGeo({ b, r }); setClusters(c); })
      .catch((e) => setError(e.message));
  }, [name]);

  if (error) return <main><div className="warn">Could not load results: {error}. Run the pipeline, then start the API.</div></main>;
  const rec = summary?.designs?.optimised;
  const synthetic = summary && Object.values(summary.sources).some((v) => String(v).includes("SYNTHETIC"));
  const bars = summary ? Object.entries(summary.designs).filter(([k]) => k !== "optimised")
    .map(([k, d]) => ({ name: METHOD[k] || k, lpr: d.litres_per_rupee, highlight: k === summary.recommended })) : [];
  const sens = summary?.sensitivity?.map((s) => ({
    x: s.water_value, y: s.buildings_shared,
    tip: `₹${s.water_value}/m³: ${s.buildings_shared} shared, ${s.private} private, ${s.hubs} hubs`,
  })) || [];

  return (
    <div className="layout">
      <Sidebar index={index} name={name} setName={setName} summary={summary} />
      <main>
        <h1>{name ? title(name) : "…"}</h1>
        <div className="muted">Screening tool, not an engineering design. Ranks units worth a field survey, with reliability measured on held-out rainfall years.</div>
        {synthetic && <div className="warn">Some inputs here are synthetic stand-ins; numbers illustrate the method, not this place.</div>}
        {rec && (
          <>
            <h2>Recommended design · {METHOD[summary.recommended]}</h2>
            <div className="tiles">
              <Tile label="Water delivered" value={`${fmt.int(rec.delivered_m3_yr)} m³/yr`} sub={`P10 ${fmt.int(rec.delivered_p10)} m³/yr`} />
              <Tile label="Litres per rupee" value={fmt.one(rec.litres_per_rupee)} sub={`net ${fmt.rs(rec.net_value)}/yr`} />
              <Tile label="Buildings in shared clusters" value={`${rec.buildings_shared} / ${summary.n_buildings}`} sub={`${rec.n_shared} hubs, ${rec.n_private} private tanks`} />
              <Tile label="P(year < 80% of plan)" value={fmt.pct(rec.failure_prob)} sub={`days demand met ${fmt.pct(rec.reliability_days)}`} />
            </div>
            <h2>Design comparison on held-out years</h2>
            <DesignTable summary={summary} />
            <div className="card" style={{ marginTop: 12 }}>
              <div className="small muted">Litres delivered per rupee of annualised cost</div>
              <HBar rows={bars} valueKey="lpr" labelKey="name" format={fmt.one} title="Litres per rupee by design" />
            </div>
          </>
        )}
        <h2>Map</h2>
        <MapView buildings={geo.b} routes={geo.r} />
        <h2>Survey shortlist</h2>
        <Shortlist clusters={clusters} />
        {sens.length > 0 && (
          <>
            <h2>Where does sharing start to pay?</h2>
            <div className="card">
              <LineChart points={sens} xLabel="Value of water, ₹ per m³" yLabel="Buildings in shared clusters" xFormat={(v) => `₹${v}`} />
              <div className="small muted">Surrogate MILP re-solved at each water value, then sized exactly. A flat line at zero means private tanks win at every value tested.</div>
            </div>
          </>
        )}
        <RainfallPanel r={rain} />
        <MLPanel ml={ml} />
        <details>
          <summary>Honest limits</summary>
          <ul className="small muted">
            <li>Access across private plots and ownership are not modelled; pipes follow public streets.</li>
            <li>Underground tanks face Chennai's shallow groundwater table; not assessed.</li>
            <li>Rainfall is city-level (IMD 0.25° grid); no within-city variation.</li>
            <li>Pipe cost uses one link per building (star), an upper bound; shared trench length is reported separately.</li>
            <li>No building heights, so demand assumes a fixed number of floors.</li>
            <li>Costs are screening assumptions; edit <code>rainmap/config.py</code>.</li>
          </ul>
        </details>
      </main>
    </div>
  );
}
