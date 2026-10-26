# RainMap Chennai

**Which neighbouring buildings should share an underground rainwater tank and pipe
network, and how reliable is that design when rainfall and our own map data are
uncertain?**

RainMap produces a ranked, map-based shortlist of shared or private harvesting
units worth a field survey. Each unit's reliability is measured on rainfall years
the models never saw. It is a screening tool, not an engineering design.

Four ML components feed a robust, scenario-based optimiser:

| # | Component | What it does | Code |
|---|---|---|---|
| 1 | **Seasonal rainfall model** | Neural weather generator trained on IMD daily grids. Produces rainfall-risk metrics and the design scenarios. | `rainmap/ml/rainfall_model.py` |
| 2 | **Roof-material CNN** | ResNet-18 fine-tuned on RoofNet. Gives each building a runoff coefficient with uncertainty. Validated on hand-labelled Chennai roofs. | `rainmap/ml/roof_cnn.py` |
| 3 | **Graph neural network** | PyTorch Geometric message passing that proposes clusters, imitating the exact solver. | `rainmap/gnn.py` |
| 4 | **Learned surrogate** | Learned min-of-planes yield model that shrinks the MILP by ~100×. Designs stay feasible because each clustering is re-sized exactly. | `rainmap/ml/surrogate.py` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                                   # 12 tests, ~1 min

# data (one-off)
python -m rainmap.data.fetch_open_buildings                # Chennai footprints, ~30 s
python -c "import imdlib; imdlib.get_data('rain', 1991, 2023, file_dir='data/raw/imd')"

# models and results
python -m rainmap.train_gnn                                # label, train GNN, benchmark GNN + surrogate
python -m rainmap.pipeline                                 # rainfall model + all designs -> outputs/, RESULTS.md
#   --skip-full-milp   surrogate only (much faster)
#   --offline          synthetic neighbourhoods, no downloads

# dashboard (FastAPI serves the built React app)
cd frontend && npm install && npm run build && cd ..
uvicorn api.main:app --port 8000                           # http://localhost:8000
#   dev mode: uvicorn api.main:app --reload  +  cd frontend && npm run dev  (http://localhost:5173)
```

### Roof CNN (needs a Kaggle account and an imagery source)

```bash
kaggle datasets download noellelaw/roofnet -p data/raw/roofnet --unzip
python -m rainmap.ml.roof_cnn train                         # Colab GPU recommended
python -m rainmap.ml.roof_cnn chips --tiles "<XYZ URL>"     # check the provider's terms first
python -m rainmap.ml.roof_cnn predict
python -m rainmap.ml.roof_cnn label-sheet                   # hand-label ~300 Chennai roofs
python -m rainmap.ml.roof_cnn validate
```

Until the CNN has been validated on Chennai labels, its predictions are blended
50/50 with the prior runoff coefficient and carry wider uncertainty. With no
CNN at all, every roof uses the prior (0.80 ± 0.08).

## Data

| Input | Source | Status in this repo |
|---|---|---|
| Building footprints + confidence | Google-Microsoft Open Buildings, VIDA GeoParquet on Source Cooperative, clipped with DuckDB | real |
| Streets | OpenStreetMap via OSMnx; rotates Overpass mirrors because the main server is often at capacity | real |
| Rainfall | IMD Pune 0.25° daily gridded rainfall, 1991–2023, via `imdlib` | real |
| Terrain | any DEM GeoTIFF (`--dem`) | not supplied; treated as flat |
| Roof material | RoofNet (Kaggle, CC BY-NC 4.0) + overhead imagery | pipeline built; needs Kaggle key + imagery |

Rainfall is city-level, because the 0.25° grid is coarser than a neighbourhood.
Models train on 1991–2015 (1991–2010 to fit, 2011–2015 for early stopping) and
are judged on 2016–2023.

## Method

1. **Instance** (`instance.py`): buildings snap to the street graph, and
   candidate hub tanks are spread by k-means. Each building–hub pair within
   250 m gets a street-routed pipe cost (one trench, two lines) and a gravity
   check; a lift pump is added if the check fails.
2. **Uncertainty** (`uncertainty.py`, `optimize.scenario_set`): a scenario
   combines one rainfall year, one roof-area draw and one runoff draw. Roof
   area is the footprint area × a lognormal error that widens as footprint
   confidence falls; low-confidence footprints may not exist. Runoff comes from
   the CNN or the prior.
3. **Rainfall model** (ML 1): an MLP maps day-of-year harmonics and 1/7/30-day
   rainfall history to the wet-day probability and a gamma amount
   distribution. It is fitted by maximum likelihood with early stopping, then
   calibrated so simulated wet-day frequency, annual mean and year-to-year
   spread match the training years. 200 generated years supply the design
   scenarios.
4. **Optimisation**: capacitated facility location with a CVaR term,
   maximising `value × [(1−λ)·E[water] + λ·CVaR(worst 25%)] − annualised cost`.
   - *Full MILP* (`optimize.py`, SciPy/HiGHS): daily water balance per hub
     and scenario. Exact, but hits the time limit at 250 buildings.
   - *Surrogate MILP* (ML 4, Pyomo + HiGHS): for one rainfall year, a tank's
     best annual delivery is the optimal value of an LP in pooled roof R,
     demand D and storage S. That value is concave and degree-1 homogeneous,
     so it equals `min_k a_k·R + b_k·D + c_k·S`. The planes are learned from
     exact simulations, and each becomes one linear constraint, replacing
     hub × scenario × 365 balance variables. The surrogate only chooses the
     clustering; tanks are then re-sized with the exact daily model, so every
     design is feasible and costed exactly.
   - *GNN* (ML 3, PyTorch Geometric): edge-MLP message passing over kNN and
     building–hub link edges. It predicts which hubs open and who joins them,
     then the exact model sizes the tanks. It is trained by imitating MILP
     solutions on small synthetic instances with randomised costs, and tested
     on larger unseen ones.
   - The recommended design is whichever optimising method reaches the best
     exact objective.
5. **Baselines**: optimally sized private tanks, a design-storm sizing rule
   and naive proximity grouping.
6. **Evaluation** (`evaluate.py`): daily continuous simulation on held-out
   2016–2023 rainfall, with fresh roof-area and runoff draws. Per unit it
   reports water delivered (mean and P10), days of demand met, P(year < 80%
   of plan), litres per rupee, net value, trench length and the **sharing
   gain** against the same buildings with private tanks.
7. **Sensitivity**: the surrogate MILP is re-solved at higher water values to
   show where, if anywhere, sharing starts to pay.

## Outputs

`outputs/<neighbourhood>/`:
- `summary.json`: per-design metrics, the recommended design, sources, sensitivity and config
- `clusters_ranked.csv`: the survey shortlist
- `buildings.geojson`, `routes.geojson`: map layers

`outputs/`:
- `rainfall_risk.json`, `rainfall_model.pt`
- `gnn.pt`, `gnn_benchmark.csv`, `gnn_summary.json`
- `roof_cnn.pt`, `roof_cnn_validation.json`

API: `GET /api/neighbourhoods`, `/api/neighbourhoods/{name}[/buildings|/routes|/clusters]`, `/api/rainfall`, `/api/ml`.

## Results

See [`RESULTS.md`](RESULTS.md), regenerated from `outputs/` at the end of every pipeline run (or `python -m rainmap.report`).

## Honest limits

- Access across private plots and ownership are not modelled; pipes follow public streets.
- Underground tanks face Chennai's shallow groundwater table; not assessed.
- Pipe cost uses one link per building (star layout), an upper bound; real
  shared trenches are shorter (reported as `trench_m`).
- No building heights, so demand assumes 2 floors.
- The roof CNN is unvalidated on Chennai until hand labels exist, and RoofNet is non-commercial (CC BY-NC 4.0).
- The GNN imitates solver labels on synthetic instances. It misses borderline
  sharing opportunities, so it proposes clusters rather than replacing the solver.
- Costs in `rainmap/config.py` are screening assumptions; replace them with local quotes.
