import numpy as np
import pytest

from rainmap import baselines, evaluate, instance, optimize
from rainmap.config import Config
from rainmap.data import rainfall, synthetic
from rainmap.hydrology import simulate


@pytest.fixture(scope="module")
def small():
    cfg = Config()
    cfg.opt.periods = 73
    cfg.opt.n_scenarios = 4
    cfg.costs.water_value_per_m3 = 400
    cfg.costs.trench_per_m = 150
    cfg.costs.private_m3_per_roof_m2 = 0.02
    years = list(cfg.train_years[:6]) + list(cfg.test_years[:3])
    rain = rainfall.by_year(rainfall.synthetic_series(years, seed=3))
    train = {y: rain[y] for y in years[:6]}
    test = {y: rain[y] for y in years[6:]}
    b, G = synthetic.neighbourhood(30, seed=3)
    inst = instance.build("t", b, G, train, test, cfg)
    sc = optimize.scenario_set(inst, train, 73, 2, 4, 0)
    ev = optimize.scenario_set(inst, test, 365, 3, None, 1)
    return inst, sc, ev


def test_simulate_balance():
    inflow = np.array([5.0, 0, 0, 3, 0])
    delivered, met = simulate(inflow, np.full(5, 2.0), 4.0, warmup=False)
    # Day 1: 5 in, 2 out, keep 3 (cap 4). Day 2: 2 out. Day 3: 1 out. Day 4: 2 out, keep 1. Day 5: 1.
    assert delivered.tolist() == [2, 2, 1, 2, 1]
    assert met == pytest.approx(3 / 5)


def test_imd_reader_roundtrip(tmp_path):
    grid = np.full((365, 129, 135), -999.0, np.float32)
    grid[:, 26, 55] = 4.0
    path = tmp_path / "Rainfall_ind2001_rfp25.grd"
    grid.tofile(path)
    s = rainfall.load_imd_series(tmp_path, [2001], latlon=(13.0, 80.25))
    assert len(s) == 365 and s.iloc[0] == pytest.approx(4.0)


def test_individual_milp_matches_precompute(small):
    inst, sc, _ = small
    d = baselines.individual(inst, sc)
    assert (d.assign < 0).all()
    sizes = [s for s in inst.cfg.costs.individual_sizes_m3 if s > 0]
    y = optimize.individual_yield(sc, sizes)
    for i in np.nonzero(d.ind_size > 0)[0]:
        m = sizes.index(d.ind_size[i])
        assert d.promise_ind[i] == pytest.approx((y[i, m] * sc.p).sum())


def test_milp_dominates_baselines(small):
    inst, sc, _ = small
    best = optimize.solve(inst, sc)
    for d in (baselines.individual(inst, sc), baselines.proximity(inst, sc)):
        assert best.objective >= d.objective - 1e-6 * abs(d.objective)


def test_space_limit_respected(small):
    inst, sc, _ = small
    d = optimize.solve(inst, sc)
    space = inst.cfg.costs.private_m3_per_roof_m2 * inst.buildings.area_m2.values
    assert (d.ind_size <= space + 1e-9).all()


def test_evaluation_tables(small):
    inst, sc, ev = small
    d = optimize.solve(inst, sc)
    u = evaluate.unit_table(inst, d, ev)
    s = evaluate.summarise(inst, d, u)
    assert 0 <= s["failure_prob"] <= 1 and s["delivered_m3_yr"] >= 0
    if not u.empty:
        assert (u.delivered_m3_yr <= u.demand_m3_yr + 1e-6).all()


def test_gnn_forward_and_decode(small):
    from rainmap import gnn
    inst, sc, _ = small
    d = optimize.solve(inst, sc)
    gt, tg = gnn.graph_tensors(inst, sc), gnn.targets(inst, d)
    model = gnn.RainGNN(hid=16, layers=2)
    loss = gnn.loss_fn(model, gt, tg)
    loss.backward()
    assert np.isfinite(loss.item())
    design = gnn.gnn_design(model, inst, sc)        # untrained, but must be feasible
    assert design.objective <= d.objective + 1e-6 * abs(d.objective)


def test_rainfall_model_learns_season():
    from rainmap.ml.rainfall_model import RainfallModel, climatology_nll
    years = list(range(2000, 2012))
    rain = rainfall.by_year(rainfall.synthetic_series(years, seed=5))
    train = {y: rain[y] for y in years[:10]}
    test = {y: rain[y] for y in years[10:]}
    m = RainfallModel.fit(train, epochs=300, val_years=2, log=lambda *a: None)
    sim = m.sample(50, seed=1)
    obs_mean = np.mean([v.sum() for v in train.values()])
    assert abs(sim.sum(1).mean() - obs_mean) / obs_mean < 0.15
    assert sim[:, 273:].sum() > sim[:, :120].sum()            # NE monsoon wetter than Jan-Apr
    assert m.nll_per_day(test) < climatology_nll(train, test) + 0.1


def test_surrogate_planes_and_repair(small):
    from rainmap.ml.surrogate import YieldSurrogate, accelerated_design
    inst, sc, _ = small
    sur = YieldSurrogate.fit(sc.rain, steps=300)
    assert sur.fit_error < 0.05 and (sur.planes >= 0).all()
    # Homogeneous of degree one: doubling R, D, S doubles the predicted delivery.
    assert np.allclose(sur.predict(400.0, 2.0, 10.0) * 2, sur.predict(800.0, 4.0, 20.0))
    # Matches exact simulation on the fixture's multi-day (5-day) steps.
    from rainmap.hydrology import to_periods
    days = to_periods(np.ones(365), sc.rain.shape[1])
    for R, D, S in [(300.0, 2.0, 0.0), (1500.0, 5.0, 30.0)]:
        exact = simulate(sc.rain * R, D * days, S)[0].sum(-1)
        assert np.abs(sur.predict(R, D, S) - exact).mean() / exact.mean() < 0.1
    d = accelerated_design(inst, sc, sur)
    full = optimize.solve(inst, sc)
    assert d.objective <= full.objective + 1e-6 * abs(full.objective)   # exact repair is feasible
    assert d.objective >= baselines.individual(inst, sc).objective - 1e-6 * abs(full.objective)


def test_roof_runoff_stats():
    from rainmap.ml.roof_cnn import CLASSES, RUNOFF, runoff_stats
    p = np.zeros((2, len(CLASSES)))
    p[0, CLASSES.index("metal_sheet")] = 1
    p[1, CLASSES.index("thatch")] = p[1, CLASSES.index("metal_sheet")] = 0.5
    mean, sd = runoff_stats(p)
    assert mean[0] == pytest.approx(RUNOFF[CLASSES.index("metal_sheet")]) and sd[0] == pytest.approx(0)
    assert mean[1] == pytest.approx(0.7) and sd[1] == pytest.approx(0.2)


def test_rainfall_model_save_load(tmp_path):
    from rainmap.ml.rainfall_model import RainfallModel
    rain = rainfall.by_year(rainfall.synthetic_series(range(2000, 2006), seed=2))
    m = RainfallModel.fit(rain, epochs=50, val_years=1, log=lambda *a: None)
    m.save(tmp_path / "m.pt")
    m2 = RainfallModel.load(tmp_path / "m.pt")
    assert np.allclose(m.sample(5, seed=3), m2.sample(5, seed=3))


def test_report_flags_missing_outputs(tmp_path, monkeypatch):
    from rainmap import report
    monkeypatch.setattr(report, "OUT", tmp_path)
    text = report.write(tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "Missing or out of date" in text and "Not trained or validated yet" in text
