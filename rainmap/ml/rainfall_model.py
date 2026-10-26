"""Seasonal stochastic rainfall model (neural weather generator).

A small network maps the day of year (Fourier features) and recent rainfall
history to (a) the probability that today is wet and (b) a gamma distribution for
the amount on a wet day. It is trained by maximum likelihood on IMD training
years and sampled autoregressively to generate as many synthetic years as needed.

Daily generators reproduce the seasonal cycle but usually under-represent
year-to-year swings (the 2003 drought vs the 2005 floods). A year-level
lognormal multiplier on wet-day amounts is calibrated so the spread of
simulated annual totals matches the training years.

Validation uses held-out years: per-day log-likelihood against a monthly
climatology baseline, and whether the observed annual totals, monthly totals and
annual maxima fall inside the simulated distribution.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

WET_MM = 1.0           # a day with >= 1 mm counts as wet
N_HARMONICS = 3
HISTORY = (1, 7, 30)   # windows of past rainfall fed to the model


def _doy_features(doy: torch.Tensor) -> torch.Tensor:
    w = 2 * np.pi * doy[..., None] / 365.0
    k = torch.arange(1, N_HARMONICS + 1, dtype=torch.float32)
    return torch.cat([torch.sin(w * k), torch.cos(w * k)], -1)


def _history_features(hist: torch.Tensor) -> torch.Tensor:
    """hist: (..., 30) past rainfall, most recent last."""
    wet_yesterday = (hist[..., -1] >= WET_MM).float()[..., None]
    means = [torch.log1p(hist[..., -w:].mean(-1, keepdim=True)) for w in HISTORY]
    return torch.cat([wet_yesterday] + means, -1)


N_FEATURES = 2 * N_HARMONICS + 1 + len(HISTORY)


class RainNet(nn.Module):
    def __init__(self, hid: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEATURES, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 3))
        with torch.no_grad():                      # start near Chennai's climatology
            self.net[-1].bias.copy_(torch.tensor([-1.0, -0.5, np.log(14.0)]))

    def forward(self, doy, hist):
        out = self.net(torch.cat([_doy_features(doy), _history_features(hist)], -1))
        p_logit = out[..., 0]
        shape = nn.functional.softplus(out[..., 1]) + 0.05
        mean = torch.exp(out[..., 2].clamp(max=6.0)) + 0.1
        return p_logit, shape, mean


def _nll(p_logit, shape, mean, r):
    wet = r >= WET_MM
    l_occ = nn.functional.binary_cross_entropy_with_logits(p_logit, wet.float(), reduction="none")
    rate = shape / mean
    rw = torch.where(wet, r, torch.ones_like(r))
    l_amt = -(shape * torch.log(rate) + (shape - 1) * torch.log(rw) - rate * rw - torch.lgamma(shape))
    return l_occ + torch.where(wet, l_amt, torch.zeros_like(r))


def _windows(rain_by_year: dict[int, np.ndarray]):
    """Concatenate years and build (doy, 30-day history, today) samples."""
    years = sorted(rain_by_year)
    series = np.concatenate([rain_by_year[y] for y in years]).astype(np.float32)
    doy = np.concatenate([np.arange(365) for _ in years]).astype(np.float32)
    H = max(HISTORY)
    idx = np.arange(H, len(series))
    hist = np.stack([series[i - H:i] for i in idx])
    return torch.tensor(doy[idx]), torch.tensor(hist), torch.tensor(series[idx])


@dataclass
class RainfallModel:
    net: RainNet
    year_sigma: float = 0.0
    logit_offset: float = 0.0      # post-hoc calibration of simulated wet-day frequency
    amount_scale: float = 1.0      # post-hoc calibration of simulated annual mean

    # --- training -------------------------------------------------------------------
    @classmethod
    def fit(cls, train: dict[int, np.ndarray], epochs: int = 1500, val_years: int = 5,
            seed: int = 0, log=print) -> "RainfallModel":
        """MLE with early stopping on the last `val_years` training years, then calibration."""
        torch.manual_seed(seed)
        years = sorted(train)
        fit_y, val_y = years[:-val_years], years[-val_years:]
        doy, hist, r = _windows({y: train[y] for y in fit_y})
        vdoy, vhist, vr = _windows({y: train[y] for y in val_y})
        net = RainNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
        best, best_state, patience = np.inf, None, 0
        for ep in range(epochs):
            loss = _nll(*net(doy, hist), r).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                v = _nll(*net(vdoy, vhist), vr).mean().item()
            if v < best - 1e-4:
                best, best_state, patience = v, copy.deepcopy(net.state_dict()), 0
            else:
                patience += 1
            if ep % 200 == 0:
                log(f"  rainfall model epoch {ep:4d}  train nll {loss.item():.4f}  val nll {v:.4f}")
            if patience > 100:
                log(f"  early stop at epoch {ep}, best val nll {best:.4f}")
                break
        net.load_state_dict(best_state)
        model = cls(net)
        model._calibrate(train, seed)
        return model

    def _calibrate(self, train, seed, n: int = 300):
        """Match simulated wet-day frequency, annual mean, then CV of annual totals."""
        obs = np.stack([train[y] for y in sorted(train)])
        target_wet, target_mean = (obs >= WET_MM).mean(), obs.sum(1).mean()
        target_cv = obs.sum(1).std() / target_mean
        lo, hi = -2.0, 2.0
        for _ in range(14):
            self.logit_offset = (lo + hi) / 2
            if (self.sample(n, seed=seed) >= WET_MM).mean() < target_wet:
                lo = self.logit_offset
            else:
                hi = self.logit_offset
        for _ in range(3):                         # amount scale: fixed-point on the mean
            self.amount_scale *= target_mean / self.sample(n, seed=seed).sum(1).mean()
        lo, hi = 0.0, 1.0
        for _ in range(12):
            self.year_sigma = (lo + hi) / 2
            sim = self.sample(n, seed=seed).sum(1)
            if sim.std() / sim.mean() < target_cv:
                lo = self.year_sigma
            else:
                hi = self.year_sigma

    # --- sampling -------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_years: int, seed: int = 0, warmup_days: int = 60) -> np.ndarray:
        """Synthetic daily rainfall, shape (n_years, 365)."""
        g = torch.Generator().manual_seed(seed)
        H = max(HISTORY)
        mult = self.amount_scale * torch.exp(torch.randn(n_years, generator=g) * self.year_sigma
                                             - self.year_sigma ** 2 / 2)
        hist = torch.zeros(n_years, H)
        out = torch.zeros(n_years, 365)
        for step in range(-warmup_days, 365):
            d = torch.full((n_years,), float(step % 365))
            p_logit, shape, mean = self.net(d, hist)
            wet = torch.rand(n_years, generator=g) < torch.sigmoid(p_logit + self.logit_offset)
            gam = torch._standard_gamma(shape, generator=g) * (mean / shape) * mult
            r = torch.where(wet, torch.clamp(gam, min=WET_MM), torch.zeros(n_years))
            hist = torch.cat([hist[:, 1:], r[:, None]], 1)
            if step >= 0:
                out[:, step] = r
        return out.numpy()

    @torch.no_grad()
    def nll_per_day(self, data: dict[int, np.ndarray]) -> float:
        doy, hist, r = _windows(data)
        return float(_nll(*self.net(doy, hist), r).mean())

    def save(self, path: Path):
        torch.save({"state": self.net.state_dict(), "year_sigma": float(self.year_sigma),
                    "logit_offset": float(self.logit_offset),
                    "amount_scale": float(self.amount_scale)}, path)

    @classmethod
    def load(cls, path: Path) -> "RainfallModel":
        d = torch.load(path)
        net = RainNet()
        net.load_state_dict(d["state"])
        return cls(net, d["year_sigma"], d["logit_offset"], d["amount_scale"])


def climatology_nll(train: dict, test: dict) -> float:
    """Baseline: monthly wet-day probability and monthly gamma (method of moments)."""
    month = np.repeat(np.arange(12), np.diff([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]))
    tr = np.stack([train[y] for y in sorted(train)])
    params = []
    for m in range(12):
        x = tr[:, month == m].ravel()
        w = x[x >= WET_MM]
        p = np.clip((x >= WET_MM).mean(), 1e-3, 1 - 1e-3)
        k = w.mean() ** 2 / w.var() if len(w) > 2 else 1.0
        params.append((p, k, w.mean() if len(w) else 5.0))
    p, k, mu = (torch.tensor([params[m][j] for m in month], dtype=torch.float32) for j in range(3))
    _, _, r = _windows(test)
    n_years = len(test)
    idx = np.tile(np.arange(365), n_years)[max(HISTORY):]
    logit = torch.logit(p[idx])
    return float(_nll(logit, k[idx], mu[idx], r).mean())


def risk_report(model: RainfallModel, train: dict, test: dict, n_sim: int = 2000,
                seed: int = 0) -> dict:
    sim = model.sample(n_sim, seed=seed)
    ann = sim.sum(1)
    edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    monthly = np.stack([sim[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    obs_test = {y: v for y, v in test.items()}
    test_ann = np.array([v.sum() for v in obs_test.values()])
    maxday = sim.max(1)

    def pct_rank(x, ref):
        return float((ref < x).mean())

    def dry_spell(x):
        best = cur = 0
        for v in x[:273]:                       # Jan-Sep, before the NE monsoon
            cur = cur + 1 if v < WET_MM else 0
            best = max(best, cur)
        return best

    return dict(
        model_nll_test=model.nll_per_day(test),
        climatology_nll_test=climatology_nll(train, test),
        model_nll_train=model.nll_per_day(train),
        year_sigma=model.year_sigma,
        annual_mean_sim=float(ann.mean()), annual_sd_sim=float(ann.std()),
        annual_mean_train=float(np.mean([v.sum() for v in train.values()])),
        annual_q=dict(zip(["p05", "p25", "p50", "p75", "p95"],
                          np.percentile(ann, [5, 25, 50, 75, 95]).round(1).tolist())),
        p_annual_below_1000mm=float((ann < 1000).mean()),
        p_ond_below_500mm=float((sim[:, 273:].sum(1) < 500).mean()),
        maxday_return_levels={f"{T}yr": float(np.percentile(maxday, 100 * (1 - 1 / T)))
                              for T in (2, 10, 50)},
        longest_dry_spell_median_days=float(np.median([dry_spell(x) for x in sim[:300]])),
        test_years={str(y): dict(annual=float(v.sum()), percentile=pct_rank(v.sum(), ann),
                                 maxday=float(v.max()), maxday_percentile=pct_rank(v.max(), maxday))
                    for y, v in obs_test.items()},
        test_annual_inside_90pct=float(np.mean((test_ann > np.percentile(ann, 5)) &
                                               (test_ann < np.percentile(ann, 95)))),
        monthly_band=dict(p10=np.percentile(monthly, 10, 0).round(1).tolist(),
                          p50=np.percentile(monthly, 50, 0).round(1).tolist(),
                          p90=np.percentile(monthly, 90, 0).round(1).tolist(),
                          test_mean=np.mean([[v[a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
                                             for v in obs_test.values()], 0).round(1).tolist()),
    )


def write_report(report: dict, path: Path):
    path.write_text(json.dumps(report, indent=2))
