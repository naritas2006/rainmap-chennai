"""Learned cluster proposer: a message-passing GNN that imitates the MILP.

Graph: buildings are nodes; edges are k-nearest neighbours plus every pipeable
building-hub link. The network predicts (a) which candidate hubs to open and
(b) for each building a choice among its links, "private tank" or "nothing".
Decoding opens the predicted hubs, assigns buildings, and hands the fixed
clustering to the MILP to size tanks (an easy problem once clusters are fixed),
so every GNN design is feasible and costed exactly like the solver's.

Trained by imitation on solver solutions of synthetic neighbourhoods with
randomised costs, then judged on quality gap, speed and transfer to larger and
real neighbourhoods. Built on PyTorch Geometric (``MessagePassing`` layers on
``Data`` graphs).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from .baselines import close_singletons
from .instance import Instance
from .optimize import Design, ScenarioSet, solve

K_NEAREST = 6
NODE_F, EDGE_F, GLOBAL_F = 8, 4, 8


def _globals(inst: Instance, sc: ScenarioSet) -> np.ndarray:
    c = inst.cfg.costs
    return np.array([c.water_value_per_m3 / 300, c.trench_per_m / 400, c.pipe_per_m / 150,
                     c.private_m3_per_roof_m2 * 10, c.tank_fixed / 25_000, c.hub_pump / 30_000,
                     c.tank_per_m3 / 9_000, sc.rain.sum(1).mean() / 1.3], np.float32)


def graph_tensors(inst: Instance, sc: ScenarioSet) -> dict:
    c = inst.cfg.costs
    b = inst.buildings
    xy = inst.xy
    n = inst.n
    harvest = (sc.roof * sc.rain.sum(1)[:, None]).mean(0)            # m3/yr per building
    demand = sc.demand.sum(1)
    is_hub = np.zeros(n)
    is_hub[inst.hubs] = 1
    space = c.private_m3_per_roof_m2 * b.area_m2.values
    x = np.c_[np.log1p(b.area_m2.values) / 5, b.confidence.values, np.log1p(demand) / 5,
              harvest / np.maximum(demand, 1e-6), (inst.elev - inst.elev.mean()) / 3,
              is_hub, space / 10, np.log1p(harvest) / 5].astype(np.float32)

    li, lh = inst.links()
    value = c.water_value_per_m3 * np.maximum(harvest[li], 1e-3)
    link_f = np.c_[inst.dist[li, lh] / 100, np.minimum(inst.link_cost[li, lh] / value, 10),
                   inst.gravity_ok[li, lh], np.ones(len(li))].astype(np.float32)

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    k = min(K_NEAREST, n - 1)
    nb = np.argsort(d, axis=1)[:, :k]
    ks, kd = np.repeat(np.arange(n), k), nb.ravel()
    knn_f = np.c_[d[ks, kd] / 100, np.zeros((len(ks), 2)), np.zeros(len(ks))].astype(np.float32)

    hub_node = inst.hubs[lh]
    src = np.r_[ks, li, hub_node]
    dst = np.r_[kd, hub_node, li]
    ef = np.r_[knn_f, link_f, link_f]
    t = torch.as_tensor
    return Data(x=t(x), edge_index=t(np.stack([src, dst])), edge_attr=t(ef),
                g=t(_globals(inst, sc))[None], hubs=t(inst.hubs), li=t(li), lh=t(lh),
                link_f=t(link_f), num_nodes=n)


def targets(inst: Instance, design: Design) -> dict:
    """Option index per building: link index, or L + building (private), or nothing."""
    li, lh = inst.links()
    L = len(li)
    choice = np.full(inst.n, 2)                                      # 0 link, 1 private, 2 none
    link_pick = np.full(inst.n, -1)
    for l, (i, h) in enumerate(zip(li, lh)):
        if design.assign[i] == h:
            choice[i], link_pick[i] = 0, l
    choice[(design.assign < 0) & (design.ind_size > 0)] = 1
    return dict(hub=torch.as_tensor(design.hub_open.astype(np.float32)),
                choice=torch.as_tensor(choice), link_pick=torch.as_tensor(link_pick), L=L)


def _mlp(i, o, hid):
    return nn.Sequential(nn.Linear(i, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, o))


class EdgeMLPConv(MessagePassing):
    """h_i <- LayerNorm(h_i + MLP([h_i, mean_j MLP([h_i, h_j, e_ij])]))."""

    def __init__(self, hid):
        super().__init__(aggr="mean")
        self.msg = _mlp(3 * hid, hid, hid)
        self.upd = _mlp(2 * hid, hid, hid)
        self.norm = nn.LayerNorm(hid)

    def forward(self, h, edge_index, e):
        agg = self.propagate(edge_index, h=h, e=e)
        return self.norm(h + self.upd(torch.cat([h, agg], 1)))

    def message(self, h_i, h_j, e):
        return self.msg(torch.cat([h_i, h_j, e], 1))


class RainGNN(nn.Module):
    def __init__(self, hid: int = 64, layers: int = 4):
        super().__init__()
        self.enc = _mlp(NODE_F + GLOBAL_F, hid, hid)
        self.eenc = _mlp(EDGE_F, hid, hid)
        self.convs = nn.ModuleList(EdgeMLPConv(hid) for _ in range(layers))
        self.hub_head = _mlp(hid, 1, hid)
        self.link_head = _mlp(2 * hid + EDGE_F, 1, hid)
        self.alone_head = _mlp(hid, 2, hid)                         # private, nothing

    def forward(self, gt: Data):
        n = gt.num_nodes
        h = self.enc(torch.cat([gt.x, gt.g.expand(n, -1)], 1))
        e = self.eenc(gt.edge_attr)
        for conv in self.convs:
            h = conv(h, gt.edge_index, e)
        hub_logit = self.hub_head(h[gt.hubs]).squeeze(1)
        link_logit = self.link_head(torch.cat([h[gt.li], h[gt.hubs[gt.lh]], gt.link_f], 1)).squeeze(1)
        return hub_logit, link_logit, self.alone_head(h)


def _option_logprobs(gt, link_logit, alone_logit):
    """Log-softmax over each building's options: its links, private, nothing."""
    n, li = gt.num_nodes, gt.li
    mx = torch.maximum(alone_logit.max(1).values,
                       torch.full((n,), -1e9).scatter_reduce(0, li, link_logit, "amax"))
    se = torch.exp(alone_logit - mx[:, None]).sum(1).index_add(0, li, torch.exp(link_logit - mx[li]))
    lse = mx + torch.log(se)
    return link_logit - lse[li], alone_logit - lse[:, None]


def loss_fn(model, gt, tg):
    hub_logit, link_logit, alone_logit = model(gt)
    l_hub = nn.functional.binary_cross_entropy_with_logits(hub_logit, tg["hub"])
    link_lp, alone_lp = _option_logprobs(gt, link_logit, alone_logit)
    n = gt.num_nodes
    lp = torch.where(tg["choice"] == 0, link_lp[tg["link_pick"].clamp(min=0)],
                     alone_lp[torch.arange(n), (tg["choice"] - 1).clamp(min=0)])
    return l_hub + (-lp).mean()


def train(data: list, val: list, epochs: int = 60, lr: float = 1e-3, seed: int = 0,
          log=print) -> RainGNN:
    torch.manual_seed(seed)
    model = RainGNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = np.inf, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for j in rng.permutation(len(data)):
            gt, tg = data[j]
            loss = loss_fn(model, gt, tg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_fn(model, gt, tg).item() for gt, tg in val]))
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"epoch {ep:3d}  train {tot / len(data):.4f}  val {v:.4f}")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def propose(model: RainGNN, inst: Instance, sc: ScenarioSet, threshold: float = 0.5):
    """Clustering from the GNN; returns (assign, inference seconds)."""
    t0 = time.perf_counter()
    gt = graph_tensors(inst, sc)
    model.eval()
    hub_logit, link_logit, alone_logit = model(gt)
    open_ = torch.sigmoid(hub_logit).numpy() > threshold
    li, lh = gt.li.numpy(), gt.lh.numpy()
    link_lp, alone_lp = _option_logprobs(gt, link_logit, alone_logit)
    best = alone_lp.max(1).values.numpy()
    assign = np.full(inst.n, -1)
    for l in np.argsort(-link_lp.numpy()):
        i, h = li[l], lh[l]
        if open_[h] and assign[i] < 0 and link_lp[l] > best[i]:
            assign[i] = h
    for h, k in enumerate(inst.hubs):
        if open_[h]:
            assign[k] = h
    return close_singletons(inst, assign), time.perf_counter() - t0


def gnn_design(model: RainGNN, inst: Instance, sc: ScenarioSet,
               thresholds=(0.5, 0.3, 0.15)) -> Design:
    """Decode at several hub thresholds; keep the clustering with the best exact objective.

    Marginal hubs get probabilities just under 0.5, so a single cut-off misses them.
    """
    best, t_total = None, 0.0
    tried = set()
    for th in thresholds:
        assign, t_inf = propose(model, inst, sc, threshold=th)
        t_total += t_inf
        key = assign.tobytes()
        if key in tried:
            continue
        tried.add(key)
        d = solve(inst, sc, method="gnn", fixed_assign=assign)
        t_total += d.runtime_s
        if best is None or d.objective > best.objective:
            best, best_th = d, th
    best.status = f"gnn, {len(tried)} decodes, best threshold {best_th}"
    best.runtime_s = t_total
    return best
