"""Roof-material CNN: better runoff coefficients from overhead imagery.

    python -m rainmap.ml.roof_cnn train    --roofnet data/raw/roofnet
    python -m rainmap.ml.roof_cnn chips    --tiles "<XYZ tile URL>"      # per-building image chips
    python -m rainmap.ml.roof_cnn predict                                 # -> data/roof_predictions.csv
    python -m rainmap.ml.roof_cnn label-sheet                             # Chennai hand-label template
    python -m rainmap.ml.roof_cnn validate --labels data/chennai_roof_labels.csv

Training data: RoofNet (Law & Miura 2025, Kaggle ``noellelaw/roofnet``, CC BY-NC 4.0),
~51.5k image samples over 14 roof materials from 184 sites, India included.
Download needs a Kaggle API token. A ResNet-18 pretrained on ImageNet is
fine-tuned on it.

Chennai is not guaranteed to look like RoofNet's sites (flat RCC roofs, water
tanks and clutter on top), so predictions are marked *unvalidated* until a
hand-labelled Chennai sample is scored with ``validate``. Unvalidated
predictions are blended with the prior and given wider runoff uncertainty.

Output per building: class probabilities, expected runoff coefficient and its
standard deviation (from the class probabilities), which the uncertainty model
samples per scenario.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "roof_cnn.pt"
CHIPS = ROOT / "data" / "roof_chips"
PRED = ROOT / "data" / "roof_predictions.csv"
VALIDATION = ROOT / "outputs" / "roof_cnn_validation.json"

# RoofNet's 14 materials and screening runoff coefficients (share of rain reaching the tank).
CLASSES = ["thatch", "green_vegetative", "stone_slates", "clay_tiles", "asphalt_tiles",
           "concrete_tiles", "wood_tiles", "metal_sheet", "polycarbonate_sheet", "glass_sheet",
           "amorphous_fabric", "amorphous_membrane", "amorphous_concrete", "amorphous_asphalt"]
RUNOFF = np.array([0.50, 0.30, 0.80, 0.75, 0.85, 0.80, 0.70, 0.90, 0.90, 0.90,
                   0.60, 0.85, 0.85, 0.85])
CHIP_PX = 128
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# --- data ------------------------------------------------------------------------------
def _norm_label(s: str) -> str:
    s = str(s).lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    for c in CLASSES:
        if c in s or s in c:
            return c
    aliases = {"metal": "metal_sheet", "concrete": "amorphous_concrete", "tile": "clay_tiles",
               "asphalt": "asphalt_tiles", "green": "green_vegetative", "vegetat": "green_vegetative"}
    for k, v in aliases.items():
        if k in s:
            return v
    return ""


def roofnet_index(root: Path) -> pd.DataFrame:
    """Find (image path, class) pairs in a RoofNet download.

    Supports a metadata CSV with image and label columns, or class-named folders.
    The Kaggle layout should be checked after download; adjust here if it differs.
    """
    csvs = sorted(root.rglob("*.csv"))
    for c in csvs:
        df = pd.read_csv(c)
        img = next((k for k in df.columns if any(t in k.lower() for t in ("image", "path", "file"))), None)
        lab = next((k for k in df.columns if any(t in k.lower() for t in ("material", "label", "class"))), None)
        if img and lab:
            out = pd.DataFrame({"path": [str(next(root.rglob(Path(p).name), root / p)) for p in df[img]],
                                "label": df[lab].map(_norm_label)})
            return out[out.label != ""].reset_index(drop=True)
    rows = [(str(p), _norm_label(p.parent.name)) for p in root.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")]
    out = pd.DataFrame(rows, columns=["path", "label"])
    return out[out.label != ""].reset_index(drop=True)


class ImageTable(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        from torchvision import transforms as T
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(90),
               T.ColorJitter(0.2, 0.2, 0.2)] if train else []
        self.tf = T.Compose([T.Resize((CHIP_PX, CHIP_PX)), *aug, T.ToTensor(),
                             T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        r = self.df.iloc[i]
        y = CLASSES.index(r.label) if isinstance(r.get("label", None), str) and r.label in CLASSES else -1
        return self.tf(Image.open(r.path).convert("RGB")), y


def build_model(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


# --- training ----------------------------------------------------------------------------
def train(roofnet: Path, epochs: int = 8, batch: int = 64, lr: float = 3e-4, seed: int = 0):
    torch.manual_seed(seed)
    df = roofnet_index(roofnet)
    if df.empty:
        raise SystemExit(f"No labelled images found under {roofnet}; check the RoofNet layout.")
    print(f"{len(df)} images; class counts:\n{df.label.value_counts().to_string()}")
    rng = np.random.default_rng(seed)
    val_mask = rng.random(len(df)) < 0.15
    dl = torch.utils.data.DataLoader(ImageTable(df[~val_mask], True), batch, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(ImageTable(df[val_mask], False), batch, num_workers=2)
    counts = df[~val_mask].label.value_counts().reindex(CLASSES).fillna(0).values
    weight = torch.tensor(1 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=(weight / weight.mean()).to(device))
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, runoff_mae = _score(model, vl, device)
        print(f"epoch {ep}  val accuracy {acc:.3f}  runoff MAE {runoff_mae:.3f}")
        if acc > best:
            best = acc
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH} (val accuracy {best:.3f})")


@torch.no_grad()
def _score(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(torch.softmax(model(xb.to(device)), 1).cpu())
        ys.append(yb)
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    return float((p.argmax(1) == y).mean()), float(np.abs(p @ RUNOFF - RUNOFF[y]).mean())


# --- Chennai imagery ---------------------------------------------------------------------
def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def fetch_chips(footprints: Path, tile_url: str, zoom: int = 19, pad_m: float = 4.0):
    """Cut one chip per building from an XYZ tile service.

    ``tile_url`` like ``https://server/{z}/{y}/{x}`` or ``.../{z}/{x}/{y}.png``. Check the
    imagery provider's terms allow this use before running.
    """
    import io

    import geopandas as gpd
    import requests
    from PIL import Image
    b = gpd.read_parquet(footprints).to_crs(4326)
    CHIPS.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    session = requests.Session()
    rows = []
    for idx, geom in zip(b.index, b.geometry):
        minx, miny, maxx, maxy = geom.bounds
        dlat = pad_m / 111_320
        dlon = dlat / math.cos(math.radians(miny))
        x0, y0 = _tile_xy(maxy + dlat, minx - dlon, zoom)
        x1, y1 = _tile_xy(miny - dlat, maxx + dlon, zoom)
        canvas = Image.new("RGB", ((int(x1) - int(x0) + 1) * 256, (int(y1) - int(y0) + 1) * 256))
        for tx in range(int(x0), int(x1) + 1):
            for ty in range(int(y0), int(y1) + 1):
                if (tx, ty) not in cache:
                    r = session.get(tile_url.format(z=zoom, x=tx, y=ty), timeout=30)
                    r.raise_for_status()
                    cache[(tx, ty)] = Image.open(io.BytesIO(r.content)).convert("RGB")
                canvas.paste(cache[(tx, ty)], ((tx - int(x0)) * 256, (ty - int(y0)) * 256))
        box = tuple(int(v) for v in ((x0 - int(x0)) * 256, (y0 - int(y0)) * 256,
                                     (x1 - int(x0)) * 256, (y1 - int(y0)) * 256))
        rid = int(b.loc[idx, "row_id"])
        path = CHIPS / f"{b.loc[idx, 'neighbourhood']}_{rid}.png"
        canvas.crop(box).resize((CHIP_PX, CHIP_PX)).save(path)
        rows.append(dict(row=rid, neighbourhood=b.loc[idx, "neighbourhood"], path=str(path)))
    pd.DataFrame(rows).to_csv(CHIPS / "index.csv", index=False)
    print(f"wrote {len(rows)} chips to {CHIPS}")


@torch.no_grad()
def predict(index_csv: Path | None = None):
    df = pd.read_csv(index_csv or CHIPS / "index.csv")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    dl = torch.utils.data.DataLoader(ImageTable(df, False), 128)
    probs = torch.cat([torch.softmax(model(xb), 1) for xb, _ in dl]).numpy()
    out = df[["row", "neighbourhood"]].copy()
    for j, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, j]
    out["roof_type"] = [CLASSES[j] for j in probs.argmax(1)]
    out["runoff_mean"], out["runoff_sd"] = runoff_stats(probs)
    out.to_csv(PRED, index=False)
    print(f"wrote {PRED}")


def runoff_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = probs @ RUNOFF
    sd = np.sqrt(np.clip(probs @ RUNOFF ** 2 - mean ** 2, 0, None))
    return mean, sd


# --- Chennai validation --------------------------------------------------------------------
def label_sheet(n: int = 300, seed: int = 0):
    """Random chips to hand-label; fill the `label` column with one of CLASSES."""
    df = pd.read_csv(CHIPS / "index.csv").sample(frac=1, random_state=seed).head(n)
    df["label"] = ""
    path = ROOT / "data" / "chennai_roof_labels.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}; label each chip with one of: {', '.join(CLASSES)}")


def validate(labels: Path):
    df = pd.read_csv(labels)
    df = df[df.label.isin(CLASSES)]
    if df.empty:
        raise SystemExit("No labelled rows yet.")
    pred = pd.read_csv(PRED).merge(df[["row", "neighbourhood", "label"]], on=["row", "neighbourhood"])
    probs = pred[[f"p_{c}" for c in CLASSES]].values
    y = pred.label.map(CLASSES.index).values
    res = dict(n=int(len(pred)), accuracy=float((probs.argmax(1) == y).mean()),
               runoff_mae=float(np.abs(probs @ RUNOFF - RUNOFF[y]).mean()),
               prior_runoff_mae=float(np.abs(0.80 - RUNOFF[y]).mean()),
               confusion=pd.crosstab(pred.label, pred.roof_type).to_dict())
    VALIDATION.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "confusion"}, indent=2))


def building_runoff(buildings, neighbourhood: str, prior, prior_sd: float = 0.08):
    """Runoff mean/sd per building: CNN if available (blended if unvalidated), else prior."""
    prior = np.broadcast_to(np.asarray(prior, float), (len(buildings),))
    mean = prior.copy()
    sd = np.full(len(buildings), prior_sd)
    status = "prior (no CNN predictions)"
    if PRED.exists() and "source_row" in buildings:
        p = pd.read_csv(PRED)
        p = p[p.neighbourhood == neighbourhood].set_index("row")
        hit = buildings.source_row.isin(p.index).values
        if hit.any():
            pm = p.loc[buildings.source_row[hit], "runoff_mean"].values
            ps = p.loc[buildings.source_row[hit], "runoff_sd"].values
            validated = VALIDATION.exists() and \
                json.loads(VALIDATION.read_text())["runoff_mae"] < json.loads(VALIDATION.read_text())["prior_runoff_mae"]
            w = 1.0 if validated else 0.5
            mean[hit] = w * pm + (1 - w) * prior[hit]
            sd[hit] = np.sqrt(w * ps ** 2 + (1 - w) * prior_sd ** 2 + w * (1 - w) * (pm - prior[hit]) ** 2)
            status = f"CNN ({'validated on Chennai labels' if validated else 'UNVALIDATED, blended 50/50 with prior'})"
    return mean, sd, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "chips", "predict", "label-sheet", "validate"])
    ap.add_argument("--roofnet", default=str(ROOT / "data" / "raw" / "roofnet"))
    ap.add_argument("--footprints", default=str(ROOT / "data" / "raw" / "open_buildings_chennai.parquet"))
    ap.add_argument("--tiles")
    ap.add_argument("--labels", default=str(ROOT / "data" / "chennai_roof_labels.csv"))
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "train":
        train(Path(a.roofnet), a.epochs)
    elif a.cmd == "chips":
        if not a.tiles:
            raise SystemExit("--tiles is required (an XYZ imagery URL whose terms you have checked)")
        fetch_chips(Path(a.footprints), a.tiles)
    elif a.cmd == "predict":
        predict()
    elif a.cmd == "label-sheet":
        label_sheet()
    else:
        validate(Path(a.labels))


if __name__ == "__main__":
    main()
