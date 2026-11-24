"""Qualitative panels for the report: per selected evaluation image, the intensity image, the polarisation p, the
label-free prediction m(c) ~ E[p | c], the residual r = p - m(c) (all three on the same standardised scale), the
ground-truth instances with their middle clicks, and the predicted instances of every method (seed 0 checkpoints).

  python -m polar.scripts.qualitative [--seed 0] [--per 4] [--dev cuda]

Images are picked from the seed's instances.csv (per-image mean IoU): largest null_p - rgb, largest null_p - simcmf,
largest fuse_p - rgb, most negative null_p - rgb. Writes polar/results/qual/<ds>_n<label>/*.jpg and qual.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from polar.src.data import EvalSet, features, sam_norm
from polar.src.inputs import Inputs, down
from polar.src.train import EVAL_SPLIT, build

ROOT = Path("polar/results")
METHODS = ["simcmf", "rgb", "fuse_p", "null_p"]
SETTINGS = [("zju", "all"), ("zju", "20"), ("pgsnet", "all"), ("pgsnet", "100")]
PICKS = [("null_p", "rgb", 1, "null_p ≫ rgb"), ("null_p", "simcmf", 1, "null_p ≫ simcmf"),
         ("fuse_p", "rgb", 1, "fuse_p ≫ rgb"), ("null_p", "rgb", -1, "null_p ≪ rgb")]
W = 420  # display width


def per_image(ds, n, seed):
    out = {}
    for m in METHODS:
        df = pd.read_csv(ROOT / ds / f"{m}_n{n}_s{seed}" / "instances.csv")
        out[m] = df
    img = pd.DataFrame({m: d.groupby("image").iou.mean() for m, d in out.items()})
    return out, img


def pick(img, per):
    chosen, used = [], set()
    for a, b, sign, why in PICKS[:per]:
        d = (img[a] - img[b]) * sign
        for i in d.sort_values(ascending=False).index:
            if i not in used:
                used.add(i); chosen.append((int(i), why)); break
    return chosen


def colours(k):
    rng = np.random.default_rng(3)
    return (rng.uniform(0.25, 1, (k, 3)) * 255).astype(np.uint8)


def overlay(base, inst, ids, pal, alpha=0.55):
    """base (H, W, 3) uint8, inst list of (H, W) bool aligned with ids -> overlay."""
    out = base.astype(np.float32).copy()
    for m, r in zip(inst, ids):
        out[m] = (1 - alpha) * out[m] + alpha * pal[r % len(pal)]
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        out[edge] = pal[r % len(pal)]
    return out.clip(0, 255).astype(np.uint8)


def pol_rgb(v, scale):
    """v (6, H, W) standardised s1, s2 per colour -> HSV picture: hue = 2*angle, value = |v| / scale."""
    s1, s2 = v[:3].mean(0), v[3:].mean(0)
    mag = np.clip(np.hypot(s1, s2) / scale, 0, 1)
    hue = (np.arctan2(s2, s1) / (2 * np.pi)) % 1
    hsv = np.stack([hue * 255, np.full_like(hue, 230), mag * 255], -1).astype(np.uint8)
    return np.array(Image.fromarray(hsv, "HSV").convert("RGB"))


def to_disp(x, H, W_, nearest=False):
    """(…, 1024|256, same) -> display size keeping the native aspect."""
    h = round(W * H / W_)
    t = torch.as_tensor(x, dtype=torch.float32)
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None] if t.shape[-1] == 3 else t[None]
    t = F.interpolate(t, size=(h, W), mode="nearest" if nearest else "bilinear", align_corners=None if nearest else False)[0]
    return t.permute(1, 2, 0).numpy() if t.shape[0] == 3 else t.numpy() if t.shape[0] > 1 else t[0].numpy()


def save(a, path):
    Image.fromarray(np.asarray(a).astype(np.uint8)).save(path, quality=88)


@torch.no_grad()
def predict(net, x, pts, dev):
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        emb = net.image_encoder(x)
    n = pts.shape[0]
    se, de = net.prompt_encoder(points=(pts.to(dev).unsqueeze(1), torch.ones(n, 1, dtype=torch.int, device=dev)),
                                boxes=None, masks=None)
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        pred, _ = net.mask_decoder(image_embeddings=emb, image_pe=net.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
    return (pred.float()[:, 0] > 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--only", default=None, help="e.g. zju_n20")
    a = ap.parse_args()
    dev = torch.device(a.dev)
    meta = json.loads((ROOT / "qual" / "qual.json").read_text()) if (ROOT / "qual" / "qual.json").exists() else {}
    for ds, n in SETTINGS:
        key = f"{ds}_n{n}"
        if a.only and key != a.only:
            continue
        if not all((ROOT / ds / f"{m}_n{n}_s{a.seed}" / "trainable.pt").exists() for m in METHODS):
            continue
        out = ROOT / "qual" / key
        out.mkdir(parents=True, exist_ok=True)
        inst, img = per_image(ds, n, a.seed)
        chosen = pick(img, a.per)
        ev = EvalSet(ds, EVAL_SPLIT[ds])
        rows = []
        samples = {i: ev[i] for i, _ in chosen}
        preds = {i: {} for i, _ in chosen}
        nullin = None
        for m in METHODS:
            inputs = Inputs(ds, m, a.seed, dev, print)
            net, _ = build(m, ds)
            sd = torch.load(ROOT / ds / f"{m}_n{n}_s{a.seed}" / "trainable.pt", map_location="cpu")
            missing = net.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
            net = net.to(dev).eval()
            for i, _ in chosen:
                raw, pts, tg, ids, cnt, _ = samples[i]
                preds[i][m] = predict(net, inputs(raw[None].to(dev)), pts, dev)
            if m == "null_p":
                nullin = inputs
            del net
            torch.cuda.empty_cache() if dev.type == "cuda" else None
        for i, why in chosen:
            raw, pts, tg, ids, cnt, _ = samples[i]
            H, W_ = raw.shape[-2:]
            f = features(ds, raw[None].to(dev))
            c = f["c"][0].permute(1, 2, 0).clamp(0, 255).cpu().numpy()
            base = to_disp(c, H, W_).clip(0, 255)
            base = (base * min(4.0, 230 / max(np.percentile(base, 99), 1))).clip(0, 255)  # display gain only
            p = nullin._pstd(f)
            mc = F.interpolate(nullin.m(down(sam_norm(f["c"]))), size=p.shape[-2:], mode="bilinear", align_corners=False)
            r = p - mc
            scale = float(torch.quantile(p.abs().flatten()[::97].float(), 0.95)) or 1.0
            pv, mv, rv = (np.stack([to_disp(t[0, k].cpu().numpy(), H, W_) for k in range(6)]) for t in (p, mc, r))
            tiles = {"c": base, "p": pol_rgb(pv, scale), "m": pol_rgb(mv, scale), "r": pol_rgb(rv, scale)}
            pal = colours(256)
            gt = [to_disp(tg[k].numpy(), H, W_, nearest=True) > 0.5 for k in range(len(ids))]
            tiles["gt"] = overlay(base, gt, ids.tolist(), pal)
            ious, check = {}, {}
            for m in METHODS:
                pm = [to_disp(preds[i][m][k].astype(np.float32), H, W_, nearest=True) > 0.5 for k in range(len(ids))]
                tiles[m] = overlay(base, pm, ids.tolist(), pal)
                ious[m] = float(inst[m][inst[m].image == i].iou.mean())
                pb, gb = preds[i][m], tg.numpy() > 0
                check[m] = float(np.mean(((pb & gb).sum((1, 2)) + 1e-6) / ((pb | gb).sum((1, 2)) + 1e-6)))
            # clicks on the GT tile (1024 square -> display)
            sx, sy = W / 1024, round(W * H / W_) / 1024
            g = tiles["gt"]
            for (x, y) in pts.numpy():
                cx, cy = int(x * sx), int(y * sy)
                g[max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5] = 255
                g[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3] = 0
            files = {}
            for k, t in tiles.items():
                fn = f"{i:04d}_{k}.jpg"; save(t, out / fn); files[k] = f"{key}/{fn}"
            rows.append({"image": i, "name": ev.names[i], "why": why, "n_inst": int(len(ids)), "iou": ious, "iou_recomputed": check,
                         "files": files, "pol_scale": scale})
            print(key, i, why, {m: (round(v * 100, 1), round(check[m] * 100, 1)) for m, v in ious.items()})
        meta[key] = {"seed": a.seed, "images": rows}
        (ROOT / "qual" / "qual.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
