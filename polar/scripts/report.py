"""HTML report of the polarisation benchmark (quantitative + qualitative), self-contained (images inlined).

  python -m polar.scripts.report [--root polar/results] [--out polar/results/report.html]

Confidence intervals (all 95 %):
  * per method: t interval over seeds (training randomness incl. the label subset), and a two-level bootstrap that
    resamples seeds and evaluation instances (training randomness + finite test set);
  * per pair of methods: the same two on the paired difference (same seed = same label subset, same instances),
    plus the instance bootstrap of the seed-averaged difference (test-set sampling only, as in summarize.py).
Also: learning curves, IoU by instance size, the learned gate weights w of fuse_p / null_p, the R^2 of m(c).
Qualitative panels come from polar.scripts.qualitative (polar/results/qual/).
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from polar.scripts.orth_report import CSS as ORTH_CSS, orth_section

ORDER = ["simcmf", "rgb", "fuse_p", "null_p"]
LABEL = {"simcmf": "SimCMF", "rgb": "RGB only", "fuse_p": "fuse_p (RGB+p)", "null_p": "null_p (RGB+r)"}
PAIRS = [("null_p", "simcmf"), ("null_p", "rgb"), ("null_p", "fuse_p"), ("fuse_p", "rgb"), ("simcmf", "rgb")]
SETTINGS = [("zju", 344, "ZJU-RGBP", "全ラベル (344)"), ("zju", 20, "ZJU-RGBP", "20 ラベル"),
            ("pgsnet", 3205, "RGBP-Glass", "全ラベル (3205)"), ("pgsnet", 100, "RGBP-Glass", "100 ラベル")]
QKEY = {("zju", 344): "zju_nall", ("zju", 20): "zju_n20", ("pgsnet", 3205): "pgsnet_nall", ("pgsnet", 100): "pgsnet_n100"}
RNG = np.random.default_rng(0)
NB = 4000


# ---------------------------------------------------------------- data
def load(root: Path):
    runs = defaultdict(dict)
    for f in sorted(root.glob("*/*/result.json")):
        r = json.loads(f.read_text())
        with open(f.parent / "instances.csv") as fh:
            rows = list(csv.DictReader(fh))
        keys = [(int(x["image"]), int(x["region"])) for x in rows]
        r["_iou"] = dict(zip(keys, (float(x["iou"]) for x in rows)))
        r["_px"] = dict(zip(keys, (int(x["pixels"]) for x in rows)))
        r["_dir"] = f.parent
        runs[(r["ds"], r["n_label"])].setdefault(r["method"], {})[r["seed"]] = r
    return runs


def matrix(g, m, seeds, keys):
    return np.array([[g[m][s]["_iou"][k] for k in keys] for s in seeds])


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def hboot(M):
    """M (seeds, instances) -> 95 % CI of the grand mean, resampling seeds and instances."""
    S, N = M.shape
    out = np.empty(NB)
    for b in range(NB):
        cols = RNG.integers(0, N, N)
        per = M[:, cols].mean(1)
        out[b] = per[RNG.integers(0, S, S)].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def iboot(v):
    bs = v[RNG.integers(0, len(v), (NB, len(v)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def gate(d: Path):
    sd = torch.load(d / "trainable.pt", map_location="cpu")
    k = [k for k in sd if k.endswith("projector.fc.weight")]
    return sd[k[0]].flatten().tolist() if k else None


def analyse(runs):
    out = []
    for ds, n, dname, bname in SETTINGS:
        g = runs.get((ds, n))
        if not g:
            continue
        seeds = sorted(set.intersection(*[set(g[m]) for m in ORDER if m in g]))
        keys = sorted(set.intersection(*[set(g[m][s]["_iou"]) for m in ORDER for s in seeds]))
        px = np.array([g["rgb"][seeds[0]]["_px"][k] for k in keys])
        M = {m: matrix(g, m, seeds, keys) for m in ORDER}
        meth = {}
        for m in ORDER:
            per = M[m].mean(1)
            meth[m] = {"per_seed": per.tolist(), "mean": float(per.mean()),
                       "sd": float(per.std(ddof=1)) if len(per) > 1 else None, "t_ci": tci(per),
                       "h_ci": hboot(M[m]),
                       "peak": [g[m][s]["peak_mIoU"] for s in seeds],
                       "trainable": g[m][seeds[0]]["trainable"],
                       "hist": [[h["step"], h["mIoU"]] for h in g[m][seeds[0]]["history"]],
                       "hist_all": [[h["mIoU"] for h in g[m][s]["history"]] for s in seeds],
                       "minutes": float(np.mean([g[m][s]["seconds"] for s in seeds]) / 60)}
        pairs = {}
        for a, b in PAIRS:
            D = M[a] - M[b]
            per = D.mean(1)
            avg = D.mean(0)
            pairs[f"{a}-{b}"] = {"per_seed": per.tolist(), "mean": float(per.mean()), "t_ci": tci(per),
                                 "h_ci": hboot(D), "i_ci": iboot(avg),
                                 "win": float((avg > 0).mean()), "lose": float((avg < 0).mean())}
        q = np.quantile(px, [0.25, 0.5, 0.75])
        bins = np.digitize(px, q)
        edges = [int(px.min())] + [int(v) for v in q] + [int(px.max())]
        size = {m: [float(M[m][:, bins == j].mean()) for j in range(4)] for m in ORDER}
        size_n = [int((bins == j).sum()) for j in range(4)]
        size_d = {}
        for a, b in [("null_p", "rgb"), ("fuse_p", "rgb"), ("null_p", "simcmf")]:
            size_d[f"{a}-{b}"] = [(float((M[a] - M[b])[:, bins == j].mean()), iboot((M[a] - M[b])[:, bins == j].mean(0)))
                                  for j in range(4)]
        gates = {m: [gate(g[m][s]["_dir"]) for s in seeds] for m in ("fuse_p", "null_p")}
        r2 = [g["null_p"][s].get("inputs_info", {}).get("predictor", {}).get("r2_eval") for s in seeds]
        out.append({"ds": ds, "n": n, "dname": dname, "bname": bname, "seeds": seeds, "n_inst": len(keys),
                    "n_img": len({k[0] for k in keys}), "iters": g["rgb"][seeds[0]]["iters"],
                    "methods": meth, "pairs": pairs, "size": size, "size_n": size_n, "size_edges": edges,
                    "size_d": size_d, "gates": gates, "r2": r2})
    return out



# ---------------------------------------------------------------- non-neural m(c), strata (from their own JSONs)
NN_LABEL = {"rgb": "RGB only", "fuse_p": "fuse_p", "null_p": "null_p (CNN m)", "null_orth": "null_orth (画素直交)",
            "null_orthf": "null_orthf (特徴直交)", "null_phys": "null_phys (物理演算子)",
            "null_fres": "null_fres (法線+フレネル)"}
NN_SET = [("zju-all", "ZJU-RGBP · 全ラベル"), ("zju-20", "ZJU-RGBP · 20 ラベル"),
          ("pgsnet-all", "RGBP-Glass · 全ラベル"), ("pgsnet-100", "RGBP-Glass · 100 ラベル")]


def cip(v):  # [mean, lo, hi] in pt
    return f"<td class='{cls(v[1:3])}'>{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]</td>"


def nonneural_section(root: Path):
    f = root / "nonneural.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    ms = list(NN_LABEL)
    h = ["<h2 id='nonneural'>m(c) をニューラルネット以外にした場合</h2>",
         "<p>null_p の m(c) (CNN) を学習なし・ほぼ学習なしの演算に置き換えた. モジュール・学習手順・評価は null_p と同一で, 偏光枝への入力だけが異なる.</p>",
         "<div class='tw'><table class='txt'><tr><th>手法</th><th>偏光枝への入力</th><th>学習するもの</th></tr>"
         "<tr><td>null_orth</td><td>SAM の 16×16 パッチごとに p を (パッチ内で中心化した) c の 3ch で最小二乗回帰し, その成分を除いた残差 "
         "(u = g − proj<sub>c</sub> g の形の射影を入力に適用; パッチ平均の偏光は残す)</td><td>なし</td></tr>"
         "<tr><td>null_orthf</td><td>p をそのまま埋め込み, 偏光トークン Σ w<sub>i</sub>E<sub>i</sub>(p<sub>i</sub>) から RGB トークン E(c) 方向の成分をトークンごとに射影除去</td><td>なし (射影のみ)</td></tr>"
         "<tr><td>null_phys</td><td>r = p − m(c), m(c) = 偏光角が輝度勾配方位に平行/直交 (拡散/鏡面) という物理仮定の基底 × (エッジ強度, 鏡面性 min/mean, 輝度) の線形和</td>"
         "<td>色ごと 8 係数を train 画像で閉形式当てはめ (ラベル不使用)</td></tr>"
         "<tr><td>null_fres</td><td>r = p − m(c), m(c) = Metric3D v2 の単眼法線から天頂角 θ・方位角 α を出し, フレネル式 (η=1.5) の偏光度 ρ_d(θ) (拡散, 偏光角 α), ρ_s(θ) (鏡面, 偏光角 α+90°) を基底にした閉形式の線形当てはめ</td>"
         "<td>色ごと 8 係数を train 画像で閉形式当てはめ (ラベル不使用)</td></tr></table></div>",
         "<h3>mIoU (%)</h3><div class='tw'><table><tr><th>設定</th>" + "".join(f"<th>{esc(NN_LABEL[m])}</th>" for m in ms) + "</tr>"]
    for k, name in NN_SET:
        r = d.get(k)
        if not r:
            continue
        best = max(np.mean(v) for v in r["miou"].values())
        cells = []
        for m in ms:
            v = r["miou"].get(m)
            if not v:
                cells.append("<td>–</td>"); continue
            b = "<b>" if np.mean(v) == best else ""
            cells.append(f"<td data-tip='seed別: {', '.join(f'{x:.2f}' for x in v)}'>{b}{np.mean(v):.2f}{'</b>' if b else ''}"
                         f"<span class='muted'> ({len(v)})</span></td>")
        h.append(f"<tr><td>{esc(name)}</td>{''.join(cells)}</tr>")
    h.append("</table></div><p class='muted'>括弧内は seed 数. 太字は各設定の最大.</p>")
    h.append("<h3>m(c) の R² (評価集合, c から p を説明できる割合, 6ch 平均)</h3><div class='tw'><table><tr><th>設定</th>"
             "<th>null_p (CNN)</th><th>null_orth</th><th>null_phys</th><th>null_fres</th></tr>")
    for k, name in NN_SET:
        r = d.get(k, {}).get("r2", {})
        h.append(f"<tr><td>{esc(name)}</td>" + "".join(
            f"<td>{np.mean(r[m]):.3f}</td>" if m in r else "<td>–</td>" for m in ("null_p", "null_orth", "null_phys", "null_fres")) + "</tr>")
    h.append("</table></div>")
    h.append("<h3>対比較 (pt, 2段 bootstrap 95% CI; 同じ seed 同士で対応付け)</h3><div class='tw'><table><tr><th>比較</th><th>設定</th>"
             "<th>全インスタンス</th><th>RGB で難しいインスタンス (IoU &lt; 0.5)</th><th>seed 数</th></tr>")
    for a in ("null_orth", "null_orthf", "null_phys", "null_fres"):
        for b in ("null_p", "fuse_p", "rgb"):
            for k, name in NN_SET:
                r = d.get(k, {})
                v, hv = r.get("pairs", {}).get(f"{a}-{b}"), r.get("hard", {}).get(f"{a}-{b}")
                if v:
                    h.append(f"<tr><td>{a} − {b}</td><td>{esc(name)}</td>{cip(v)}{cip(hv)}<td>{v[3]}</td></tr>")
    h.append("</table></div><p class='muted'>「RGB で難しい」は, 比較する seed 以外の seed の RGB only モデルでの IoU が 0.5 未満のインスタンス "
             "(同じ seed で選ぶと平均への回帰で偏光側が有利に見えるため). 比較は計 36 組あり, 偶然でも 1〜2 組は有意になりうる.</p>")
    return "".join(h)


def strata_section(root: Path):
    f = root / "strata.json"
    if not f.exists():
        return ""
    d = json.loads(f.read_text())
    h = ["<h2 id='strata'>層別: ZJU のクラス別・RGB 難易度別</h2>",
         "<p>ZJU-RGBP の semantic ラベル (8 クラス) を各インスタンスに多数決で割り当てた (RGBP-Glass は glass の 1 クラスのみ). "
         "seed 0–2. 差は 2段 bootstrap 95% CI.</p>"]
    cc = d["train_class_counts"]
    cls_names = [c for c in cc["20-0"] if c != "background"]
    h.append("<h3>学習データ中のインスタンス数 (seed 0 / 1 / 2)</h3><div class='tw'><table><tr><th>ラベル数</th>" +
             "".join(f"<th>{c}</th>" for c in cls_names) + "</tr>")
    for n in ("all", "20"):
        h.append(f"<tr><td>{n}</td>" + "".join(f"<td>{' / '.join(str(cc[f'{n}-{s}'][c]) for s in range(3))}</td>" for c in cls_names) + "</tr>")
    h.append("</table></div>")
    for tag, name in (("zju-20", "ZJU-RGBP · 20 ラベル"), ("zju-all", "ZJU-RGBP · 全ラベル")):
        r = d["class"][tag]
        h.append(f"<h3>クラス別 · {esc(name)}</h3><div class='tw'><table><tr><th>クラス</th><th>n</th><th>RGB</th><th>fuse_p</th><th>null_p</th>"
                 "<th>null_p − fuse_p</th><th>null_p − rgb</th><th>fuse_p − rgb</th></tr>")
        for c, v in r.items():
            mi = v["miou"]
            h.append(f"<tr><td>{c}</td><td>{v['n']}</td><td>{mi['rgb']:.1f}</td><td>{mi['fuse_p']:.1f}</td><td>{mi['null_p']:.1f}</td>"
                     f"{cip(v['null_p-fuse_p'])}{cip(v['null_p-rgb'])}{cip(v['fuse_p-rgb'])}</tr>")
        h.append("</table></div>")
    h.append("<h3>RGB 難易度別 (leave-one-seed-out)</h3><div class='tw'><table><tr><th>設定</th><th>RGB IoU</th><th>n / seed</th>"
             "<th>RGB</th><th>fuse_p</th><th>null_p</th><th>null_p − fuse_p</th><th>null_p − rgb</th><th>fuse_p − rgb</th></tr>")
    for tag, name in NN_SET:
        for b, v in d["hard"].get(tag, {}).items():
            mi = v["miou"]
            h.append(f"<tr><td>{esc(name)}</td><td>{esc(b)}</td><td>{'/'.join(map(str, v['n_per_seed']))}</td>"
                     f"<td>{mi['rgb']:.1f}</td><td>{mi['fuse_p']:.1f}</td><td>{mi['null_p']:.1f}</td>"
                     f"{cip(v['null_p-fuse_p'])}{cip(v['null_p-rgb'])}{cip(v['fuse_p-rgb'])}</tr>")
    h.append("</table></div><p class='muted'>クラス別は 8 クラス × 2 設定 × 3 比較の多重比較. 評価インスタンスが 5〜17 個のクラスは区間が広く判断できない.</p>")
    return "".join(h)


# ---------------------------------------------------------------- svg helpers
def esc(s):
    return html.escape(str(s), quote=True)


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def spct(v, d=1):
    return f"{100 * v:+.{d}f}"


def ci_s(ci, sign=False, d=1):
    if ci is None:
        return "–"
    f = spct if sign else pct
    return f"[{f(ci[0], d)}, {f(ci[1], d)}]"


def nice_range(lo, hi, step):
    return np.floor(lo / step) * step, np.ceil(hi / step) * step


def method_dots(s):
    """Per method: seed dots, mean tick, t-CI whisker (thick) and two-level bootstrap CI (thin)."""
    W, H, L, R, T, B = 520, 210, 118, 20, 14, 34
    vals = []
    for m in ORDER:
        d = s["methods"][m]
        vals += d["per_seed"] + [c for c in (d["t_ci"] or []) + d["h_ci"]]
    lo, hi = nice_range(100 * min(vals) - 0.5, 100 * max(vals) + 0.5, 2)
    x = lambda v: L + (100 * v - lo) / (hi - lo) * (W - L - R)
    rowh = (H - T - B) / len(ORDER)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="mIoU per method, {esc(s["dname"])} {esc(s["bname"])}">']
    for t in np.arange(lo, hi + 1e-9, 2):
        o.append(f'<line x1="{x(t / 100):.1f}" x2="{x(t / 100):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                 f'<text x="{x(t / 100):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{t:.0f}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" class="axis-title" text-anchor="middle">mIoU (%)</text>')
    for j, m in enumerate(ORDER):
        d = s["methods"][m]
        y = T + rowh * (j + 0.5)
        o.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="lab" text-anchor="end">{esc(LABEL[m])}</text>')
        hc = d["h_ci"]
        o.append(f'<line x1="{x(hc[0]):.1f}" x2="{x(hc[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="whisk-thin s-{m}"/>')
        if d["t_ci"]:
            tc = d["t_ci"]
            o.append(f'<line x1="{x(tc[0]):.1f}" x2="{x(tc[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="whisk s-{m}"/>')
        for sd_, v in zip(s["seeds"], d["per_seed"]):
            o.append(f'<circle cx="{x(v):.1f}" cy="{y - 9:.1f}" r="4" class="dot f-{m}" '
                     f'data-tip="{esc(LABEL[m])} · seed {sd_}: {pct(v, 2)}"/>')
        tip = (f'{LABEL[m]} · 平均 {pct(d["mean"], 2)} · t-CI {ci_s(d["t_ci"], d=2)} · '
               f'2段bootstrap {ci_s(hc, d=2)}')
        o.append(f'<rect x="{x(d["mean"]) - 2:.1f}" y="{y - 7:.1f}" width="4" height="14" rx="1" class="f-{m}"/>'
                 f'<rect x="{L}" y="{y - rowh / 2:.1f}" width="{W - L - R}" height="{rowh:.1f}" class="hit" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def forest(sets, key="h_ci"):
    """Paired differences, all settings, one row per (setting, pair)."""
    rows = [(s, p) for p in [f"{a}-{b}" for a, b in PAIRS] for s in sets]
    W, L, R, T, B = 760, 250, 24, 10, 36
    rh = 20
    H = T + B + rh * len(rows) + 10 * (len(PAIRS) - 1)
    vals = [v for s, p in rows for v in s["pairs"][p][key] + (s["pairs"][p]["t_ci"] or [])]
    lo, hi = nice_range(100 * min(vals), 100 * max(vals), 2)
    lo, hi = min(lo, -2), max(hi, 2)
    x = lambda v: L + (100 * v - lo) / (hi - lo) * (W - L - R)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="paired differences in mIoU">']
    for t in np.arange(lo, hi + 1e-9, 2):
        o.append(f'<line x1="{x(t / 100):.1f}" x2="{x(t / 100):.1f}" y1="{T}" y2="{H - B}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{x(t / 100):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{t:+.0f}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" class="axis-title" text-anchor="middle">mIoU 差 (pt)　← 右側が優位 →</text>')
    y = T
    for gi, p in enumerate([f"{a}-{b}" for a, b in PAIRS]):
        a, b = p.split("-")
        for s in sets:
            d = s["pairs"][p]
            yc = y + rh / 2
            sig = d[key][0] > 0 or d[key][1] < 0
            o.append(f'<text x="{L - 10}" y="{yc + 4:.1f}" class="lab" text-anchor="end">{esc(a)} − {esc(b)} · '
                     f'{esc(s["dname"].split("-")[0])} {esc(s["bname"].split(" ")[0])}</text>')
            o.append(f'<line x1="{x(d[key][0]):.1f}" x2="{x(d[key][1]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" class="whisk s-{a}"/>')
            if d["t_ci"]:
                o.append(f'<line x1="{x(d["t_ci"][0]):.1f}" x2="{x(d["t_ci"][1]):.1f}" y1="{yc + 5:.1f}" y2="{yc + 5:.1f}" class="whisk-thin s-{a}"/>')
            o.append(f'<circle cx="{x(d["mean"]):.1f}" cy="{yc:.1f}" r="{5 if sig else 4}" class="{"f-" + a if sig else "hollow s-" + a}"/>')
            tip = (f'{a} − {b} · {s["dname"]} {s["bname"]}: {spct(d["mean"], 2)} pt · 2段bootstrap {ci_s(d["h_ci"], True, 2)}'
                   f' · seed t {ci_s(d["t_ci"], True, 2)} · seed別 {", ".join(spct(v, 2) for v in d["per_seed"])}')
            o.append(f'<rect x="0" y="{y:.1f}" width="{W}" height="{rh}" class="hit" data-tip="{esc(tip)}"/>')
            y += rh
        y += 10
    o.append("</svg>")
    return "".join(o)


def curves(s):
    W, H, L, R, T, B = 520, 230, 40, 64, 12, 34
    m0 = s["methods"]["rgb"]
    steps = [h[0] for h in m0["hist"]]
    means = {m: np.mean(s["methods"][m]["hist_all"], 0) for m in ORDER}
    allv = np.concatenate(list(means.values()))
    lo, hi = nice_range(100 * allv.min() - 0.5, 100 * allv.max() + 0.5, 5)
    x = lambda st: L + st / steps[-1] * (W - L - R)
    y = lambda v: T + (1 - (100 * v - lo) / (hi - lo)) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="learning curves">']
    for t in np.arange(lo, hi + 1e-9, 5):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t / 100):.1f}" y2="{y(t / 100):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{y(t / 100) + 4:.1f}" class="tick" text-anchor="end">{t:.0f}</text>')
    for st in steps:
        o.append(f'<text x="{x(st):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{st / 1000:.1f}k</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" class="axis-title" text-anchor="middle">iteration</text>')
    ends = []
    for m in ORDER:
        pts = " ".join(f"{x(st):.1f},{y(v):.1f}" for st, v in zip(steps, means[m]))
        o.append(f'<polyline points="{pts}" class="line s-{m}"/>')
        for st, v in zip(steps, means[m]):
            o.append(f'<circle cx="{x(st):.1f}" cy="{y(v):.1f}" r="4" class="dot f-{m}" '
                     f'data-tip="{esc(LABEL[m])} · step {st}: {pct(v, 2)} (seed平均)"/>')
        ends.append([y(means[m][-1]), m])
    ends.sort()
    for j in range(1, len(ends)):  # leader-free nudge only when far enough; otherwise legend carries identity
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 11)
    for yy, m in ends:
        o.append(f'<text x="{W - R + 6}" y="{yy + 4:.1f}" class="lab small">{esc(m)}</text>')
    o.append("</svg>")
    return "".join(o)


def size_chart(s):
    W, H, L, R, T, B = 420, 210, 40, 14, 12, 46
    vals = [v for m in ORDER for v in s["size"][m]]
    lo, hi = nice_range(100 * min(vals) - 1, 100 * max(vals) + 1, 10)
    bw = (W - L - R) / 4
    x = lambda j, k: L + bw * j + bw * (0.2 + 0.6 * k / (len(ORDER) - 1))
    y = lambda v: T + (1 - (100 * v - lo) / (hi - lo)) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="IoU by instance size">']
    for t in np.arange(lo, hi + 1e-9, 10):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t / 100):.1f}" y2="{y(t / 100):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{y(t / 100) + 4:.1f}" class="tick" text-anchor="end">{t:.0f}</text>')
    e = s["size_edges"]
    names = ["Q1 小", "Q2", "Q3", "Q4 大"]
    for j in range(4):
        o.append(f'<text x="{L + bw * (j + 0.5):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{names[j]}</text>'
                 f'<text x="{L + bw * (j + 0.5):.1f}" y="{H - B + 30}" class="tick" text-anchor="middle">n={s["size_n"][j]}</text>')
        for k, m in enumerate(ORDER):
            v = s["size"][m][j]
            o.append(f'<circle cx="{x(j, k):.1f}" cy="{y(v):.1f}" r="4.5" class="dot f-{m}" '
                     f'data-tip="{esc(LABEL[m])} · {names[j]} ({e[j]:,}–{e[j + 1]:,} px@1024²): {pct(v, 2)}"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 2}" class="axis-title" text-anchor="middle">インスタンス面積の四分位</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- html
CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--simcmf:#2a78d6;--rgb:#898781;--fuse_p:#eb6834;--null_p:#1baf7a;
--good:#006300;--bad:#d03b3b;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--simcmf:#3987e5;--rgb:#9a9890;--fuse_p:#d95926;--null_p:#199e70;--good:#0ca30c;--bad:#e66767;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--simcmf:#3987e5;--rgb:#9a9890;--fuse_p:#d95926;
--null_p:#199e70;--good:#0ca30c;--bad:#e66767;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:20px;margin:48px 0 10px;padding-top:8px;border-top:1px solid var(--grid)}
h3{font-size:16px;margin:24px 0 8px}p,li{color:var(--ink2);max-width:80ch}.sub{color:var(--muted);margin:0 0 20px}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;margin:16px 0 8px;font-size:14px}nav a{color:var(--accent);text-decoration:none}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin:12px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:12px}
.tile .v{font-size:26px;font-weight:600}.tile .l{color:var(--ink2);font-size:13px}.tile .d{color:var(--muted);font-size:12px}
.chart{width:100%;height:auto;display:block;font-family:inherit}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .zero{stroke:var(--axis);stroke-width:1.5}
.chart .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.chart .axis-title{fill:var(--muted);font-size:11px}
.chart .lab{fill:var(--ink2);font-size:12px}.chart .small{font-size:11px}
.chart .whisk{stroke-width:3;stroke-linecap:round}.chart .whisk-thin{stroke-width:1.5;stroke-linecap:round;opacity:.7}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.chart .stem{stroke-width:2;opacity:.35}
.chart .dot{stroke:var(--surface);stroke-width:2}.chart .hollow{fill:var(--surface);stroke-width:2}
.chart .hit{fill:transparent}.chart .hit:hover{fill:var(--grid);fill-opacity:.35}
.s-simcmf{stroke:var(--simcmf)}.s-rgb{stroke:var(--rgb)}.s-fuse_p{stroke:var(--fuse_p)}.s-null_p{stroke:var(--null_p)}
.f-simcmf{fill:var(--simcmf)}.f-rgb{fill:var(--rgb)}.f-fuse_p{fill:var(--fuse_p)}.f-null_p{fill:var(--null_p)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--ink2);margin:4px 0 8px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:var(--c);vertical-align:-1px}
.legend .bar::before{width:16px;height:3px;border-radius:2px;vertical-align:3px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:500}
td:first-child,th:first-child{text-align:left}.tw{overflow-x:auto}
.txt td,.txt th{text-align:left;white-space:normal}
.pos{color:var(--good);font-weight:600}.neg{color:var(--bad);font-weight:600}.ns{color:var(--muted)}.muted{color:var(--muted)}
.qrow{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,200px),1fr));gap:6px}
.qrow figure{margin:0}.qrow img{width:100%;display:block;border-radius:6px}
.qrow figcaption{font-size:12px;color:var(--ink2);padding:2px 2px 0}.qrow figcaption b{color:var(--ink)}
.qhead{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;margin:18px 0 6px;font-size:14px}
.qhead .why{font-weight:600}.qhead .nm{color:var(--muted);font-size:12px}
.callout{border-left:3px solid var(--accent);padding:4px 14px;margin:12px 0;background:var(--surface);border-radius:0 8px 8px 0}
code{font-size:13px;background:var(--surface);border:1px solid var(--border);padding:0 4px;border-radius:4px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;padding:6px 9px;
border-radius:6px;max-width:340px;opacity:0;transition:opacity .08s;z-index:9}
details summary{cursor:pointer;color:var(--ink2);font-size:14px}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
 if(!t){tip.style.opacity=0;return}tip.textContent=t.dataset.tip;tip.style.opacity=1;
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=e.clientY+16+tip.offsetHeight>innerHeight?e.clientY-tip.offsetHeight-10:e.clientY+16;
 tip.style.left=x+'px';tip.style.top=y+'px'});
"""


def legend(bar=False):
    return '<div class="legend">' + "".join(
        f'<span class="{"bar" if bar else ""}" style="--c:var(--{m})">{esc(LABEL[m])}</span>' for m in ORDER) + "</div>"


def cls(ci):
    return "pos" if ci[0] > 0 else "neg" if ci[1] < 0 else "ns"


def method_table(s):
    h = ["<div class='tw'><table><tr><th>手法</th><th>seed別 mIoU</th><th>平均 ± SD</th><th>95% CI (seed t)</th>"
         "<th>95% CI (2段 bootstrap)</th><th>peak 平均</th><th>学習可能パラメータ</th><th>学習時間</th></tr>"]
    for m in ORDER:
        d = s["methods"][m]
        sd = f" ± {100 * d['sd']:.2f}" if d["sd"] is not None else ""
        h.append(f"<tr><td>{esc(LABEL[m])}</td><td>{', '.join(pct(v, 2) for v in d['per_seed'])}</td>"
                 f"<td><b>{pct(d['mean'], 2)}</b>{sd}</td><td>{ci_s(d['t_ci'], d=2)}</td><td>{ci_s(d['h_ci'], d=2)}</td>"
                 f"<td>{100 * np.mean(d['peak']):.2f}</td><td>{d['trainable'] / 1e6:.2f} M</td><td>{d['minutes']:.0f} 分</td></tr>")
    return "".join(h) + "</table></div>"


def pair_table(sets):
    h = ["<div class='tw'><table><tr><th>比較</th><th>設定</th><th>差 (pt)</th><th>seed別</th><th>95% CI 2段 bootstrap</th>"
         "<th>95% CI seed t</th><th>95% CI instance bootstrap</th><th>勝 / 負 インスタンス</th></tr>"]
    for a, b in PAIRS:
        for s in sets:
            d = s["pairs"][f"{a}-{b}"]
            h.append(f"<tr><td>{a} − {b}</td><td>{esc(s['dname'])} · {esc(s['bname'])}</td>"
                     f"<td class='{cls(d['h_ci'])}'>{spct(d['mean'], 2)}</td>"
                     f"<td>{', '.join(spct(v, 2) for v in d['per_seed'])}</td>"
                     f"<td class='{cls(d['h_ci'])}'>{ci_s(d['h_ci'], True, 2)}</td>"
                     f"<td class='{cls(d['t_ci']) if d['t_ci'] else 'ns'}'>{ci_s(d['t_ci'], True, 2)}</td>"
                     f"<td class='{cls(d['i_ci'])}'>{ci_s(d['i_ci'], True, 2)}</td>"
                     f"<td>{100 * d['win']:.0f}% / {100 * d['lose']:.0f}%</td></tr>")
    return "".join(h) + "</table></div>"


def size_table(s):
    e = s["size_edges"]
    h = ["<div class='tw'><table><tr><th>四分位 (px @1024²)</th><th>n</th>" +
         "".join(f"<th>{esc(m)}</th>" for m in ORDER) + "<th>null_p − rgb [95% CI]</th><th>fuse_p − rgb [95% CI]</th></tr>"]
    for j in range(4):
        cells = "".join(f"<td>{pct(s['size'][m][j], 1)}</td>" for m in ORDER)
        dd = ""
        for p in ("null_p-rgb", "fuse_p-rgb"):
            v, ci = s["size_d"][p][j]
            dd += f"<td class='{cls(ci)}'>{spct(v, 1)} {ci_s(ci, True, 1)}</td>"
        h.append(f"<tr><td>{e[j]:,}–{e[j + 1]:,}</td><td>{s['size_n'][j]}</td>{cells}{dd}</tr>")
    return "".join(h) + "</table></div>"


def gate_table(sets):
    ch = ["s1/S0 R", "s1/S0 G", "s1/S0 B", "s2/S0 R", "s2/S0 G", "s2/S0 B"]
    h = ["<div class='tw'><table><tr><th>設定</th><th>手法</th>" + "".join(f"<th>w · {c}</th>" for c in ch) +
         "<th>‖w‖ (seed平均)</th><th>寄与比 median (p90), seed別</th><th>m(c) の R² (6ch の範囲)</th></tr>"]
    cf = Path("polar/results/contrib.json")
    contrib = json.loads(cf.read_text()) if cf.exists() else {}
    for s in sets:
        for m in ("fuse_p", "null_p"):
            G = np.array([g for g in s["gates"][m] if g is not None])
            if not len(G):
                continue
            mean = G.mean(0)
            r2 = ""
            cs = [contrib.get(f"{s['ds']}-{s['n']}-{m}-{sd}") for sd in s["seeds"]]
            cstr = ", ".join(f"{100 * c['median']:.1f}% ({100 * c['p90']:.0f}%)" for c in cs if c) or "–"
            if m == "null_p":
                R = np.array([r for r in s["r2"] if r is not None])
                r2 = f"{R.mean(0).min():.2f}–{R.mean(0).max():.2f}" if len(R) else "–"
            h.append(f"<tr><td>{esc(s['dname'])} · {esc(s['bname'])}</td><td>{m}</td>" +
                     "".join(f"<td>{v:+.3f}</td>" for v in mean) +
                     f"<td>{np.linalg.norm(G, axis=1).mean():.3f}</td><td>{cstr}</td><td>{r2}</td></tr>")
    return "".join(h) + "</table></div>"


def img64(p: Path):
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def qual_section(root: Path, sets):
    f = root / "qual" / "qual.json"
    if not f.exists():
        return "<p>定性結果はまだ生成されていません (<code>python -m polar.scripts.qualitative</code>)。</p>"
    meta = json.loads(f.read_text())
    cap = {"c": "強度画像 c", "p": "偏光 p (標準化)", "m": "m(c) ≈ E[p|c]", "r": "残差 r = p − m(c)", "gt": "正解 + クリック"}
    o = []
    for s in sets:
        k = QKEY[(s["ds"], s["n"])]
        if k not in meta:
            continue
        o.append(f"<h3>{esc(s['dname'])} · {esc(s['bname'])} (seed {meta[k]['seed']})</h3>")
        for r in meta[k]["images"]:
            o.append(f"<div class='qhead'><span class='why'>{esc(r['why'])}</span><span>画像 #{r['image']} · "
                     f"{r['n_inst']} インスタンス</span><span class='nm'>{esc(r['name'])}</span></div><div class='qrow'>")
            for t in ["c", "p", "m", "r", "gt"] + ORDER:
                c = cap.get(t) or f"{LABEL[t]} <b>{pct(r['iou'][t])}</b>"
                o.append(f"<figure><img loading='lazy' src='{img64(root / 'qual' / r['files'][t])}' alt='{esc(t)}'>"
                         f"<figcaption>{c}</figcaption></figure>")
            o.append("</div>")
    return "".join(o)


def tiles(sets):
    o = ['<div class="tiles">']
    for s in sets:
        d = s["pairs"]["null_p-simcmf"]
        e = s["pairs"]["null_p-rgb"]
        o.append(f'<div class="card tile"><div class="l">{esc(s["dname"])} · {esc(s["bname"])} · null_p − SimCMF</div>'
                 f'<div class="v">{spct(d["mean"])} pt</div>'
                 f'<div class="d">95% CI {ci_s(d["h_ci"], True)} · 対 RGB only {spct(e["mean"])} {ci_s(e["h_ci"], True)}</div></div>')
    return "".join(o) + "</div>"


def build_html(sets, root, findings):
    seeds_txt = ", ".join(f"{s['dname']} {s['bname']}: {len(s['seeds'])} seed" for s in sets)
    o = [f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Polarisation Null-Adapter Report</title><style>{CSS}{ORTH_CSS}</style></head><body><main>",
         "<h1>偏光ベンチマーク: SimCMF vs Null-Adapter</h1>",
         f"<p class='sub'>SAM-B + LoRA (SimCMF 実装), クリック 1 点のインスタンス分割, ZJU-RGBP / RGBP-Glass · {esc(seeds_txt)}</p>",
         "<nav><a href='#summary'>要約</a><a href='#setup'>設定</a><a href='#main'>主結果</a><a href='#pairs'>対比較</a>"
         "<a href='#curves'>学習曲線</a><a href='#size'>サイズ別</a><a href='#gate'>ゲート重み・m(c)</a><a href='#nonneural'>非ニューラル m(c)</a><a href='#orth'>直交化: 少数ラベル・改善案</a><a href='#strata'>層別</a><a href='#qual'>定性結果</a>"
         "<a href='#notes'>注意点</a></nav>",
         "<h2 id='summary'>要約</h2>", tiles(sets), findings,
         "<h2 id='setup'>実験設定</h2>",
         "<div class='tw'><table class='txt'><tr><th>手法</th><th>SAM パッチ埋め込みへの入力</th></tr>"
         "<tr><td>SimCMF</td><td>SimCMF の projector に [I, AoLP, DoLP] 9ch (SimCMF の正規化)</td></tr>"
         "<tr><td>RGB only</td><td>強度画像 c のみ, 事前学習済み埋め込み</td></tr>"
         "<tr><td>fuse_p</td><td>E(c) + Σ w<sub>i</sub> E<sub>i</sub>(p<sub>i</sub>), p = 色ごとの s1/S0, s2/S0 (6ch), w はゼロ初期化</td></tr>"
         "<tr><td>null_p</td><td>同じモジュールに r = p − m(c). m ≈ E[p|c] はラベルなしで全 train 画像の (c, p) 組から学習した CNN</td></tr>"
         "</table></div>",
         "<p>全手法で LoRA (q, v) + projector のみ学習. 学習手順と評価 (インスタンスの中央クリック, logit 0 で 2 値化した IoU の全インスタンス平均) は SimCMF のもの. "
         "報告値は最終チェックポイント (SimCMF はテスト集合上の最良エポックを採るため, 参考に peak も併記). "
         "seed はラベル部分集合 (入れ子), m(c) の学習, LoRA 初期化・データ順をすべて変える. "
         "ZJU-RGBP は val (36 枚 277 インスタンス), RGBP-Glass は test (5317 インスタンス) で評価.</p>",
         "<div class='callout'><b>信頼区間の読み方.</b> <i>seed t</i> は seed 間の t 区間 (学習の揺らぎのみ, seed 数 3 で自由度 2 のため幅が広い). "
         "<i>2段 bootstrap</i> は seed とインスタンスの両方を復元抽出 (学習の揺らぎ + 有限テスト集合) — 本レポートの主たる区間. "
         "<i>instance bootstrap</i> は seed 平均した IoU 差のインスタンス抽出のみ (テスト集合の揺らぎのみ; 従来の summarize.py と同じ).</div>",
         "<h2 id='main'>主結果: 手法ごとの mIoU</h2>",
         "<p>点 = seed ごとの mIoU, 縦棒 = seed 平均, 太線 = 95% CI (seed t), 細線 = 95% CI (2段 bootstrap).</p>", legend(),
         "<div class='grid2'>"]
    for s in sets:
        o.append(f"<div class='card'><h3 style='margin-top:0'>{esc(s['dname'])} · {esc(s['bname'])}</h3>{method_dots(s)}"
                 f"<details><summary>表で見る</summary>{method_table(s)}</details></div>")
    o.append("</div>")
    o += ["<h2 id='pairs'>対比較 (同一 seed・同一インスタンスでの差)</h2>",
          "<p>塗りの点 = 2段 bootstrap の 95% CI が 0 をまたがない差, 白抜き = またぐ差. 太線 = 2段 bootstrap CI, 下の細線 = seed t CI.</p>",
          f"<div class='card'>{forest(sets)}</div>", "<details open><summary>表</summary>", pair_table(sets), "</details>",
          "<h2 id='curves'>学習曲線 (評価集合 mIoU, seed 平均)</h2>", legend(True), "<div class='grid2'>"]
    for s in sets:
        o.append(f"<div class='card'><h3 style='margin-top:0'>{esc(s['dname'])} · {esc(s['bname'])}</h3>{curves(s)}</div>")
    o += ["</div>", "<h2 id='size'>インスタンス面積別の IoU (seed 平均)</h2>",
          "<p>評価インスタンスを面積 (1024×1024 換算画素数) の四分位で 4 群に分割. 差の CI はインスタンス bootstrap.</p>", legend(), "<div class='grid2'>"]
    for s in sets:
        o.append(f"<div class='card'><h3 style='margin-top:0'>{esc(s['dname'])} · {esc(s['bname'])}</h3>{size_chart(s)}"
                 f"<details><summary>表で見る</summary>{size_table(s)}</details></div>")
    o += ["</div>", "<h2 id='gate'>偏光枝のゲート重み w と m(c) の当てはまり</h2>",
          "<p>w は追加チャネルの埋め込みを RGB 埋め込みに足し込む係数 (ゼロ初期化). p と r はどちらも標準化済みなので大きさは比較可能. "
          "寄与比 = ‖Σ w<sub>i</sub> E<sub>i</sub>(·)‖ / ‖E(c)‖ をパッチトークンごとに計算した値 (評価画像 16 枚; E<sub>i</sub> も学習されるため w 単独では大きさが決まらない). "
          "R² は評価集合での「c から p を予測できる割合」(チャネル順は表の w と同じ).</p>",
          gate_table(sets), nonneural_section(root), orth_section(root), strata_section(root), "<h2 id='qual'>定性結果</h2>",
          "<p>各設定で seed 0 のモデルを使い, 画像単位の mIoU 差が大きい例を選んだ (null_p ≫ rgb, null_p ≫ SimCMF, fuse_p ≫ rgb, null_p ≪ rgb). "
          "成功例だけでなく失敗例も含む. 偏光の図は色相 = 偏光角 (2θ), 明度 = 偏光度 (3 色平均) で, p, m(c), r は同じスケール. "
          "予測パネルの数値はその画像のインスタンス平均 IoU. 暗い画像は表示用にのみ明るさを自動調整した (モデル入力は無加工).</p>",
          qual_section(root, sets),
          "<h2 id='notes'>注意点</h2><ul>"
          "<li>ZJU-RGBP の評価集合は 36 枚 277 インスタンスと小さく, 区間は広い.</li>"
          "<li>seed は 3 つ. seed t の区間は自由度 2 で保守的; seed 間の揺らぎを過小評価しないよう 2段 bootstrap を主に使った.</li>"
          "<li>評価集合での選択はしていない (最終チェックポイント). peak は参考値.</li>"
          "<li>定性例は差の大きい画像を選んだもので, 平均的な挙動ではない.</li>"
          "<li><b>評価コードの修正.</b> train.py の evaluate() は入力 (null_p では m(c) の CNN) を bf16 autocast 内で計算しており, fp32 で計算する学習時と食い違っていた. "
          "修正後に null_p の全 12 チェックポイントの最終値を再評価した (変化は −0.6〜+0.7 pt, 向きは一定でない; 旧値は result.json の mIoU_orig と instances_orig.csv). "
          "学習曲線と peak は旧評価のまま. 他の 3 手法は入力が要素演算のみで影響なし (再評価で完全一致を確認).</li>"
          "<li>定性パネルの表示 IoU は instances.csv の値で, 再推論の値と一致することを確認済み.</li>"
          "<li>2段 bootstrap は seed を 3 個から復元抽出するため, seed 間分散をやや過小評価しうる (係数 (n−1)/n). seed t 区間と併せて読むこと.</li></ul>",
          "<div id='tip'></div><script>", JS, "</script></main></body></html>"]
    return "".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="polar/results")
    ap.add_argument("--out", default="polar/results/report.html")
    ap.add_argument("--findings", default="polar/results/findings.html", help="HTML fragment for the summary")
    a = ap.parse_args()
    root = Path(a.root)
    sets = analyse(load(root))
    (root / "report_stats.json").write_text(json.dumps(sets, indent=1, default=str))
    fp = Path(a.findings)
    findings = fp.read_text() if fp.exists() else ""
    Path(a.out).write_text(build_html(sets, root, findings))
    for s in sets:
        print(f"\n== {s['dname']} {s['bname']}  seeds {s['seeds']}  {s['n_inst']} instances")
        for m in ORDER:
            d = s["methods"][m]
            print(f"  {m:7s} {pct(d['mean'], 2)}  t {ci_s(d['t_ci'], d=2)}  h {ci_s(d['h_ci'], d=2)}")
        for p, d in s["pairs"].items():
            print(f"  {p:14s} {spct(d['mean'], 2)}  h {ci_s(d['h_ci'], True, 2)}  t {ci_s(d['t_ci'], True, 2)}  i {ci_s(d['i_ci'], True, 2)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
