"""Report section for the orthogonalisation line of work: label-budget curve (RGB only vs null_orth vs null_orth2_g)
and the null_orth improvements (null_orth2, null_orth_g, null_orth2_g). Read live from polar/results/<ds>/<m>_n<n>_s<s>,
so re-running report.py after new runs picks them up.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polar.scripts.strata import boot, load_iou

BUDGETS = {"zju": [("5", 5), ("10", 10), ("20", 20), ("all", 344)],
           "pgsnet": [("10", 10), ("30", 30), ("100", 100), ("all", 3205)]}
DNAME = {"zju": "ZJU-RGBP", "pgsnet": "RGBP-Glass"}
CURVE = ("rgb", "null_orth", "null_orth2_g")
TABLE = ("rgb", "null_orth", "null_orth2", "null_orth_g", "null_orth2_g")
LAB = {"rgb": "RGB only", "null_orth": "null_orth", "null_orth2": "null_orth2", "null_orth_g": "null_orth_g",
       "null_orth2_g": "null_orth2_g"}
PAIRS = [("null_orth", "rgb"), ("null_orth2_g", "rgb"), ("null_orth2_g", "null_orth")]
CSS = """
:root{--null_orth:#6250d6;--null_orth2_g:#e87ba4}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--null_orth:#9085e9;--null_orth2_g:#d55181}}
:root[data-theme="dark"]{--null_orth:#9085e9;--null_orth2_g:#d55181}
.s-null_orth{stroke:var(--null_orth)}.s-null_orth2_g{stroke:var(--null_orth2_g)}
.f-null_orth{fill:var(--null_orth)}.f-null_orth2_g{fill:var(--null_orth2_g)}
"""


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def miou(root, ds, n, m):
    out = {}
    for s in range(10):
        f = root / ds / f"{m}_n{n}_s{s}" / "result.json"
        if f.exists():
            out[s] = 100 * json.loads(f.read_text())["mIoU"]
    return out


def ci_cell(v):
    c = "pos" if v[1] > 0 else "neg" if v[2] < 0 else "ns"
    return f"<td class='{c}'>{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]</td>"


def pair(root, ds, n, a, b, rng):
    sa, sb, sr = miou(root, ds, n, a), miou(root, ds, n, b), miou(root, ds, n, "rgb")
    ss = [s for s in sa if s in sb]
    if not ss:
        return None
    A = {(m, s): load_iou(ds, n, m, s) for m in (a, b) for s in ss}
    keys = sorted(A[(b, ss[0])])
    d = [np.array([A[(a, s)][k] - A[(b, s)][k] for k in keys]) for s in ss]
    rgb = {s: load_iou(ds, n, "rgb", s) for s in sr}
    dh = []
    for i, s in enumerate(ss):
        h = np.array([np.mean([rgb[t][k] for t in rgb if t != s]) for k in keys])
        dh.append(d[i][h < .5])
    return {"all": boot(d, rng), "hard": boot(dh, rng), "n": len(ss), "per_seed": [100 * x.mean() for x in d]}


def curve(root, ds):
    pts = BUDGETS[ds]
    W, H, L, R, T, B = 520, 240, 44, 96, 12, 40
    data = {m: [miou(root, ds, n, m) for n, _ in pts] for m in CURVE}
    vals = [v for m in CURVE for d in data[m] for v in d.values()]
    lo, hi = np.floor((min(vals) - 1) / 5) * 5, np.ceil((max(vals) + 1) / 5) * 5
    lx = np.log10([k for _, k in pts])
    x = lambda i: L + (lx[i] - lx[0]) / (lx[-1] - lx[0]) * (W - L - R)
    y = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="mIoU vs number of labelled images, {DNAME[ds]}">']
    for t in np.arange(lo, hi + 1e-9, 5):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.0f}</text>')
    for i, (n, k) in enumerate(pts):
        o.append(f'<text x="{x(i):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{k if n != "all" else f"全 {k}"}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" class="axis-title" text-anchor="middle">ラベル付き画像数 (対数軸)</text>')
    ends = []
    for m in CURVE:
        xy = [(x(i), y(np.mean(list(d.values()))), d) for i, d in enumerate(data[m]) if d]
        if not xy:
            continue
        o.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in xy)}" class="line s-{m}"/>')
        for (a, b, d), (n, k) in zip(xy, [p for p, d in zip(pts, data[m]) if d]):
            tip = f"{LAB[m]} · {k} 枚: 平均 {np.mean(list(d.values())):.2f} (seed {', '.join(f'{s}:{v:.1f}' for s, v in d.items())})"
            o.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" class="dot f-{m}" data-tip="{esc(tip)}"/>')
        ends.append([xy[-1][1], m, xy[-1][0]])
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 12)
    for yy, m, xx in ends:
        o.append(f'<text x="{W - R + 8}" y="{yy + 4:.1f}" class="lab small">{esc(LAB[m])}</text>')
    o.append("</svg>")
    return "".join(o)


def orth_section(root: Path):
    rng = np.random.default_rng(0)
    h = ["<h2 id='orth'>直交化 null_orth の少数ラベル性能と改善案</h2>",
         "<p>m(c) を使わない画素レベルの直交化 null_orth を軸に, (i) ラベル数をさらに減らした場合と, (ii) 2 つの改善案を調べた. "
         "ラベル部分集合は seed ごとに入れ子 (5 ⊂ 10 ⊂ 20 ⊂ 全), 学習反復数はラベル数によらず固定 (ZJU 4386, RGBP-Glass 8020).</p>",
         "<div class='tw'><table class='txt'><tr><th>手法</th><th>偏光枝への入力</th><th>projector での処理</th></tr>"
         "<tr><td>null_orth</td><td rowspan='4'>p を SAM の 16×16 パッチごとに中心化した c (3ch) で最小二乗回帰し, その成分を除いた残差 (パッチ平均は残す)</td>"
         "<td>E(c) + Σ w<sub>i</sub>E<sub>i</sub>(·), w はゼロ初期化</td></tr>"
         "<tr><td>null_orth2 (案 1)</td><td>偏光トークンからさらに RGB トークン E(c) 方向の成分を除く (画素 + 特徴の二重直交化)</td></tr>"
         "<tr><td>null_orth_g (案 2)</td><td>偏光トークンに RGB トークンから計算するゲート σ(v·E(c)+b) を掛ける (v=0, b=0 で開始 → 0.5). "
         "学習時は確率 0.3 でサンプルごとに偏光枝を丸ごと落とす (modality dropout, 評価時は落とさない)</td></tr>"
         "<tr><td>null_orth2_g (案 1+2)</td><td>二重直交化 + ゲート + modality dropout</td></tr></table></div>",
         "<h3>ラベル数と mIoU (seed 平均)</h3>",
         "<div class='legend'>" + "".join(f"<span class='bar' style='--c:var(--{m})'>{esc(LAB[m])}</span>" for m in CURVE) + "</div>",
         "<div class='grid2'>"]
    for ds in BUDGETS:
        h.append(f"<div class='card'><h3 style='margin-top:0'>{DNAME[ds]}</h3>{curve(root, ds)}</div>")
    h.append("</div><p class='muted'>点にカーソルを置くと seed 別の値. 線のない区間は未実行.</p>")
    h.append("<h3>mIoU (%) · 全手法</h3><div class='tw'><table><tr><th>設定</th>" +
             "".join(f"<th>{esc(LAB[m])}</th>" for m in TABLE) + "</tr>")
    for ds, pts in BUDGETS.items():
        for n, k in pts:
            row = {m: miou(root, ds, n, m) for m in TABLE}
            if not row["null_orth"] and not row["null_orth2_g"]:
                continue
            best = max(np.mean(list(v.values())) for v in row.values() if v)
            cells = []
            for m in TABLE:
                v = row[m]
                if not v:
                    cells.append("<td>–</td>"); continue
                mu = np.mean(list(v.values()))
                b = mu == best
                cells.append(f"<td data-tip='seed別: {', '.join(f'{s}:{x:.2f}' for s, x in v.items())}'>{'<b>' if b else ''}{mu:.2f}"
                             f"{'</b>' if b else ''}<span class='muted'> ({len(v)})</span></td>")
            h.append(f"<tr><td>{DNAME[ds]} · {k if n != 'all' else f'全 {k}'} 枚</td>{''.join(cells)}</tr>")
    h.append("</table></div><p class='muted'>括弧内は seed 数 (手法ごとに異なる; 平均は存在する seed で取る). 太字は各設定の最大.</p>")
    h.append("<h3>対比較 (pt, 2段 bootstrap 95% CI; 両手法に共通の seed で対応付け)</h3><div class='tw'><table><tr><th>設定</th><th>比較</th>"
             "<th>全インスタンス</th><th>RGB で難しいインスタンス (IoU &lt; 0.5)</th><th>seed 別の差</th><th>seed 数</th></tr>")
    for ds, pts in BUDGETS.items():
        for n, k in pts:
            for a, b in PAIRS:
                r = pair(root, ds, n, a, b, rng)
                if r:
                    h.append(f"<tr><td>{DNAME[ds]} · {k if n != 'all' else f'全 {k}'} 枚</td><td>{a} − {b}</td>{ci_cell(r['all'])}{ci_cell(r['hard'])}"
                             f"<td>{' / '.join(f'{x:+.1f}' for x in r['per_seed'])}</td><td>{r['n']}</td></tr>")
    h.append("</table></div><p class='muted'>緑/赤 = CI が 0 をまたがない. 「RGB で難しい」は比較する seed 以外の RGB only モデルで IoU &lt; 0.5 "
             "(leave-one-seed-out). RGBP-Glass 10 枚では RGB only の mIoU が 32% と低く, 大半のインスタンスがこの群に入る. "
             "多数の比較を並べているため, 偶然でもいくつかは有意になりうる.</p>")
    return "".join(h)
