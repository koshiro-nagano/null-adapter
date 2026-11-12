"""Cross-check every ENVI .hdr of LIB-HSI against the layout assumed in phase2/src/libhsi.py.

Usage: python -I phase2/scripts/check_headers.py <root> <out.json>
Checks per cube: samples/lines/bands = 512/512/204, data type 4 (float32), byte order 0 (little endian),
interleave BIL, header offset 0, wavelengths == libhsi.WAVELENGTHS, .dat size, and that rgb/labels exist.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase2.src.libhsi import H, N_BANDS, W, WAVELENGTHS  # noqa: E402

EXPECT = {"samples": str(W), "lines": str(H), "bands": str(N_BANDS), "data type": "4", "byte order": "0",
          "interleave": "bil", "header offset": "0"}


def parse(text):
    out = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, re.M):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


root, out_path = Path(sys.argv[1]), sys.argv[2]
report, bad = {}, []
for split in ("train", "validation", "test"):
    hdrs = sorted((root / split / "reflectance_cubes").glob("*.hdr"))
    fields = Counter()
    for h in hdrs:
        d, sid = parse(h.read_text()), h.stem
        for k, v in EXPECT.items():
            if d.get(k, "").lower() != v:
                bad.append((split, sid, k, d.get(k)))
        wl = np.array([float(t) for t in d.get("wavelength", "{}").strip("{}").split(",") if t.strip()])
        if wl.shape != WAVELENGTHS.shape or np.abs(wl - WAVELENGTHS).max() > 1e-6:
            bad.append((split, sid, "wavelength", wl.shape))
        dat = h.with_suffix(".dat")
        if not dat.exists() or dat.stat().st_size != H * W * N_BANDS * 4:
            bad.append((split, sid, "dat size", dat.stat().st_size if dat.exists() else None))
        for sub in ("rgb", "labels"):
            if not (root / split / sub / f"{sid}.png").exists():
                bad.append((split, sid, f"missing {sub}", None))
        fields.update(k for k in d)
    report[split] = {"n_hdr": len(hdrs), "n_dat": len(list((root / split / "reflectance_cubes").glob("*.dat"))),
                     "n_rgb": len(list((root / split / "rgb").glob("*.png"))),
                     "n_labels": len(list((root / split / "labels").glob("*.png"))),
                     "fields": dict(fields)}
report["mismatches"] = [list(map(str, b)) for b in bad]
json.dump(report, open(out_path, "w"), indent=1)
for s in ("train", "validation", "test"):
    print(s, {k: v for k, v in report[s].items() if k != "fields"})
print("mismatches:", len(bad), bad[:10])
