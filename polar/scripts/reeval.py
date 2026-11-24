"""Re-evaluate saved checkpoints with the current polar.src.train.evaluate (final checkpoint only).

  python -m polar.scripts.reeval --method null_p      # all datasets / budgets / seeds found

Used after fixing evaluate() to build the network input outside bf16 autocast, as training does (before the fix,
null_p's m(c) ran in bf16 at evaluation only). The old instances.csv / mIoU are kept as instances_orig.csv and
result.json["mIoU_orig"]; history and peak_mIoU are the logged values (old evaluation).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from polar.src.data import EvalSet
from polar.src.inputs import Inputs
from polar.src.train import EVAL_SPLIT, build, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="null_p")
    ap.add_argument("--root", default="polar/results")
    a = ap.parse_args()
    dev = torch.device("cuda")
    for f in sorted(Path(a.root).glob(f"*/{a.method}_n*_s*/result.json")):
        d = f.parent
        res = json.loads(f.read_text())
        if "mIoU_orig" in res:
            continue
        inputs = Inputs(res["ds"], res["method"], res["seed"], dev, lambda *_: None)
        net, _ = build(res["method"], res["ds"])
        net.load_state_dict(torch.load(d / "trainable.pt", map_location="cpu"), strict=False)
        net = net.to(dev).eval()
        el = DataLoader(EvalSet(res["ds"], EVAL_SPLIT[res["ds"]]), batch_size=1, num_workers=8, pin_memory=True)
        rows = evaluate(net, inputs, el, dev)
        shutil.copy(d / "instances.csv", d / "instances_orig.csv")
        with open(d / "instances.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "region", "pixels", "iou", "dice"]); w.writerows(rows)
        res["mIoU_orig"], res["mDice_orig"] = res["mIoU"], res["mDice"]
        res["mIoU"] = float(np.mean([r[3] for r in rows])); res["mDice"] = float(np.mean([r[4] for r in rows]))
        res["reeval"] = "final checkpoint re-evaluated with fp32 inputs (evaluate fix); history/peak are the old evaluation"
        f.write_text(json.dumps(res, indent=1))
        print(f"{d}: {100 * res['mIoU_orig']:.2f} -> {100 * res['mIoU']:.2f}", flush=True)
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
