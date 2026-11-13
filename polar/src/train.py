"""SAM-B + LoRA (SimCMF code) on ZJU-RGBP / RGBP-Glass with one projector per method; SimCMF's training recipe
(Adam 3e-4, batch 4, DiceCE, one random region + random click per image, StepLR x0.5) and its evaluation
(middle click per instance >= threshold, IoU at logit 0, mean over all instances).

Differences from SimCMF's train.py, the same for every method:
  * fixed number of iterations (default = SimCMF's 51 epochs over the full train set), LR halved every 10 epochs'
    worth of iterations, whatever the label budget;
  * the reported score is the last checkpoint (SimCMF keeps the best evaluation epoch, i.e. selects on the test
    set); the scores at intermediate checkpoints are logged, so the SimCMF-style "peak" can also be read off;
  * bf16 autocast; the label budget n_label takes the first n images of a seeded permutation (nested budgets).

  python -m polar.src.train --ds zju --method null_p --n_label 34 --seed 0 --out polar/results/zju
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import bridge
from .data import EvalSet, TrainSet, DrawSampler, eligible
from .inputs import METHODS, N_EXTRA, Inputs
from .projectors import GroupResidualProjector

EVAL_SPLIT = {"zju": "val", "pgsnet": "test"}


def build(method: str, ds: str):
    bridge.use_sdpa_attention()
    net, _ = bridge.build_sam_lora("simcmf" if method == "simcmf" else "zeroshot", "zju-rgbp" if ds == "zju" else "pgsnet_rgbp")
    if method != "simcmf":
        net.image_encoder.projector = GroupResidualProjector(N_EXTRA[method], net.image_encoder.patch_embed.proj)
    names = bridge.set_trainable_params(net, net_name="sam_lora")
    return net, names


@torch.no_grad()
def evaluate(net, inputs, loader, dev):
    net.eval()
    rows = []
    for raw, pts, tg, ids, cnt, i in loader:
        if pts.shape[1] == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = net.image_encoder(inputs(raw.to(dev, non_blocking=True)))
        n = pts.shape[1]
        coords = pts[0].to(dev).unsqueeze(1)                            # (n, 1, 2)
        labels = torch.ones(n, 1, dtype=torch.int, device=dev)
        se, de = net.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = net.mask_decoder(image_embeddings=emb, image_pe=net.prompt_encoder.get_dense_pe(),
                                       sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
        pb = (pred.float()[:, 0] > 0)
        gt = tg[0].to(dev) > 0
        inter, union = (pb & gt).sum((1, 2)).float(), (pb | gt).sum((1, 2)).float()
        iou = ((inter + 1e-6) / (union + 1e-6)).cpu().numpy()
        dice = ((2 * inter + 1e-4) / (pb.sum((1, 2)) + gt.sum((1, 2)) + 1e-4)).cpu().numpy()
        for k in range(n):
            rows.append((int(i), int(ids[0, k]), int(cnt[0, k]), float(iou[k]), float(dice[k])))
    net.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=("zju", "pgsnet"), required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--n_label", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None, help="default: 51 epochs of the full train set at batch 4")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--micro", type=int, default=4, help="micro-batch (gradient accumulation) to fit 16 GB")
    ap.add_argument("--n_eval", type=int, default=5, help="evaluations, evenly spaced, the last at the end")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out) / f"{a.method}_n{a.n_label}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    pool = eligible(a.ds, "train")
    perm = np.random.default_rng(1000 + a.seed).permutation(pool)
    lab = perm if a.n_label == "all" else perm[: int(a.n_label)]
    epoch_iters = math.ceil(len(pool) / a.bs)
    iters = a.iters or 51 * epoch_iters
    step_every = max(1, round(iters * 10 / 51))
    evals = sorted({round(iters * (k + 1) / a.n_eval) for k in range(a.n_eval)})
    log(f"{vars(a)} | labelled images {len(lab)} / {len(pool)} | iters {iters} | lr x0.5 every {step_every} | eval at {evals}")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    inputs = Inputs(a.ds, a.method, a.seed, dev, log)
    torch.manual_seed(a.seed)
    net, names = build(a.method, a.ds)
    net = net.to(dev).train()
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log(f"trainable parameters {n_train}")
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_every, gamma=0.5)
    from monai.losses import DiceCELoss
    lossf = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    tl = DataLoader(TrainSet(a.ds, lab, a.seed), batch_size=a.bs, sampler=DrawSampler(len(lab), a.seed),
                    num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    el = DataLoader(EvalSet(a.ds, EVAL_SPLIT[a.ds]), batch_size=1, num_workers=a.workers, pin_memory=True)
    hist, t0, run = [], time.time(), []
    it = iter(tl)
    for step in range(1, iters + 1):
        raw, pt, tg = next(it)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for mb in torch.split(torch.arange(len(raw)), a.micro):  # SAM has no batch-coupled layers: same gradient
            x = inputs(raw[mb].to(dev, non_blocking=True))
            prompt = (pt[mb].to(dev), torch.ones(len(mb), 1, device=dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x, prompt)
            l = lossf(pred.float(), tg[mb].to(dev)) * len(mb) / len(raw)
            l.backward()
            loss = loss + l.detach()
        opt.step(); sched.step()
        run.append(loss)
        if step % 100 == 0:
            log(f"step {step}/{iters} loss {float(torch.stack(run).mean()):.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{(time.time() - t0) / step:.3f}s/it")
            run = []
        if step in evals:
            rows = evaluate(net, inputs, el, dev)
            miou = float(np.mean([r[3] for r in rows]))
            hist.append({"step": step, "mIoU": miou, "mDice": float(np.mean([r[4] for r in rows])), "n_inst": len(rows)})
            log(f"eval step {step}: mIoU {100 * miou:.2f} over {len(rows)} instances")
    with open(out / "instances.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "region", "pixels", "iou", "dice"]); w.writerows(rows)
    res = {"ds": a.ds, "method": a.method, "n_label": len(lab), "seed": a.seed, "iters": iters,
           "mIoU": hist[-1]["mIoU"], "mDice": hist[-1]["mDice"], "peak_mIoU": max(h["mIoU"] for h in hist),
           "history": hist, "trainable": n_train, "seconds": time.time() - t0, "inputs_info": inputs.info}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    trainable = {k: v.detach().cpu() for k, v in net.state_dict().items() if k in set(names)}
    torch.save(trainable, out / "trainable.pt")
    log(f"done: mIoU {100 * res['mIoU']:.2f} (peak {100 * res['peak_mIoU']:.2f}) in {res['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
