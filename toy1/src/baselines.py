"""End-to-end baseline modeled on HSI-Adapter (Hurtado et al., RA-L 2026).

Per-pixel analogue of the official backbone (mmseg/models/backbones/hsi_adapter.py):
  * spectral module: per-pixel spectral transformer with skip connection, which for a
    single-token sequence is an MLP C -> 3 (adapter_hsi_modules_dino.py), randomly initialised;
  * its 3-channel output feeds the frozen "RGB foundation model" phi;
  * a trainable spectral adapter branch on the full HSI x is injected into phi's features
    through a zero-initialised gate (init_values=0.0 in configs/HSIAdapter/*.py);
  * a linear classification head; everything except phi is trained end to end on the
    K labeled samples. Only HSI x is used as input (as in HSI-Adapter).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import encoders as E
from .encoders import Standardizer, _t, mlp


class HSIAdapterLite(nn.Module):
    def __init__(self, phi: nn.Module, n_bands: int, latent: int, width: int):
        super().__init__()
        self.spectral = nn.Sequential(nn.Linear(n_bands, 32), nn.GELU(), nn.Linear(32, 32))
        self.spectral_out = nn.Linear(32, 3)
        self.phi = phi.requires_grad_(False)
        self.adapter = mlp(n_bands, latent, width, 2)
        self.gamma = nn.Parameter(torch.zeros(latent))  # zero-init injector gate
        self.head = nn.Linear(latent, 1)

    def forward(self, x_std):
        e = self.spectral[0](x_std)
        pseudo_rgb = self.spectral_out(self.spectral[2](self.spectral[1](e)) + e)  # skip connection
        f = self.phi(pseudo_rgb) + self.gamma * self.adapter(x_std)
        return self.head(f).squeeze(-1)


def fit_predict_hsia(phi, sx: Standardizer, x_tr, y_tr, x_eval: dict, cfg: dict, seed: int) -> dict:
    b = cfg["baselines"]["hsia"]
    torch.manual_seed(seed)
    model = HSIAdapterLite(phi, x_tr.shape[1], cfg["encoders"]["phi"]["latent"], b["adapter_width"]).to(E.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=b["lr"], weight_decay=b["weight_decay"])
    X, Y = _t(sx(x_tr)), _t(y_tr)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(b["steps"]):  # full batch (2K samples)
        loss = lossf(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return {k: (model(_t(sx(v))).cpu().numpy() > 0).astype(int) for k, v in x_eval.items()}
