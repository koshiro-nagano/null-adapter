"""Projectors that replace SimCMF's `image_encoder.projector` (forward(input, func), func = frozen SAM patch embed).

  rgb     func(c)                                     RGB-only (+ LoRA)
  fuse_p  func(c) + sum_i w_i E_i(p_i)                RGB + polarisation, simple additive fusion
  null_p  func(c) + sum_i w_i E_i(r_i), r = p - m(c)  RGB + conditional polarisation residual (Null-Adapter)
  null_orthf  func(c) + P_perp(func(c)) sum_i w_i E_i(p_i)   polarisation token projected orthogonal to the RGB token

Options (any input):
  orth   project the polarisation token orthogonal to the RGB token (null_orthf; null_orth2* = pixel- + token-level)
  gate   per-token gate sigmoid(v . func(c) + b), v = 0, b = 0 at start (0.5 everywhere): lets the model learn where
         (by the RGB token) polarisation is added; w still starts at zero, so the start is still the pretrained SAM
  mdrop  modality dropout: during training the whole polarisation branch of a sample is dropped with this probability
         (no rescaling), so the RGB path must stay sufficient on its own

E_i is a 1-channel 16 x 16 patch embedding initialised, as in SimCMF's novel branch, with the pretrained RGB patch
embedding summed over colour; w (Linear(k, 1), no bias) starts at zero, so all three start as the pretrained SAM on c.
fuse_p and null_p are the same module; only the extra input differs.
"""
from __future__ import annotations

import torch
from torch import nn


class GroupResidualProjector(nn.Module):
    def __init__(self, n_extra: int, patch_embed: nn.Conv2d, orth: bool = False, gate: bool = False, mdrop: float = 0.0):
        super().__init__()
        self.n_extra, self.orth, self.mdrop = n_extra, orth, mdrop
        self.gate = None
        if n_extra:
            w = patch_embed.weight.detach().sum(1, keepdim=True)
            b = patch_embed.bias.detach()
            self.group_patch_embedding = nn.ModuleList()
            for _ in range(n_extra):
                e = nn.Conv2d(1, w.shape[0], 16, stride=16)
                e.load_state_dict({"weight": w.clone(), "bias": b.clone()})
                self.group_patch_embedding.append(e)
            self.fc = nn.Linear(n_extra, 1, bias=False)
            nn.init.zeros_(self.fc.weight)
            if gate:
                self.gate = nn.Linear(w.shape[0], 1)
                nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)

    def forward(self, input: torch.Tensor, func):
        out = func(input[:, :3])
        if self.n_extra:
            u = input[:, 3:].unsqueeze(2)
            novel = torch.stack([self.group_patch_embedding[i](u[:, i]) for i in range(self.n_extra)], -1)  # B,C,h,w,k
            add = self.fc(novel.permute(0, 2, 3, 1, 4))[..., 0]                              # B,h,w,C (as func)
            if self.orth:  # null_orthf: keep only the part of the polarisation token orthogonal to the RGB token
                a, o = add.float(), out.float()
                add = (a - (a * o).sum(-1, keepdim=True) / (o * o).sum(-1, keepdim=True).clamp(min=1e-6) * o).to(add.dtype)
            if self.gate is not None:
                add = add * torch.sigmoid(self.gate(out))                                       # B,h,w,1
            if self.training and self.mdrop > 0:
                add = add * (torch.rand(add.shape[0], 1, 1, 1, device=add.device) >= self.mdrop).to(add.dtype)
            out = out + add
        return out, None


class ResidualPredictor(nn.Module):
    """Label-free m(c) ~ E[p | c]: dilated residual CNN on the SAM-normalised intensity image (run at 512 x 512;
    receptive field 47 px there)."""

    def __init__(self, n_out: int = 6, width: int = 64, dilations=(1, 2, 4, 8, 4, 2, 1)):
        super().__init__()
        self.inp = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), nn.Conv2d(width, width, 3, padding=d, dilation=d),
                                                  nn.GELU(), nn.Conv2d(width, width, 1)) for d in dilations)
        self.out = nn.Conv2d(width, n_out, 1)

    def forward(self, c):
        h = self.inp(c)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
