"""Projectors that replace SimCMF's `image_encoder.projector` (forward(input, func), func = frozen SAM patch embed).

  rgb     func(c)                                     RGB-only (+ LoRA)
  fuse_p  func(c) + sum_i w_i E_i(p_i)                RGB + polarisation, simple additive fusion
  null_p  func(c) + sum_i w_i E_i(r_i), r = p - m(c)  RGB + conditional polarisation residual (Null-Adapter)

E_i is a 1-channel 16 x 16 patch embedding initialised, as in SimCMF's novel branch, with the pretrained RGB patch
embedding summed over colour; w (Linear(k, 1), no bias) starts at zero, so all three start as the pretrained SAM on c.
fuse_p and null_p are the same module; only the extra input differs.
"""
from __future__ import annotations

import torch
from torch import nn


class GroupResidualProjector(nn.Module):
    def __init__(self, n_extra: int, patch_embed: nn.Conv2d):
        super().__init__()
        self.n_extra = n_extra
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

    def forward(self, input: torch.Tensor, func):
        out = func(input[:, :3])
        if self.n_extra:
            u = input[:, 3:].unsqueeze(2)
            novel = torch.stack([self.group_patch_embedding[i](u[:, i]) for i in range(self.n_extra)], -1)  # B,C,h,w,k
            out = out + self.fc(novel.permute(0, 2, 3, 1, 4))[..., 0]
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
