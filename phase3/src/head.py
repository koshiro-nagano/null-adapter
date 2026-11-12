"""Shared spatial head of Phase 3 (PLAN.md): z = P_f f + gamma * a(u), logits = out(z + T(z)), on a 128 x 128 map.

P_f is a 1x1 conv; it is applied on the 32 x 32 patch grid and the result is bilinearly upsampled to 128 x 128.
Both steps are affine with weights summing to 1, so this equals upsampling f first and then applying P_f.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MAP = 128          # side of the working map (stride 4)
STRIDE = 512 // MAP


class FusionHead(nn.Module):
    """d_f = None: no f branch (S1). d_u = None: no u branch (S0). gamma (zero-initialised) only when both exist.
    Modules are created in a fixed order (proj, body, out, adapter) so that, under the same torch seed, every
    method with an f branch starts with the same P_f, T and output layer."""

    def __init__(self, d_f: int | None, d_u: int | None, n_classes: int = 44, width: int = 128, adapter: int = 64):
        super().__init__()
        self.proj = nn.Conv2d(d_f, width, 1) if d_f else None
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(width, n_classes, 1)
        self.adapter = nn.Sequential(nn.Conv2d(d_u, adapter, 1), nn.GELU(), nn.Conv2d(adapter, width, 1)) if d_u else None
        self.gamma = nn.Parameter(torch.zeros(width)) if (d_f and d_u) else None

    def forward(self, f32: torch.Tensor | None, u: torch.Tensor | None) -> torch.Tensor:
        """f32: (B, d_f, 32, 32) standardised patch tokens; u: (B, d_u, 128, 128) standardised extra input."""
        if self.proj is None:
            z = self.adapter(u)
        else:
            z = F.interpolate(self.proj(f32), size=(MAP, MAP), mode="bilinear", align_corners=False)
            if self.adapter is not None:
                z = z + self.gamma.view(1, -1, 1, 1) * self.adapter(u)
        return self.out(z + self.body(z))


def read_at(maps: torch.Tensor, b: torch.Tensor, pix: torch.Tensor, side: int = 512) -> torch.Tensor:
    """Values of `maps` (B, C, h, w) bilinearly upsampled to side x side (align_corners=False), read at flat pixel
    indices pix of images b. Equals F.interpolate(maps, side)[b, :, pix // side, pix % side] without building it."""
    h = maps.shape[2]
    scale = side / h

    def axis(p):
        s = ((p.float() + 0.5) / scale - 0.5).clamp(min=0)
        i0 = s.floor().long().clamp(max=h - 1)
        return i0, (i0 + 1).clamp(max=h - 1), (s - i0).unsqueeze(1)

    y0, y1, wy = axis(pix // side)
    x0, x1, wx = axis(pix % side)
    q = lambda yy, xx: maps[b, :, yy, xx]
    return (1 - wy) * ((1 - wx) * q(y0, x0) + wx * q(y0, x1)) + wy * ((1 - wx) * q(y1, x0) + wx * q(y1, x1))


def neighbourhood(c: torch.Tensor, r: int = 2) -> torch.Tensor:
    """(B, 3, h, w) -> (B, 3 * (2r+1)^2, h, w): each cell's (2r+1)^2 neighbourhood, edges replicated,
    ordered channel-major then row then column (F.unfold order)."""
    B, C, h, w = c.shape
    k = 2 * r + 1
    p = F.pad(c, (r, r, r, r), mode="replicate")
    return F.unfold(p, k).view(B, C * k * k, h, w)
