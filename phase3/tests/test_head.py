"""Phase 3 head, pixel read-out, neighbourhoods and pooling."""
import numpy as np
import torch
from torch.nn import functional as F

from phase2.src.render import build_operator
from phase3.scripts.prepare import pool
from phase3.src.head import FusionHead, neighbourhood, read_at
from phase3.src.run import NB, Components


def test_read_at_matches_interpolate():
    g = torch.Generator().manual_seed(0)
    maps = torch.randn(3, 5, 128, 128, generator=g, dtype=torch.float64)
    up = F.interpolate(maps, size=(512, 512), mode="bilinear", align_corners=False)
    b = torch.randint(0, 3, (4000,), generator=g)
    pix = torch.cat([torch.randint(0, 512 * 512, (3990,), generator=g), torch.tensor([0, 511, 511 * 512, 512 * 512 - 1, 1, 2, 3, 4, 5, 6])])
    torch.testing.assert_close(read_at(maps, b, pix), up[b, :, pix // 512, pix % 512])


def test_zero_gate_equals_f_only_head():
    f32, u = torch.randn(2, 768, 32, 32), torch.randn(2, 204, 128, 128)
    torch.manual_seed(7); s0 = FusionHead(768, None)
    torch.manual_seed(7); s5 = FusionHead(768, 204)
    torch.testing.assert_close(s0(f32, None), s5(f32, u))
    s5.gamma.data.fill_(0.1)
    assert not torch.allclose(s0(f32, None), s5(f32, u))


def test_projection_commutes_with_upsampling():
    torch.manual_seed(0)
    h = FusionHead(16, None, width=8)
    f32 = torch.randn(2, 16, 32, 32)
    a = F.interpolate(h.proj(f32), size=(128, 128), mode="bilinear", align_corners=False)
    b = h.proj(F.interpolate(f32, size=(128, 128), mode="bilinear", align_corners=False))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_neighbourhood_gather_matches_unfold():
    c = torch.randn(3, 3, 128, 128, dtype=torch.float64)
    full = neighbourhood(c, NB)                                  # (3, 75, 128, 128), used on maps
    cpad = F.pad(c, (NB,) * 4, mode="replicate")
    img, cell = torch.tensor([0, 1, 2, 2, 0]), torch.tensor([0, 127, 128 * 128 - 1, 5000, 130])
    got = Components._nb_raw(cpad, img, cell)                    # used on the unlabelled cells
    torch.testing.assert_close(got, full[img, :, cell // 128, cell % 128])
    assert got[0, 0 * 25 + 12] == c[0, 0, 0, 0] and got[0, 2 * 25 + 0] == c[0, 2, 0, 0]  # centre; replicated corner


def test_pool_and_pooled_rgb():
    rng = np.random.default_rng(0)
    cube = rng.random((512, 512, 204), dtype=np.float32)
    p = pool(cube)
    assert p.shape == (128, 128, 204)
    np.testing.assert_allclose(p[3, 5], cube[12:16, 20:24].mean((0, 1)), rtol=1e-6)
    A = build_operator()
    np.testing.assert_allclose(p[3, 5] @ A.T, (cube[12:16, 20:24].astype(np.float64) @ A.T).mean((0, 1)), rtol=1e-6)
