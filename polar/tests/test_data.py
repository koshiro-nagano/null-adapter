"""Features, projector initialisation and sampling (CPU; no dataset needed except the last test)."""
import itertools
import math

import numpy as np
import pytest
import torch

from polar.src.data import CACHE, SIMCMF_STATS, DrawSampler, click, features, regions
from polar.src.projectors import GroupResidualProjector, ResidualPredictor


def test_zju_p_is_normalised_stokes_and_matches_simcmf_dolp():
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 256, (1, 12, 1024, 1024), generator=g, dtype=torch.uint8)  # 1024: resize is identity
    f = features("zju", raw)
    x = raw.float()
    I1, I2, I3, I4 = x.view(1, 4, 3, 1024, 1024).unbind(1)
    S0 = 0.5 * (I1 + I2 + I3 + I4) + 1e-4
    ref = torch.cat([(I1 - I3) / S0, (I2 - I4) / S0], 1)
    assert torch.allclose(f["p"], ref, atol=1e-5)
    assert torch.allclose(f["c"], S0 / 2, atol=1e-3)
    # SimCMF's DoLP (channels 6:9 before normalisation) is |p| where it is not clipped at 1
    mu, sd = [torch.tensor(v).view(1, -1, 1, 1) for v in SIMCMF_STATS["zju"]]
    dolp_sim = f["simcmf"] * sd + mu
    dolp_sim = dolp_sim[:, 6:9]
    norm_p = torch.sqrt(f["p"][:, :3] ** 2 + f["p"][:, 3:] ** 2)
    ok = dolp_sim < 0.99
    assert torch.allclose(dolp_sim[ok], norm_p[ok], atol=1e-4)


def test_pgsnet_p_has_norm_dolp():
    raw = torch.zeros(1, 9, 8, 8, dtype=torch.uint8)
    raw[:, 3:6] = 64   # AoLP = 64/255 * pi
    raw[:, 6:9] = 128  # DoLP = 128/255
    p = features("pgsnet", raw)["p"]
    rho, th = 128 / 255, 64 / 255 * math.pi
    assert torch.allclose(p[:, :3], torch.full_like(p[:, :3], rho * math.cos(2 * th)), atol=1e-5)
    assert torch.allclose(torch.sqrt(p[:, :3] ** 2 + p[:, 3:] ** 2), torch.full_like(p[:, :3], rho), atol=1e-5)


def test_residual_projector_starts_as_rgb_only():
    torch.manual_seed(0)
    pe = torch.nn.Conv2d(3, 768, 16, stride=16)
    func = lambda c: pe(c).permute(0, 2, 3, 1)
    x = torch.randn(2, 9, 64, 64)
    for k in (0, 6):
        proj = GroupResidualProjector(k, pe)
        out, _ = proj(x[:, : 3 + k], func)
        assert torch.equal(out, func(x[:, :3]))
    proj = GroupResidualProjector(6, pe)
    torch.nn.init.normal_(proj.fc.weight)
    assert not torch.allclose(proj(x, func)[0], func(x[:, :3]))
    assert ResidualPredictor()(torch.randn(1, 3, 32, 32)).shape == (1, 6, 32, 32)


def test_draw_sampler_reproducible_and_covers_epochs():
    a = list(itertools.islice(iter(DrawSampler(5, 3)), 15))
    b = list(itertools.islice(iter(DrawSampler(5, 3)), 15))
    assert a == b and [d for _, d in a] == list(range(15))
    for e in range(3):
        assert sorted(k for k, _ in a[5 * e: 5 * e + 5]) == list(range(5))


def test_click_inside_region_and_middle_rule():
    m = np.zeros((10, 10), np.uint8)
    m[2:5, 3:8] = 4
    ids, cnt = regions(m, 1)
    assert ids.tolist() == [4] and cnt.tolist() == [15]
    x, y = click(m, 4, None)
    ind = np.argwhere(m == 4)
    assert (y, x) == tuple(ind[len(ind) // 2])
    x, y = click(m, 4, np.random.default_rng(0))
    assert m[int(y), int(x)] == 4


@pytest.mark.skipif(not (CACHE / "zju" / "val_raw.npy").exists(), reason="ZJU cache not built")
def test_eval_set_targets_match_clicks():
    from polar.src.data import EvalSet
    ds = EvalSet("zju", "val")
    raw, pts, tg, ids, cnt, i = ds[0]
    assert raw.shape == (12, 1024, 1224) and tg.shape[1:] == (256, 256) and len(pts) == len(ids) > 0
    for (x, y), t in zip(pts.numpy(), tg.numpy()):
        assert t[int(y) // 4, int(x) // 4] in (0, 1)
