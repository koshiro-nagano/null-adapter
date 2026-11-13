"""The SDPA rewrite of SimCMF's LoRA attention (global blocks) is at least as close to the fp32 original as the
original itself run under bf16 autocast (the training setting), for values and input gradients."""
import pytest
import torch

from polar.src import bridge
from models.sam_lora.modeling import image_encoder as ie

ORIG = ie.Attention.forward
bridge.use_sdpa_attention()


def _run(fn, a, x, g, bf16):
    xx = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        y = fn(a, xx)
    (y.float() * g).sum().backward()
    return y.float().detach(), xx.grad


@pytest.mark.parametrize("H,B", [(14, 3), (64, 1)])
def test_sdpa_attention_matches(H, B):
    torch.manual_seed(0)
    a = ie.Attention(768, num_heads=12, qkv_bias=True, use_rel_pos=True, input_size=(H, H)).cuda()
    for p in a.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x, g = torch.randn(B, H, H, 768, device="cuda"), torch.randn(B, H, H, 768, device="cuda")
    yr, gr = _run(ORIG, a, x, g, False)
    rel = lambda u, v: float((u - v).norm() / v.norm())
    yo, go = _run(ORIG, a, x, g, True)
    yn, gn = _run(ie.Attention.forward, a, x, g, True)
    assert rel(yn, yr) <= max(1.2 * rel(yo, yr), 1e-6)
    assert rel(gn, gr) <= max(1.2 * rel(go, gr), 1e-6)
