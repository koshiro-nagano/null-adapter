"""Access to the SimCMF code base (github.com/mt-cly/SimCMF @ 0cdc5c1), used unchanged for SAM-B + LoRA,
its projector, loss and click/IoU helpers. The repository has no license file, so it is not vendored here:
clone it next to this repository (../ext/SimCMF) or set SIMCMF_ROOT."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get("SIMCMF_ROOT", Path(__file__).resolve().parents[3] / "ext" / "SimCMF"))
if not (ROOT / "models" / "projector.py").exists():
    raise FileNotFoundError(f"SimCMF not found at {ROOT}; set SIMCMF_ROOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conf import settings  # noqa: E402
from utils import get_network, set_trainable_params, random_click  # noqa: E402

SAM_CKPT = str(ROOT / "checkpoint" / "sam" / "sam_vit_b_01ec64.pth")


def build_sam_lora(proj_type: str, modality: str = "zju-rgbp"):
    """SAM-B with SimCMF's LoRA (r = settings.LORA_R on q and v) and the requested SimCMF projector.
    Returns (net, names of trainable parameters): LoRA + image_encoder.projector, as in SimCMF."""
    args = SimpleNamespace(modality=modality, sam_ckpt=SAM_CKPT, uniformInit=False)
    net = get_network(args, "sam_lora", proj_type)
    return net, args


def _sdpa_forward(self, x):
    """SimCMF sam_lora Attention.forward with the same arithmetic, written for scaled_dot_product_attention.
    The decomposed relative-position bias q.Rh[qh, kh] + q.Rw[qw, kw] is exact as extra dot-product dimensions:
    q' = [q * scale, q.Rh, q.Rw], k' = [k, onehot(kh), onehot(kw)], softmax(q' k'^T) (scale 1)."""
    import torch
    from torch.nn import functional as F
    from models.sam_lora.modeling.image_encoder import get_rel_pos
    B, H, W, _ = x.shape
    if H * W < 1024:  # 14 x 14 windows: cheap, and the fp32 efficient kernel was wrong there on sm_120
        return _ORIG_FORWARD[0](self, x)
    nh = self.num_heads
    qkv = self.qkv(x).reshape(B, H * W, 3, nh, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * nh, H * W, -1).unbind(0)
    q = q + self.lora_q_b(self.lora_q_a(x)).reshape(B, H * W, nh, -1).permute(0, 2, 1, 3).reshape(B * nh, H * W, -1)
    v = v + self.lora_v_b(self.lora_v_a(x)).reshape(B, H * W, nh, -1).permute(0, 2, 1, 3).reshape(B * nh, H * W, -1)
    qs = q * self.scale
    if self.use_rel_pos:
        Rh, Rw = get_rel_pos(H, H, self.rel_pos_h), get_rel_pos(W, W, self.rel_pos_w)
        r_q = q.reshape(B * nh, H, W, -1)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B * nh, H * W, H)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B * nh, H * W, W)
        eye_h = torch.eye(H, device=x.device, dtype=k.dtype).repeat_interleave(W, 0)   # key (kh, kw) -> onehot(kh)
        eye_w = torch.eye(W, device=x.device, dtype=k.dtype).repeat(H, 1)              # -> onehot(kw)
        qs = torch.cat([qs, rel_h.to(qs.dtype), rel_w.to(qs.dtype)], -1)
        k = torch.cat([k, eye_h.expand(B * nh, -1, -1), eye_w.expand(B * nh, -1, -1)], -1)
    # one common head dim, a multiple of 32, for q, k and v (zero padding changes nothing); checked against the
    # original in tests/test_bridge.py
    d = -(-qs.shape[-1] // 32) * 32
    qs, k, vp = F.pad(qs, (0, d - qs.shape[-1])), F.pad(k, (0, d - k.shape[-1])), F.pad(v, (0, d - v.shape[-1]))
    # always bf16: the fp32 backward of the efficient kernel was inaccurate on sm_120 (6% relative gradient error)
    bf = torch.bfloat16
    o = F.scaled_dot_product_attention(qs.to(bf).unsqueeze(0), k.to(bf).unsqueeze(0), vp.to(bf).unsqueeze(0),
                                       scale=1.0)[0][..., :v.shape[-1]].to(v.dtype)
    return self.proj(o.view(B, nh, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1))


_ORIG_FORWARD = []


def use_sdpa_attention():
    """Global-attention blocks (64 x 64 tokens) through SDPA: same function, far less memory."""
    from models.sam_lora.modeling import image_encoder
    if not _ORIG_FORWARD:
        _ORIG_FORWARD.append(image_encoder.Attention.forward)
        image_encoder.Attention.forward = _sdpa_forward
