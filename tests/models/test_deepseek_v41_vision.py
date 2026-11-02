"""DeepSeek-V4.1 vision tower: span layout, aligner padding and host streaming on a tiny tower."""

from __future__ import annotations

import pytest
import torch

from freetoken.message import MMItem
from freetoken.models.deepseek_v41.args import DeepseekV41VisionArgs, load_vision_args
from freetoken.models.deepseek_v41.vision import (
    VisionAttention,
    VisionTower,
    _apply_vision_rotary,
    _vision_cos_sin,
)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _single_rank():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _tiny_args():
    return DeepseekV41VisionArgs(
        num_hidden_layers=2, hidden_size=64, num_attention_heads=4, intermediate_size=128,
        patch_size=4, rope_theta=10000.0, downsample_ratio=2,
    )


def _build(va):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = VisionTower(va, text_dim=48)
    finally:
        torch.set_default_dtype(torch.float32)
    for t in tower.state_dict().values():
        t.normal_(0, 0.02)
    return tower


def _item(n_h, n_w, llm_h, llm_w):
    return MMItem(
        modality="image", hash=1, pad_value=1, offsets=[],
        feature=torch.randn(n_h * n_w, 3, 4, 4, dtype=torch.bfloat16, device="cuda"),
        model_specific_data={"vit_grid": (n_h, n_w), "llm_grid": (llm_h, llm_w)},
    )


@_cuda
def test_encode_closes_each_llm_row_with_the_newline_between_start_and_end():
    tower = _build(_tiny_args())
    item = _item(2, 3, 1, 2)  # ratio 2: a 2x3 patch grid merges into 1x2 llm tokens
    span = tower.encode(item)
    assert span.shape == (5, 48) and span.dtype == torch.bfloat16
    feats = tower.aligner.forward(tower.forward(item.feature, 2, 3), 2, 3)
    assert torch.equal(span[0], tower.image_start) and torch.equal(span[-1], tower.image_end)
    # START + llm_h rows of (llm_w features + NEWLINE) + END -> START, f0, f1, NL, END
    assert torch.equal(span[1], feats[0]) and torch.equal(span[2], feats[1])
    assert torch.equal(span[3], tower.image_newline)


@_cuda
def test_aligner_zero_pads_a_grid_that_is_not_a_ratio_multiple():
    tower = _build(_tiny_args())
    out = tower.aligner.forward(torch.randn(27, 64, dtype=torch.bfloat16, device="cuda"), 3, 9)
    # 3x9 pads up to 4x10 -> 2x5 = 10 llm rows, none dropped
    assert out.shape == (10, 48)
    item = _item(3, 9, 2, 5)
    assert tower.encode(item).shape == (2 + 2 * 6, 48)


@_cuda
def test_host_streamed_weights_compute_the_same_output():
    tower = _build(_tiny_args())
    item = _item(2, 3, 1, 2)
    resident = tower.encode(item)
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    # block weights stream from the pinned bank; delimiters and the aligner stay resident
    assert tower.blocks.op_list[0].attn.wqkv.weight.device.type == "cpu"
    assert tower.image_start.device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.encode(item), resident)
    tower.place_weights("gpu")
    assert tower.blocks.op_list[0].attn.wqkv.weight.device.type == "cuda"
    assert torch.equal(tower.encode(item), resident)


def test_load_vision_args_drops_unknown_keys():
    va = load_vision_args({"hidden_size": 64, "num_attention_heads": 4, "torch_dtype": "bfloat16", "unused": 1})
    assert va.hidden_size == 64 and va.num_attention_heads == 4
    assert va.num_hidden_layers == 32 and va.patch_size == 14  # released defaults for the omitted fields


def test_vision_attention_matches_the_math_reference():
    # the 4D SDPA input keeps the fused kernels eligible; this pins its numbers to
    # an explicit softmax(QK^T/sqrt(d))V reference
    torch.manual_seed(3)
    va = _tiny_args()
    attn = VisionAttention(va)
    for t in attn.state_dict().values():
        t.normal_(0, 0.05)
    n, h, d = 12, va.num_attention_heads, va.head_dim
    x = torch.randn(n, va.hidden_size, dtype=torch.bfloat16)
    cos, sin = _vision_cos_sin(3, 4, va.hidden_size // h // 2, va.rope_theta, x.device)
    out = attn.forward(x, cos, sin)

    q, k, v = (t.view(n, h, d).float() for t in attn.wqkv.forward(x).chunk(3, dim=-1))
    q = _apply_vision_rotary(q, cos, sin).float()
    k = _apply_vision_rotary(k, cos, sin).float()
    att = torch.einsum("qhd,khd->hqk", q, k) * d ** -0.5
    ref = torch.einsum("hqk,khd->qhd", att.softmax(-1), v).reshape(n, -1)
    ref = attn.wo.forward(ref.to(torch.bfloat16))
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@_cuda
def test_large_grid_forward_stays_flash_and_finite():
    # a real large image (released towers see grids well past 4000 patches) must ride
    # the fused 4D attention, not fall back to a materialized N^2 math matrix
    tower = _build(_tiny_args())
    n_h, n_w = 54, 76  # 4104 patches > 4000
    item = _item(n_h, n_w, n_h // 2, n_w // 2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    span = tower.encode(item)
    torch.cuda.synchronize()
    assert span.shape == (2 + (n_h // 2) * (n_w // 2 + 1), 48)
    assert torch.isfinite(span.float()).all()
    # one N^2 matrix for the 4 heads alone would be ~135 MB; the fused path needs none
    n2_bytes = 4 * n_h * n_w * n_h * n_w * span.element_size()
    assert torch.cuda.max_memory_allocated() < n2_bytes // 2
