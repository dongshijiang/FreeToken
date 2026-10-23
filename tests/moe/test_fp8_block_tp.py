"""TP sharding of the fp8 block (128x128 scale) expert kernel.

TP shards the intermediate axis in block-aligned bank order: the real intermediate is
laminated over a padded world of ``tp_size`` 128-aligned shards, so rank ``r`` keeps rows
``[r*span, (r+1)*span)`` of the original at the start of its bank and zeros after (a rank
that falls past the real intermediate contributes nothing to the all-reduce). Under TP the
weight reader hands ``pack`` rank-local pieces (each piece is already the rank's slice), and
``pack`` lays them at the start of the local bank. Verified as:

  - ``unusable_reason`` accepts the TP8 geometry of the production 1536-wide intermediate
    (local 192 padded to 256) and rejects a shard that pads past 2x / does not divide;
  - resident experts stay TP=1;
  - rank-local pieces pack into zero-padded banks, and the concatenated rank banks
    reconstruct the full gates byte-exactly.
"""

import torch

from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel

E, H, I, TOP_K, B = 4, 4096, 1536, 8, 128  # qwen3-235b: moe_inter 1536, hidden 4096


def _cfg(rank, size, *, intermediate=I, strategy="offload"):
    return MoEConfig(
        num_experts=E, hidden=H, intermediate=intermediate, top_k=TOP_K,
        tp_rank=rank, tp_size=size, strategy=strategy,
    )


def _aligned(rows):
    return ((rows + B - 1) // B) * B


def _pieces():
    g = torch.Generator(device="cpu").manual_seed(0)

    def fp8(*shape):
        return torch.randint(0, 8, shape, dtype=torch.uint8, generator=g).view(torch.float8_e4m3fn)

    def bf(*shape):
        return (torch.randn(shape, generator=g) * 0.1).to(torch.bfloat16)

    return {
        "gate": fp8(E, I, H),
        "up": fp8(E, I, H),
        "gate_scale": bf(E, I // B, H // B),
        "up_scale": bf(E, I // B, H // B),
        "down": fp8(E, H, I),
        "down_scale": bf(E, H // B, I // B),
    }


def _rank_pieces(rank, size, *, I=I):
    """The rank-local pieces the qwen3_moe reader slices out of the full checkpoint tensors."""
    span = _aligned(I // size)
    lo = rank * span
    take = min(max(I - lo, 0), span)
    s0, s1 = lo // B, (lo + take + B - 1) // B
    f = _pieces()
    return {
        "gate": f["gate"][:, lo:lo + take],
        "up": f["up"][:, lo:lo + take],
        "gate_scale": f["gate_scale"][:, s0:s1],
        "up_scale": f["up_scale"][:, s0:s1],
        "down": f["down"][:, :, lo:lo + take],
        "down_scale": f["down_scale"][:, :, s0:s1],
    }


# ---------------------------------------------------------------------------
# Kernel acceptance / layout.
# ---------------------------------------------------------------------------

def test_unusable_reason_accepts_tp8():
    k = TritonFp8BlockMoEKernel()
    # 1536/8 = 192 -> padded to 256 (<= 2x), block-aligned: accepted.
    for r in (0, 4, 7):
        assert k.unusable_reason(_cfg(r, 8)) is None
    # TP2 / TP4: local 768 / 384 are already 128-aligned: accepted.
    assert k.unusable_reason(_cfg(0, 2)) is None
    assert k.unusable_reason(_cfg(1, 4)) is None


def test_unusable_reason_rejects_bad_geometry():
    k = TritonFp8BlockMoEKernel()
    reason = k.unusable_reason(_cfg(0, 5))
    assert reason is not None and "not divisible" in reason
    # local 48 -> padded 128 is > 2x: refused by _tp_geometry_reason.
    reason = k.unusable_reason(_cfg(0, 8, intermediate=384))
    assert reason is not None and "would pad" in reason
    # resident experts stay TP=1 (the layer all-reduce sums the replicated copies).
    reason = k.unusable_reason(_cfg(0, 2, strategy="resident"))
    assert reason is not None and "resident" in reason


def test_layout_pads_bank_intermediate():
    k = TritonFp8BlockMoEKernel()
    # TP8: bank intermediate 256 (local 192 padded), TP1: 1536.
    assert k.layout(_cfg(0, 8))["gate_up"].shape == (2 * 256, H)
    assert k.layout(_cfg(0, 8))["gate_up_scale"].shape == (2 * 256 // B, H // B)
    assert k.layout(_cfg(0, 8))["down"].shape == (H, 256)
    assert k.layout(_cfg(0, 1))["gate_up"].shape == (2 * I, H)


# ---------------------------------------------------------------------------
# Packing: rank-local pieces + zero padding reconstruct the full banks.
# ---------------------------------------------------------------------------

def _pack(cfg, pieces=None):
    k = TritonFp8BlockMoEKernel()
    out = {
        role: torch.empty(E, *spec.shape, dtype=spec.dtype)
        for role, spec in k.layout(cfg).items()
    }
    k.pack(_pieces() if pieces is None else pieces, cfg, out)
    return out


def _is_zero(t):
    return t.to(torch.float32).abs().max().item() == 0.0


def test_pack_tp8_rank_local_reconstructs_full_banks():
    span = 256  # padded bank intermediate for TP8
    f = _pack(_cfg(0, 1))
    ranks = [_pack(_cfg(r, 8), _rank_pieces(r, 8)) for r in range(8)]

    def cat(render, dim):
        return torch.cat([render(r) for r in ranks], dim=dim)

    cat_gate = cat(lambda rb: rb["gate_up"][:, :span], dim=1)
    cat_up = cat(lambda rb: rb["gate_up"][:, span:], dim=1)
    assert torch.equal(f["gate_up"][:, :I], cat_gate[:, :I])
    assert torch.equal(f["gate_up"][:, I:], cat_up[:, :I])
    assert _is_zero(cat_gate[:, I:]) and _is_zero(cat_up[:, I:])

    cat_gs = cat(lambda rb: rb["gate_up_scale"][:, : span // B], dim=1)
    cat_us = cat(lambda rb: rb["gate_up_scale"][:, span // B :], dim=1)
    assert torch.equal(f["gate_up_scale"][:, : I // B], cat_gs[:, : I // B])
    assert torch.equal(f["gate_up_scale"][:, I // B :], cat_us[:, : I // B])
    assert _is_zero(cat_gs[:, I // B :]) and _is_zero(cat_us[:, I // B :])

    cat_dn = cat(lambda rb: rb["down"], dim=2)
    assert torch.equal(f["down"][:, :, :I], cat_dn[:, :, :I])
    assert _is_zero(cat_dn[:, :, I:])

    cat_ds = cat(lambda rb: rb["down_scale"], dim=2)
    assert torch.equal(f["down_scale"][:, :, : I // B], cat_ds[:, :, : I // B])
    assert _is_zero(cat_ds[:, :, I // B :])


def test_pack_tp8_ranks_past_intermediate_are_zero():
    """TP8 ranks 6/7 hold no real rows (1536 spans ranks 0..5): their banks stay all zero."""
    for r in (6, 7):
        bank = _pack(_cfg(r, 8), _rank_pieces(r, 8))
        assert _is_zero(bank["gate_up"]) and _is_zero(bank["gate_up_scale"])
        assert _is_zero(bank["down"]) and _is_zero(bank["down_scale"])


def test_pack_tp2_no_padding_byte_exact():
    """TP2/TP4 shards are already 128-aligned: the rank banks concatenate back exactly."""
    k = TritonFp8BlockMoEKernel()
    span = I // 2
    f = _pack(_cfg(0, 1))
    r0, r1 = _pack(_cfg(0, 2), _rank_pieces(0, 2)), _pack(_cfg(1, 2), _rank_pieces(1, 2))
    assert torch.equal(f["gate_up"][:, :span], r0["gate_up"][:, :span])
    assert torch.equal(f["gate_up"][:, span:I], r1["gate_up"][:, :span])
    assert torch.equal(f["gate_up"][:, I:I + span], r0["gate_up"][:, span:])
    assert torch.equal(f["gate_up"][:, I + span:], r1["gate_up"][:, span:])
    # gate scale rows [0:12] = r0 gate + r1 gate; up scale rows [12:24] likewise.
    assert torch.equal(f["gate_up_scale"][:, : span // B], r0["gate_up_scale"][:, : span // B])
    assert torch.equal(
        f["gate_up_scale"][:, span // B : 2 * span // B], r1["gate_up_scale"][:, : span // B]
    )
    assert torch.equal(
        f["gate_up_scale"][:, I // B : I // B + span // B], r0["gate_up_scale"][:, span // B :]
    )
    assert torch.equal(
        f["gate_up_scale"][:, I // B + span // B :], r1["gate_up_scale"][:, span // B :]
    )
    assert torch.equal(f["down"][:, :, :span], r0["down"])
    assert torch.equal(f["down"][:, :, span:], r1["down"])
    assert torch.equal(f["down_scale"][:, :, : span // B], r0["down_scale"])
    assert torch.equal(f["down_scale"][:, :, span // B :], r1["down_scale"])
