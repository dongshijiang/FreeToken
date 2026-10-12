"""TP sharding of the standard OCP MXFP4 (ds_fp4) expert kernel.

TP shards only the intermediate axis: each rank keeps the gate and up rows of its
own shard (the fused bank is [gate(0:I); up(I:2I)] per expert) and the matching
down byte / scale columns. Verified here as:

  - ``unusable_reason`` accepts a TP2 geometry and rejects one whose local
    intermediate breaks the byte/scale group alignment;
  - ``pack`` rank slices reconstruct the full bank byte-exactly;
  - the kernel math: rank0 + rank1 routed outputs equal the full-bank output
    within bf16 rounding (the gate_up / swiglu / fp8-requant chain is exact per
    rank, the down GEMV differs only in fp32 accumulation order).
"""

import pytest
import torch

from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

E, H, I, TOP_K = 64, 1024, 2048, 6
LIMIT = 7.0
# TP2 local intermediate = I // 2 = 1024 (the DSV4 production value: moe_inter_dim 2048).


def _cfg(rank, size):
    return MoEConfig(
        num_experts=E, hidden=H, intermediate=I, top_k=TOP_K,
        tp_rank=rank, tp_size=size, strategy="offload",
    )


def _pieces(device, expert_n=E):
    g = torch.Generator(device="cpu").manual_seed(0)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g).to(device)

    return {
        "gate": u8(expert_n, I, H // 2),
        "up": u8(expert_n, I, H // 2),
        "gate_scale": u8(expert_n, I, H // 32, low=118, high=134),
        "up_scale": u8(expert_n, I, H // 32, low=118, high=134),
        "down": u8(expert_n, H, I // 2),
        "down_scale": u8(expert_n, H, I // 32, low=118, high=134),
    }


def _packed(cfg):
    """Full-size (TP1) banks for ``cfg``'s geometry."""
    kernel = TritonMxfp4MoEKernel()
    device = "cpu"
    out = {
        role: torch.empty(E, *spec.shape, dtype=spec.dtype)
        for role, spec in kernel.layout(cfg).items()
    }
    kernel.pack(_pieces(device), cfg, out)
    return out


# ---------------------------------------------------------------------------
# Kernel acceptance / layout.
# ---------------------------------------------------------------------------

def test_unusable_reason_accepts_tp2():
    assert TritonMxfp4MoEKernel().unusable_reason(_cfg(0, 2)) is None
    assert TritonMxfp4MoEKernel().unusable_reason(_cfg(1, 2)) is None


def test_unusable_reason_rejects_unaligned_local_intermediate():
    # 300 % 2 == 0 but local 150 breaks the 256-alignment of byte/scale groups.
    cfg = MoEConfig(
        num_experts=E, hidden=H, intermediate=300, top_k=TOP_K,
        tp_rank=0, tp_size=2, strategy="offload",
    )
    reason = TritonMxfp4MoEKernel().unusable_reason(cfg)
    assert reason is not None and "multiple of 256" in reason


def test_layout_uses_local_intermediate():
    i0 = I // 2
    out = TritonMxfp4MoEKernel().layout(_cfg(0, 2))
    assert out["gate_up"].shape == (2 * i0, H // 2)
    assert out["gate_up_scale"].shape == (2 * i0, H // 32)
    assert out["down"].shape == (H, i0 // 2)
    assert out["down_scale"].shape == (H, i0 // 32)


# ---------------------------------------------------------------------------
# Packing: rank slices reconstruct the full banks byte-exactly.
# ---------------------------------------------------------------------------

def test_pack_tp2_reconstructs_full_banks():
    kernel = TritonMxfp4MoEKernel()
    full = _packed(_cfg(0, 1))
    rank_banks = {}
    for r in (0, 1):
        out = {
            role: torch.empty(E, *spec.shape, dtype=spec.dtype)
            for role, spec in kernel.layout(_cfg(r, 2)).items()
        }
        kernel.pack(_pieces("cpu"), _cfg(r, 2), out)
        rank_banks[r] = out

    i0 = I // 2
    # rank bank = [gate_r(i0); up_r(i0)] -> full = [gate0+gate1; up0+up1].
    def recon(part):
        return torch.cat([
            torch.cat([part[0][:, :i0], part[1][:, :i0]], dim=1),
            torch.cat([part[0][:, i0:], part[1][:, i0:]], dim=1),
        ], dim=1)

    assert torch.equal(full["gate_up"], recon([rank_banks[0]["gate_up"], rank_banks[1]["gate_up"]]))
    assert torch.equal(
        full["gate_up_scale"],
        recon([rank_banks[0]["gate_up_scale"], rank_banks[1]["gate_up_scale"]]),
    )
    assert torch.equal(full["down"][..., :i0 // 2], rank_banks[0]["down"])
    assert torch.equal(full["down"][..., i0 // 2:], rank_banks[1]["down"])
    assert torch.equal(full["down_scale"][..., :i0 // 32], rank_banks[0]["down_scale"])
    assert torch.equal(full["down_scale"][..., i0 // 32:], rank_banks[1]["down_scale"])


# ---------------------------------------------------------------------------
# Kernel math: rank0 + rank1 == full (CUDA).
# ---------------------------------------------------------------------------

def _rank_banks(device):
    """Full banks plus the per-rank (gate|up-concatenated, down-sliced) banks."""
    g = torch.Generator(device="cpu").manual_seed(0)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g).to(device)

    i0 = I // 2
    gup = u8(E, 2 * I, H // 2)   # [gate(0:I); up(I:2I)]
    gus = u8(E, 2 * I, H // 32, low=118, high=134)
    dn = u8(E, H, I // 2)
    dns = u8(E, H, I // 32, low=118, high=134)

    def rank_of(rank):
        def gu(t):
            return torch.cat(
                [t[:, rank * i0:(rank + 1) * i0], t[:, I + rank * i0:I + (rank + 1) * i0]],
                dim=1,
            ).contiguous()
        return (
            gu(gup), gu(gus),
            dn[..., rank * i0 // 2:(rank + 1) * i0 // 2].contiguous(),
            dns[..., rank * i0 // 32:(rank + 1) * i0 // 32].contiguous(),
        )

    return gup, gus, dn, dns, rank_of


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_routed_experts_fp4_tp2_sum_matches_full():
    pytest.importorskip("triton")
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    device = "cuda"
    gup, gus, dn, dns, rank_of = _rank_banks(device)

    g = torch.Generator(device="cpu").manual_seed(1)
    x = (torch.randn(16, H, dtype=torch.bfloat16, generator=g) * 0.1).to(device)
    slots = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(16)]).to(
        device=device, dtype=torch.int32
    ).contiguous()
    w = torch.rand(16, TOP_K, generator=g)
    w = (w / w.sum(-1, keepdim=True)).float().to(device).contiguous()

    ref = routed_experts_fp4(x, slots, w, gup, gus, dn, dns, LIMIT)
    r0 = routed_experts_fp4(x, slots, w, *rank_of(0), LIMIT)
    r1 = routed_experts_fp4(x, slots, w, *rank_of(1), LIMIT)
    out = r0 + r1
    # The gate_up / swiglu / fp8-requant chain is bit-exact per rank and the rank
    # down GEMV equals the zero-padded full K-slice exactly, so the only divergence
    # is bf16 rounding of the per-route outputs: TP1 rounds each route once, TP2
    # rounds each route per rank AND each rank's half-sum before the external add.
    # When the route contributions cancel, the partials (and the rounded values)
    # are orders of magnitude larger than the result, so a per-element or relative
    # tolerance would be wrong (empirically the diff is exactly 2 ulps at the
    # tensor's max magnitude); the bound is a few bf16 ulps of the GLOBAL max plus
    # an absolute floor -- loose enough for worst-case (top_k x 0.5 ulp) rounding,
    # tight enough to catch a genuinely wrong shard (which errors by O(max)).
    diff = (out.float() - ref.float()).abs()
    ulp = lambda t: (
        torch.exp2(torch.floor(torch.log2(t.abs().clamp_min(2**-126))) - 7)
    )
    mmax = max(ref.abs().max().item(), r0.abs().max().item(), r1.abs().max().item())
    tol = 4 * ulp(torch.tensor(mmax)) + 1.0
    assert (diff <= tol).all(), (diff.max().item(), diff.argmax().item(), tol.item())
    assert (out == ref).float().mean().item() > 0.4
