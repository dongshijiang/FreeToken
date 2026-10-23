"""fp8 e4m3 experts with 128x128 block scales."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import FP8_BLOCK as BLOCK, QuantKind
from .base import BankSpec, ExpertView, fused_piece, gated_epilogue_reason, limit_or_inf, MoEConfig, MoEKernel, MoEMethod
from freetoken.utils import align_ceil

FP8 = torch.float8_e4m3fn


def _pad_scale(rows: int, cols: int) -> int:
    from freetoken.kernel.aot_models import fp8_block_scale_pad

    return fp8_block_scale_pad(rows, cols)


def _bank_intermediate(cfg: MoEConfig) -> int:
    """The local bank's intermediate width.

    TP shards the intermediate axis, and the block-fp8 scale tiles are indexed in 128-row
    blocks. A shard that is not block-aligned (e.g. 1536/8 = 192) is widened to the next
    block boundary; the extra rows are zero-padded, so a rank whose shard falls past the
    model's real intermediate carries all zeros and contributes nothing but participates in
    the all-reduce. The pack below fills every rank's shard from the same block-aligned
    slices of the original, so ranks 0..k-1 hold the real data in order.
    """
    if cfg.tp_size <= 1:
        return cfg.intermediate
    return align_ceil(cfg.local_intermediate, BLOCK)


class TritonFp8BlockMoEKernel(MoEKernel):
    name = "triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=True, tp_ok=True, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        if cfg.strategy == "resident" and cfg.tp_size > 1:
            # MoELayer all-reduces resident outputs, which would sum the full-width copies
            # every rank holds instead of per-shard partials; the offload cache is the
            # TP path (banks carry the shard, the layer's _maybe_all_reduce sums it).
            return "fp8 block resident experts support TP=1 only; use --moe-strategy offload"
        if cfg.tp_size > 1:
            reason = self._tp_geometry_reason(cfg)
            if reason:
                return f"fp8 block MoE kernel: {reason}"
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"fp8 block MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "fp8 block MoE kernel cannot apply the router weight on the input"
        return None

    @staticmethod
    def _tp_geometry_reason(cfg: MoEConfig) -> str | None:
        if cfg.intermediate % cfg.tp_size:
            return f"intermediate {cfg.intermediate} is not divisible by TP size {cfg.tp_size}"
        li = cfg.local_intermediate
        pad = align_ceil(li, BLOCK)
        if pad > 2 * li:
            return (
                f"local intermediate {li} would pad to {pad} (>2x) after the 128-row "
                f"scale-block alignment; use a TP size whose shard is at least {BLOCK // 2}"
            )
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h, b = _bank_intermediate(cfg), cfg.hidden, BLOCK
        return {
            "gate_up": BankSpec((2 * i, h), FP8),
            "gate_up_scale": BankSpec((2 * i // b, _pad_scale(2 * i // b, h // b)), torch.bfloat16),
            "down": BankSpec((h, i), FP8),
            "down_scale": BankSpec((h // b, _pad_scale(h // b, i // b)), torch.bfloat16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        if cfg.tp_size > 1:
            # TP: the pieces are already this rank's slice of the intermediate (width
            # ``take`` = how much of the real intermediate the rank's bank holds, 0 for
            # ranks past it), produced by the weight reader under the same geometry.
            # The rank's bank is span-wide (span = align_ceil(local_intermediate, BLOCK)),
            # so the slice sits at the start and the remainder stays zero — the all-reduce
            # sums the replicated dense path and the per-rank expert partials.
            span, b, i = _bank_intermediate(cfg), BLOCK, cfg.intermediate
            gu = fused_piece(pieces, "gate_up")  # [e, 2*take, h]
            gs = fused_piece(pieces, "gate_up_scale")  # [e, 2*n, h//b]
            dn = pieces["down"]  # [e, h, take]
            ds = pieces["down_scale"]  # [e, h//b, n]
            gu_out, gs_out = out["gate_up"], out["gate_up_scale"]
            dn_out, ds_out = out["down"], out["down_scale"]
            gu_out.zero_()
            gs_out.zero_()
            dn_out.zero_()
            ds_out.zero_()
            t = gu.shape[1] // 2
            if t:
                n = gs.shape[1] // 2
                gu_out[:, :t].copy_(gu[:, :t])
                gu_out[:, span:span + t].copy_(gu[:, t:])
                gs_out[:, :n, : gs.shape[2]].copy_(gs[:, :n])
                gs_out[:, span // b: span // b + n, : gs.shape[2]].copy_(gs[:, n:])
                dn_out[:, :, :t].copy_(dn)
                ds_out[:, :, :n].copy_(ds)
            return {}
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["down"].copy_(pieces["down"])
        # the scale banks carry 16-byte padding on the last dim
        gs = fused_piece(pieces, "gate_up_scale")
        out["gate_up_scale"][:, :, : gs.shape[2]].copy_(gs)
        ds = pieces["down_scale"]
        out["down_scale"][:, :, : ds.shape[2]].copy_(ds)
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block, fused_experts_fp8_block

        t = view.tensors
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            n = view.n if view.n is not None else layer.num_experts
            return fused_experts_fp8_block(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, n, layer.activation, layer.apply_router_weight_on_input, alpha, limit)
        return fused_experts_decode_fp8_block(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input, alpha, limit)


@register_method(QuantKind.FP8_BLOCK, LayerKind.MOE)
class Fp8BlockMoEMethod(MoEMethod):
    candidates = (TritonFp8BlockMoEKernel,)

    def create_weights(self, layer) -> None:
        g = self.cfg
        e, i, h, b = g.num_experts, g.intermediate, g.hidden, BLOCK
        layer.gate_up_proj = torch.empty(e, 2 * i, h, dtype=FP8)
        layer.gate_up_scale_inv = torch.empty(e, 2 * i // b, h // b, dtype=torch.bfloat16)
        layer.down_proj = torch.empty(e, h, i, dtype=FP8)
        layer.down_scale_inv = torch.empty(e, h // b, i // b, dtype=torch.bfloat16)

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({
            "gate_up": layer.gate_up_proj, "gate_up_scale": layer.gate_up_scale_inv,
            "down": layer.down_proj, "down_scale": layer.down_scale_inv,
        })
