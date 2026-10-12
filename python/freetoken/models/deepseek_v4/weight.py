"""Weight loading for DeepSeek-V4-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed by the model's
    attribute paths (``model.`` + the checkpoint name). ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs + e8m0 per-32
    scales, no global) as per-expert pieces for the expert quant method's banks.
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch
from tqdm import tqdm

from freetoken.layers.quantization import QuantKind
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.utils import div_ceil

from .args import DeepseekV4Args, load_args


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize 128x128 block-scaled FP8 (e4m3) to bf16.

    scale is e8m0 exponent codes, ``value = 2^(code-127)`` (Triton FP8 GEMM convention).
    Used for ``wo_a`` to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def shard_rows(value: torch.Tensor, *, rank: int, size: int, rows: int) -> torch.Tensor:
    """This rank's first-dim shard of ``rows`` (the caller validates divisibility)."""
    if size <= 1:
        return value
    start = rank * rows
    return value[start:start + rows].clone()


def shard_cols(value: torch.Tensor, *, rank: int, size: int, cols: int) -> torch.Tensor:
    """This rank's last-dim shard of ``cols`` (the caller validates divisibility)."""
    if size <= 1:
        return value
    start = rank * cols
    return value[:, start:start + cols].clone()


def vocab_rows(value: torch.Tensor, *, rank: int, size: int, vocab_size: int) -> torch.Tensor:
    """The rank's ceil-partitioned vocab rows (matches ``VocabParallelEmbedding``)."""
    if size <= 1:
        return value
    n = div_ceil(vocab_size, size)
    start = rank * n
    return value[start:start + n].clone()


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed MXFP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-strategy offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    tp = get_tp_info()
    rank, size = tp.rank, tp.size
    if size > 1:
        # TP shards attention heads (wq_b / attn_sink), output groups (wo_a / wo_b) and the
        # shared-expert intermediate; the latent projections (wq_a, wkv), compressor /
        # indexer and HC functions stay replicated.
        for name, total in (
            ("n_heads", args.n_heads),
            ("o_groups", args.o_groups),
            ("moe_inter_dim", args.moe_inter_dim),
        ):
            if total % size:
                raise ValueError(f"DeepSeek-V4 TP size {size} does not divide {name}={total}")

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(src: str, dst: str, *, rows: int | None = None, cols: int | None = None):
        """A block fp8 linear, sharded on the output (rows) or input (cols) axis; the
        e8m0 block scale follows the same axis in 128-row blocks."""
        w = get(f"{src}.weight")
        if rows is not None:
            w = shard_rows(w, rank=rank, size=size, rows=rows)
        elif cols is not None:
            w = shard_cols(w, rank=rank, size=size, cols=cols)
        yield f"{dst}.weight", w
        if reader.has(f"{src}.scale"):
            sc = get(f"{src}.scale")
            if rows is not None:
                sc = shard_rows(sc, rank=rank, size=size, rows=rows // 128)
            elif cols is not None:
                sc = shard_cols(sc, rank=rank, size=size, cols=cols // 128)
            yield f"{dst}.weight_scale_inv", sc

    heads = args.n_heads // size
    groups = args.o_groups // size
    inter = args.moe_inter_dim // size
    if size > 1:
        # e8m0 block scales are indexed in 128-row/coli blocks; the shards must align.
        for label, span in (
            ("heads x head_dim", heads * args.head_dim),
            ("groups x o_lora_rank", groups * args.o_lora_rank),
            ("moe_inter_dim", inter),
        ):
            if span % 128:
                raise ValueError(f"DeepSeek-V4 TP {label} shard {span} is not a multiple of 128")

    try:
        yield "model.embed.weight", vocab_rows(get("embed.weight"), rank=rank, size=size, vocab_size=args.vocab_size)
        yield "model.norm.weight", get("norm.weight")
        yield "model.head.weight", vocab_rows(get("head.weight"), rank=rank, size=size, vocab_size=args.vocab_size)
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield f"model.{nm}", get(nm)

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            m = f"model.{a}"
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", f"{m}.wq_b", rows=heads * args.head_dim)
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
            yield f"{m}.wo_a", shard_rows(
                _dequant_fp8_block(get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")),
                rank=rank, size=size, rows=groups * args.o_lora_rank,
            )
            yield from linear(f"{a}.wo_b", f"{m}.wo_b", cols=groups * args.o_lora_rank)
            yield f"{m}.attn_sink", shard_rows(
                get(f"{a}.attn_sink"), rank=rank, size=size, rows=heads
            )

            ratio = args.compress_ratios[L]
            if ratio:
                c = f"{a}.compressor"
                for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                    yield f"model.{c}.{nm}", get(f"{c}.{nm}")
                if ratio == 4:
                    idx = f"{a}.indexer"
                    yield from linear(f"{idx}.wq_b", f"model.{idx}.wq_b")
                    yield f"model.{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                    ic = f"{idx}.compressor"
                    for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                        yield f"model.{ic}.{nm}", get(f"{ic}.{nm}")

            yield f"model.layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"model.layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"model.{g}.weight", get(f"{g}.weight")
            if L < args.n_hash_layers:
                yield f"model.{g}.tid2eid", get(f"{g}.tid2eid")
            else:
                yield f"model.{g}.bias", get(f"{g}.bias")
            for proj in ("w1", "w2", "w3"):
                src = f"layers.{L}.ffn.shared_experts.{proj}"
                if proj == "w2":
                    yield from linear(src, f"model.{src}", cols=inter)
                else:
                    yield from linear(src, f"model.{src}", rows=inter)

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"model.layers.{L}.{nm}", get(f"layers.{L}.{nm}")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed MXFP4 expert pieces.
# --------------------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "scale": "_scale"}


def iter_expert_pieces(model_path: str, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Routed experts, one piece per expert: ``{gate, up, down}`` e2m1 pairs and their e8m0
    ``_scale`` companions (``w1`` / ``w3`` / ``w2``). The MTP layer's experts are skipped."""
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    args = load_args(model_path, max_batch_size=1)
    L, E = args.n_layers, args.n_routed_experts

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) >= L:
            return None
        return int(m["layer"]), int(m["expert"]), _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]]

    if parallel:
        tensors = iter_expert_tensors_parallel(model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk)
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = _ShardReader(model_path, _weight_map(model_path), torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading DSV4 experts (serial)", disable=not get_tp_info().is_primary()):
                for e in range(E):
                    base = f"layers.{li}.ffn.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_ in ("weight", "scale"):
                            name = f"{base}.{proj}.{kind_}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces"]
