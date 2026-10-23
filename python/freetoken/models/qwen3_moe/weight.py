"""Qwen3-MoE checkpoint reader.

The dense pass reads every Linear module under the scheme the checkpoint's QuantConfig
gives it, the same answer the model built its buffers from, so bf16 and HF fp8 block
exports (Qwen3-235B-A22B-FP8: e4m3 weights + 128x128 ``weight_scale_inv``) both land as
the model's state dict. Routed experts are read by the offload cache (``iter_expert_pieces``)
for the quantized form; bf16 experts and resident block-fp8 experts come from here.
"""

from __future__ import annotations

import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config, quantization_config_of
from freetoken.models.loader import (
    MergeRule,
    ShardReader,
    iter_merged_tensors,
    iter_stacked_experts,
    iter_weight_files,
    shard_tensor,
)
from freetoken.models.register import ModelSpec, get_model_spec
from freetoken.utils import align_ceil, cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
# per-expert tensors of a quantized checkpoint: the offload cache's expert reader takes these
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}

# leaves the model builds as Linear layers: only their tensors are read under the QuantConfig, the rest passes through as stored
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "gate", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return raw_name.removeprefix("language_model.")


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` fp8 scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"fp8 weight_scale has {flat.numel()} elements for a weight with {rows} output rows "
            f"(shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


def _dequant_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_global: torch.Tensor) -> torch.Tensor:
    """Packed NVFP4 -> bf16 on CUDA (the kernel is GPU-only, the converter reads on CPU), returned on the caller's device."""
    device = weight.device
    if device.type != "cuda":
        weight, weight_scale, weight_global = (t.to("cuda") for t in (weight, weight_scale, weight_global))
    slots = torch.zeros(1, dtype=torch.int32, device=weight.device)
    out = dequant_nvfp4(
        weight.unsqueeze(0).contiguous(), weight_scale.unsqueeze(0).contiguous(), weight_global.unsqueeze(0),
        slots, dtype=torch.bfloat16,
    )[0]
    return out.to(device)


def _dequant(scheme: QuantScheme, part: dict[str, torch.Tensor]) -> torch.Tensor:
    """bf16 weight of a module the checkpoint quantized but the family serves unquantized."""
    weight = part["weight"]
    if scheme.kind is QuantKind.FP8_TENSOR:
        return (weight.to(torch.float32) * part["weight_scale"][:, None]).to(torch.bfloat16)
    if scheme.kind is QuantKind.FP8_BLOCK:
        from freetoken.kernel.triton.fp8_block_linear import dequant_block_fp8

        return dequant_block_fp8(weight, part["weight_scale_inv"])
    if scheme.kind is QuantKind.NVFP4:
        return _dequant_nvfp4(weight, part["weight_scale"], part["weight_global"])
    raise NotImplementedError(f"no bf16 dequantization for {scheme}")


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in."""

    def __init__(self, quant: QuantConfig | None, spec: ModelSpec) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in spec.packed_modules_mapping if fused != "experts"}
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores})
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def stored(self, module: str) -> QuantScheme | None:
        """The scheme the checkpoint stores ``module`` under, before the family's unquantized_modules."""
        if self.quant is None:
            return None
        return self.quant.scheme_for_name(self.quant.name_map.to_checkpoint(module)[0])

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        fused, idx = candidates[0]
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        stored = self.stored(module)
        roles = {"weight": "weight"} if stored is None else {e.name: r for r, e in self.quant.storage(stored).items()}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {stored or 'unquantized'}, stored as {sorted(roles)}"
            )
        if stored is None and tensor.dtype in _QUANT_DTYPES:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
        if stored is not None and role == "weight" and tensor.dtype is not _ELEM_DTYPES[stored.weight.elem]:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {stored}")
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, stored))
        parts.setdefault(idx, {})[role] = tensor
        expected[idx] = set(roles.values())
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], stored)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, stored) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    def _emit(self, target: str, parts: list[dict[str, torch.Tensor]], stored: QuantScheme | None):
        if stored is not None:
            parts = [self._check(target, stored, part) for part in parts]
            if self.scheme(target) is None:
                parts = [{"weight": _dequant(stored, part)} for part in parts]
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            value = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
            out.append((f"{target}.{role}", value))
        return out

    def _check(self, target: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one part against ``scheme`` and put its scales in the layer's form."""
        part = {
            role: 1.0 / tensor.to(torch.float32) if self.quant.storage(scheme)[role].reciprocal else tensor
            for role, tensor in part.items()
        }
        weight = part["weight"]
        if weight.dtype is not _ELEM_DTYPES[scheme.weight.elem]:
            raise ValueError(f"{target}: weight is {weight.dtype} but the checkpoint's quant config declares {scheme}")
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            out[scale_role] = _per_row_scale(part[scale_role], rows)
        else:
            if rows % block_rows or cols % block_cols:
                raise ValueError(f"{target}: {rows}x{cols} weight is not a multiple of the {block_rows}x{block_cols} scale block of {scheme}")
            expected = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != expected:
                raise ValueError(f"{target}: {scale_role} is {tuple(part[scale_role].shape)}, expected {expected} for {scheme}")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{target}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.to(torch.float16).expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense weights fused to the model's buffers, and the routed experts only where a resident path takes them from here.

    Quantized checkpoints read the dense pass through the scheme reader; per-expert FP8
    experts come from ``iter_expert_pieces`` (offload) or ``_resident_fp8_experts`` (fused).
    bf16 checkpoints use the plain merge+stack pipeline.
    """
    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    if quantization_config_of(hf_config) is not None:
        if include_non_moe:
            reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0]))
            yield from _iter_shards(model_path, device, reader, config)
        if include_moe_experts and config.is_moe and config.expert_quant == "fp8_block":
            yield from _resident_fp8_experts(model_path, config)
        return
    yield from _iter_bf16(
        model_path, device, config,
        include_moe_experts=include_moe_experts, include_non_moe=include_non_moe,
    )


def _iter_shards(model_path: str, device: torch.device, reader: _DenseReader, config) -> Iterator[tuple[str, torch.Tensor]]:
    tp_info = get_tp_info()
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None or _EXPERT_RE.search(name):
                    continue
                tensor = f.get_tensor(raw_name)
                if tp_info.size > 1 and not name.endswith((".input_scale", ".input_global_scale")):
                    # TP shards the raw checkpoint tensors before the reader validates them
                    # against the rank-local module shapes: heads on dim 0 (GQA kv heads take
                    # the loader's head mapping), inputs on dim 1, vocab by ceil partition,
                    # scales and everything else replicated.
                    tensor = shard_tensor(
                        name, tensor,
                        rank=tp_info.rank, world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                emitted = reader.add(name, tensor)
                if emitted is not None:
                    yield from emitted
                else:
                    yield name, tensor
    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def _ro_sharded_tensors(
    model_path: str, device: torch.device, config, *,
    include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    tp_info = get_tp_info()

    def sharded_tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in tqdm(
            iter_weight_files(model_path),
            desc="Loading weights",
            disable=not tp_info.is_primary(),
        ):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for raw_name in f.keys():
                    name = _rename(raw_name)
                    if name is None:
                        continue
                    is_expert = _EXPERT_PATTERN.match(name) is not None
                    if is_expert and not include_moe_experts:
                        continue
                    if not is_expert and not include_non_moe:
                        continue
                    raw = f.get_tensor(raw_name)
                    tensor = shard_tensor(
                        name,
                        raw,
                        rank=tp_info.rank,
                        world_size=tp_info.size,
                        num_kv_heads=config.num_kv_heads,
                    )
                    del raw
                    yield name, tensor

    return sharded_tensors()


def _iter_bf16(
    model_path: str,
    device: torch.device,
    config,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    merged = iter_merged_tensors(
        _ro_sharded_tensors(model_path, device, config, include_moe_experts=include_moe_experts, include_non_moe=include_non_moe),
        _MERGE_RULES,
        model_name="qwen3_moe",
    )
    yield from iter_stacked_experts(
        merged,
        num_experts=config.num_experts,
        model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


def iter_weights_parallel(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    workers: int = 8,
    chunk: int = 8 << 20,
) -> Iterator[tuple[str, torch.Tensor]]:
    """experts-only iter_weights: raw experts read via the common chunked multi-threaded
    O_DIRECT reader, then same merge+stack pipeline."""
    assert include_moe_experts and not include_non_moe, (
        "qwen3_moe parallel reader is experts-only (used by the expert piece reader)"
    )
    from freetoken.models.weight import iter_expert_tensors_parallel

    config = parse_config(cached_load_hf_config(model_path))
    tp_info = get_tp_info()

    def _is_expert(raw_name: str) -> bool:
        name = _rename(raw_name)
        return name is not None and _EXPERT_PATTERN.match(name) is not None

    def raw_experts() -> Iterator[tuple[str, torch.Tensor]]:
        for raw_name, raw in iter_expert_tensors_parallel(
            model_path, _is_expert, workers=workers, chunk=chunk
        ):
            name = _rename(raw_name)
            tensor = shard_tensor(
                name, raw, rank=tp_info.rank, world_size=tp_info.size,
                num_kv_heads=config.num_kv_heads,
            )
            yield name, tensor

    merged = iter_merged_tensors(raw_experts(), _MERGE_RULES, model_name="qwen3_moe")
    yield from iter_stacked_experts(
        merged, num_experts=config.num_experts, model_name="qwen3_moe",
        expert_pattern=_EXPERT_PATTERN,
    )


# ======================================================================================
# Block-FP8 routed experts (Qwen3-235B-A22B-FP8): offload expert pieces and resident stacks.
# ======================================================================================

# Routed-expert checkpoint key (per-expert, un-fused). The optional ``language_model.``
# prefix covers the VL-style key roots; ``mtp.layers...`` is excluded by the layers anchor.
_FP8_EXPERT_KEY_RE = (
    r"^model\.(?:language_model\.)?layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|{scale})$"
)


def _moe_dims(model_config):
    L = model_config.num_moe_layers
    return (
        L, model_config.num_experts, model_config.hidden_size,
        model_config.moe_intermediate_size, model_config.num_layers - L,  # dense prefix
    )


def _resident_fp8_experts(model_path, config):
    from freetoken.kernel.triton.fp8_block_linear import FP8

    B = 128
    L, E, H, I, dense = _moe_dims(config)
    shapes = {
        "gate_up_proj": ((E, 2 * I, H), FP8),
        "gate_up_scale_inv": ((E, 2 * I // B, H // B), torch.bfloat16),
        "down_proj": ((E, H, I), FP8),
        "down_scale_inv": ((E, H // B, I // B), torch.bfloat16),
    }
    layers: dict[int, dict[str, torch.Tensor]] = {}
    placed = [0] * L
    for li, e0, e1, piece in iter_expert_pieces(model_path, config, QuantKind.FP8_BLOCK, parallel=None):
        stack = layers.setdefault(li, {n: torch.empty(shape, dtype=dt) for n, (shape, dt) in shapes.items()})
        stack["gate_up_proj"][e0:e1, :I] = piece["gate"]
        stack["gate_up_proj"][e0:e1, I:] = piece["up"]
        stack["gate_up_scale_inv"][e0:e1, : I // B] = piece["gate_scale"]
        stack["gate_up_scale_inv"][e0:e1, I // B :] = piece["up_scale"]
        stack["down_proj"][e0:e1] = piece["down"]
        stack["down_scale_inv"][e0:e1] = piece["down_scale"]
        placed[li] += e1 - e0
        if placed[li] == E:
            pre = f"model.layers.{dense + li}.mlp.experts"
            for name, tensor in layers.pop(li).items():
                yield f"{pre}.{name}", tensor
    assert not layers, f"incomplete resident fp8 experts for layers {sorted(layers)}"


def iter_expert_pieces(model_path, config, kind: QuantKind, *, parallel: bool | None = False, workers: int = 8, chunk: int = 8 << 20):
    """Block-fp8 routed experts, one piece per expert: ``{gate, up, down}`` fp8 codes and their
    ``_scale`` (block scale) companions, named as the checkpoint's dialect stores them. Other expert kinds use the generic readers.

    TP serves the rank-local intermediate shard: every rank reads only its row/column
    slice out of each raw tensor (``ShardReader.get_slice``), so the per-rank bytes are
    ``1/tp_size`` of the checkpoint instead of every rank re-reading it fully (eight
    concurrent full readers contend on the page cache and run ~30x slower than one).
    The pieces handed to ``pack`` are therefore rank-local width under TP; TP=1 keeps the
    full check-point convention."""
    if kind is not QuantKind.FP8_BLOCK:
        return None
    from freetoken.moe.expert_pieces import per_expert_pieces
    from freetoken.models.weight import experts_scattered, iter_expert_tensors_parallel

    tp_info = get_tp_info()
    L, E, H, I, dense = _moe_dims(config)
    scale = get_quant_config().stored_tensors(QuantKind.FP8_BLOCK)["weight_scale_inv"].name
    key_re = re.compile(_FP8_EXPERT_KEY_RE.format(scale=re.escape(scale)))
    suffix = {"weight": "", scale: "_scale"}

    def locate(raw_name: str):
        m = key_re.match(raw_name)
        if m is None:
            return None
        li = int(m["layer"]) - dense
        if not 0 <= li < L:
            raise ValueError(f"unexpected routed-expert layer in {raw_name}")
        return li, int(m["expert"]), m["proj"] + suffix[m["kind"]]

    if tp_info.size > 1:
        # rank-local slices; the parallel (whole-shard) reader has nothing to offer
        # a rank that only needs a slice, so it is not an option under TP.
        parallel = False
    elif parallel is None:
        parallel = experts_scattered(model_path)
    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path, lambda n: key_re.match(n) is not None, workers=workers, chunk=chunk
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        reader = ShardReader(model_path, torch.device("cpu"))
        try:
            # TP: one rank's span of the real intermediate, block-aligned banks.
            B = 128
            rank, size = tp_info.rank, tp_info.size
            span = align_ceil(I // size, B)
            lo = rank * span
            take = min(max(I - lo, 0), span)
            s0, s1 = lo // B, (lo + take + B - 1) // B
            tp = size > 1
            for li in tqdm(range(L), desc="Loading fp8 experts (serial)", disable=not tp_info.is_primary()):
                for e in range(E):
                    base = f"model.layers.{dense + li}.mlp.experts.{e}"
                    for proj in ("gate", "up", "down"):
                        for kind_name, suf in suffix.items():
                            name = f"{base}.{proj}_proj.{kind_name}"
                            if not tp:
                                yield name, reader.get_tensor(name)
                                continue
                            sl = reader.get_slice(name)
                            if proj == "down":
                                yield name, sl[:, lo:lo + take] if kind_name == "weight" else sl[:, s0:s1]
                            else:
                                yield name, sl[lo:lo + take] if kind_name == "weight" else sl[s0:s1]
        finally:
            reader.close()

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_weights_parallel", "iter_expert_pieces"]
