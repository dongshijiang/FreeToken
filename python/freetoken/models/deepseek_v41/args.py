"""DeepSeek-V4.1-Flash hyperparameters.

Field names keep the V4 engine's dialect (``dim`` / ``moe_inter_dim`` / ...) so the
ported attention / MoE machinery stays diffable against ``deepseek_v4``; ``load_args``
maps the HF ``config.json`` -> ``text_config`` keys onto them (V4.1 ships no
``inference/config.json``). V4.1-only schedules (kv/index source layers, the
candidate-source layer, Engram) keep their HF names.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Literal, Tuple


@dataclass
class DeepseekV41VisionArgs:
    """The checkpoint's ``vision_config`` section; field names stay verbatim.

    Serves the released 2D-RoPE ViT (patch 14, full bidirectional attention,
    RMSNorm eps 1e-6) and the padded spatial aligner (GELU, downsample_ratio 3).
    """

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    # image-span budget: 2 + llm_h * (llm_w + 1) tokens per image
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class DeepseekV41Args:
    # ----- runtime -----
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    # fp8 weight-scale blocks: 32x32 on V4.1 (V4 used 128x128).
    weight_block_size: Tuple[int, int] = (32, 32)
    # Checkpoint dir, set by parse_config; engram needs it for the tokenizer rebuild
    # and the CPU-resident memory tables. Never read from config.json.
    model_path: str = ""

    # ----- shape -----
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    # the 3 nextn (DSpark draft) layers are checkpoint-only and never served
    num_nextn_predict_layers: int = 3
    n_heads: int = 64

    # ----- moe -----
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # V4.1 has no hash-routed layers (the shared Gate reads it; 0 -> score routing).
    n_hash_layers: int = 0

    # ----- csa2 attention -----
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    # [0, 0] + [2] * 18 + [1] * 20 + [0] * 3; the trailing zeros cover the nextn layers
    compress_ratios: Tuple[int, ...] = (
        0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0,
    )
    # Full-mode layers that own a compressor and publish compressed_kv to every
    # downstream layer of the same ratio (2 -> 3..7, 8 -> 9..14, 14 -> 15..19, 20 -> 21..39)
    kv_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20)
    # Reindex layers re-run the indexer instead of reusing the upstream top-k
    index_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    # layer 20 additionally maintains the candidate block set (block8 x 2048 blocks,
    # newest block forced) that Reindex layers pick their top-512 blocks within
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # ----- rope / yarn -----
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- lightning indexer -----
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    # ----- hyper-connections -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ----- engram -----
    engram_layer_ids: Tuple[int, ...] = (1, 14)
    engram_num_embeddings: Tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092

    # ----- vision (absent on text-only releases) -----
    image_token_id: int | None = None
    vision_args: DeepseekV41VisionArgs | None = None

    def __post_init__(self) -> None:
        # JSON lists -> tuple so the dataclass stays hashable / immutable-ish.
        for name in (
            "compress_ratios", "kv_source_layer_ids", "index_source_layer_ids",
            "engram_layer_ids", "engram_num_embeddings", "weight_block_size",
        ):
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim


# HF text_config key -> args field, for the fields whose names the engine inherited
# from the V4 reference dialect.
_HF_ALIASES = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_experts_per_tok": "n_activated_experts",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "scoring_func": "score_func",
}

# rope_scaling entry -> args field
_SCALING_ALIASES = {
    "original_max_position_embeddings": "original_seq_len",
    "factor": "rope_factor",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
}


def load_vision_args(raw: dict) -> DeepseekV41VisionArgs:
    """Build :class:`DeepseekV41VisionArgs` from the raw ``vision_config`` JSON section.

    The checkpoint's field names already match the dataclass, so unknown keys
    (e.g. torch_dtype) are dropped and the rest map 1:1.
    """
    valid = {f.name for f in fields(DeepseekV41VisionArgs)}
    return DeepseekV41VisionArgs(**{k: v for k, v in raw.items() if k in valid})


def load_args(model_path: str, **overrides) -> DeepseekV41Args:
    """Build :class:`DeepseekV41Args` from the checkpoint's ``config.json``.

    Reads the nested ``text_config`` (multimodal wrapper); precision knobs come from
    the top-level ``quantization_config``. ``overrides`` (e.g. ``max_seq_len``)
    take precedence, letting the runner size the per-request caches.
    """
    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)
    text = raw.get("text_config") or raw
    valid = {f.name for f in fields(DeepseekV41Args)}
    kwargs = {}
    for key, value in text.items():
        key = _HF_ALIASES.get(key, key)
        if key in valid:
            kwargs[key] = value
    scaling = text.get("rope_scaling") or {}
    for src, dst in _SCALING_ALIASES.items():
        if src in scaling:
            kwargs[dst] = scaling[src]
    quant = raw.get("quantization_config") or {}
    if quant.get("quant_method") == "fp8":
        kwargs.setdefault("dtype", "fp8")
        if "scale_fmt" in quant:
            kwargs.setdefault("scale_fmt", quant["scale_fmt"])
        if "expert_dtype" in quant:
            kwargs.setdefault("expert_dtype", quant["expert_dtype"])
        if "weight_block_size" in quant:
            kwargs.setdefault("weight_block_size", tuple(quant["weight_block_size"]))
    else:
        kwargs.setdefault("dtype", "bf16")
    vision = raw.get("vision_config") or {}
    if vision:
        kwargs["vision_args"] = load_vision_args(vision)
        kwargs["image_token_id"] = raw.get("image_token_id")
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


__all__ = ["DeepseekV41Args", "DeepseekV41VisionArgs", "load_args", "load_vision_args"]
