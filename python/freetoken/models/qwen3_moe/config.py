from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import ModelConfig, RotaryConfig


def _expert_quant(hf_config: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    if not (getattr(hf_config, "num_experts", 0) or 0):
        return "none", None
    # the engine reads this tag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def parse_config(hf_config: Any) -> ModelConfig:
    num_kv_heads = getattr(
        hf_config,
        "num_key_value_heads",
        hf_config.num_attention_heads,
    )
    expert_quant, weight_block_size = _expert_quant(hf_config)
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None and rope_scaling is not None:
        rope_theta = rope_scaling["rope_theta"]

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=getattr(
            hf_config,
            "num_local_experts",
            getattr(hf_config, "num_experts", 0),
        ),
        num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", False)),
        model_type=getattr(hf_config, "model_type", "qwen3_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3MoeForCausalLM"]),
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
    )


__all__ = ["parse_config"]
