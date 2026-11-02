"""DeepSeek-V4.1 MoE: the V4 router / experts verbatim.

Same sqrtsoftplus scoring + bias + top-k + renorm + route_scale machinery; V4.1 only
drops hash routing, which the shared Gate keys off ``args.n_hash_layers`` (= 0 here).
"""

from ..deepseek_v4.moe import DSV4OffloadMoELayer, Expert, Gate, MoE

__all__ = ["DSV4OffloadMoELayer", "Expert", "Gate", "MoE"]
