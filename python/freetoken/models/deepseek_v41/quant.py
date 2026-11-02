"""Cache QAT round-trips (bf16 pools, V4.1 formats).

The paged KV pools store bf16 (the quant is a quantize->dequantize round-trip baked
into the value, so ``index_select`` staging stays byte-exact). V4.1 quantizes the
WHOLE head_dim (rope channels included) in these groups, unlike V4's partial-tensor
in-place kernels:

  - window KV: fp8 e4m3, 32-channel groups, power-of-two scale = 2^ceil(log2(amax/448))
  - compressed KV: nvfp4, 16-channel groups, e4m3 scale = amax/6
  - index q/k: mxfp4, 32-channel groups, e8m0 power-of-two scale = 2^ceil(log2(amax/6))

Mirrors the reference ``_CacheQuantization`` bit for bit (round-half-to-even fp4
levels, same scale clamps).
"""

from __future__ import annotations

import torch

_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _power_of_two_ceiling(value: torch.Tensor) -> torch.Tensor:
    # IEEE754 exponent bump without a log round-trip.
    bits = value.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    increment = (bits & ((1 << 23) - 1)) != 0
    return ((exponent + increment.to(torch.int32)) << 23).view(torch.float32)


def _fp4_round(normalized: torch.Tensor) -> torch.Tensor:
    magnitude = normalized.abs()
    midpoints = magnitude.new_tensor(_MIDPOINTS)
    levels = magnitude.new_tensor(_LEVELS)
    lower = torch.bucketize(magnitude, midpoints)
    tie = magnitude == midpoints[lower.clamp_max(6)]
    selected = lower + (tie & (lower % 2 == 1)).to(lower.dtype)
    return torch.copysign(levels[selected], normalized)


def cache_qdq(values: torch.Tensor, fmt: str, block_size: int) -> torch.Tensor:
    """Quantize->dequantize ``values[..., channels]`` in ``fmt``/``block_size`` groups.

    Returns a NEW tensor in the input dtype (the reference's straight-through
    representation, minus autograd).
    """
    channels = values.shape[-1]
    grouped = torch.nn.functional.pad(
        values.float(), (0, (-channels) % block_size)
    ).unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(-1, keepdim=True)
    if fmt == "fp8":
        scale = _power_of_two_ceiling(amax.clamp_min(1e-4) * (1 / 448))
        rounded = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        if fmt == "nvfp4":
            scale = (amax.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        else:  # mxfp4
            scale = _power_of_two_ceiling(amax.clamp_min(6 * 2**-126) * (1 / 6))
        rounded = _fp4_round((grouped / scale).clamp(-6, 6))
    return (rounded * scale).flatten(-2)[..., :channels].to(values.dtype)


__all__ = ["cache_qdq"]
