"""DeepSeek-V4.1 vision stack: the released full-attention 2D-RoPE ViT, its aligner and the delimiter vectors.

Attention is bidirectional within one image (images never attend across) and the
aligner reduces over 3x3 patch groups, so every tensor is TP-replicated. The
delimiters ride the tower because the loader remaps the checkpoint's top-level
``image_start`` / ``image_end`` / ``image_newline`` vectors under ``vision.``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Sequence

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, OPList
from freetoken.models.weight_stream import BlockWeightStreamer

if TYPE_CHECKING:
    from freetoken.message import MMItem

    from .args import DeepseekV41VisionArgs


class VisionRMSNorm(BaseOP):
    """The checkpoint's bf16 scale with fp32 arithmetic (the reference keeps an fp32
    parameter; bf16 upcasts into fp32 exactly, so the numbers match)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = torch.empty(dim, dtype=torch.bfloat16)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self._eps)
        return (self.weight.float() * normalized).to(x.dtype)


def _vision_cos_sin(
    n_h: int, n_w: int, rope_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference 2D rotary table, ``[n_h * n_w, 1, rope_dim]`` fp32: the first half of
    the frequencies encodes the patch row, the second half the column."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """DeepSeek's half-split 2D rope on ``[patches, heads, head_dim]``."""
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionAttention(BaseOP):
    def __init__(self, va: DeepseekV41VisionArgs, *, prefix: str = ""):
        self.n_heads = va.num_attention_heads
        self.head_dim = va.head_dim
        self.wqkv = LinearReplicated(va.hidden_size, 3 * va.hidden_size, True, prefix=f"{prefix}.wqkv")
        self.wo = LinearReplicated(va.hidden_size, va.hidden_size, True, prefix=f"{prefix}.wo")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv.forward(x).chunk(3, dim=-1))
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)
        # the fused kernels only take 4D; 3D input falls back to math and materializes an N^2 matrix
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
        ).squeeze(0)
        return self.wo.forward(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(BaseOP):
    def __init__(self, va: DeepseekV41VisionArgs, *, prefix: str = ""):
        self.w1 = LinearReplicated(va.hidden_size, 2 * va.intermediate_size, False, prefix=f"{prefix}.w1")
        self.w2 = LinearReplicated(va.intermediate_size, va.hidden_size, False, prefix=f"{prefix}.w2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1.forward(x).chunk(2, dim=-1)
        return self.w2.forward(F.silu(gate) * up)


class VisionBlock(BaseOP):
    def __init__(self, va: DeepseekV41VisionArgs, *, prefix: str = ""):
        self.norm1 = VisionRMSNorm(va.hidden_size)
        self.attn = VisionAttention(va, prefix=f"{prefix}.attn")
        self.norm2 = VisionRMSNorm(va.hidden_size)
        self.mlp = VisionMLP(va, prefix=f"{prefix}.mlp")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class VisionPatchEmbed(BaseOP):
    def __init__(self, va: DeepseekV41VisionArgs, *, prefix: str = ""):
        self.proj = LinearReplicated(
            3 * va.patch_size * va.patch_size, va.hidden_size, True, prefix=f"{prefix}.proj"
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(patches.flatten(1))


class VisionAligner(BaseOP):
    """Padded spatial merger into the text width: zero-pad the patch grid up to a
    ratio multiple, unfold 3x3 groups, then a GELU MLP."""

    def __init__(self, va: DeepseekV41VisionArgs, text_dim: int, *, prefix: str = ""):
        self._ratio = va.downsample_ratio
        self.w1 = LinearReplicated(
            va.hidden_size * va.downsample_ratio**2, text_dim, True, prefix=f"{prefix}.w1"
        )
        self.w2 = LinearReplicated(text_dim, text_dim, True, prefix=f"{prefix}.w2")

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self._ratio
        if x.shape[0] != n_h * n_w:
            raise ValueError(f"Expected {n_h * n_w} patch states for a {n_h}x{n_w} grid, got {x.shape[0]}")
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        # -n % ratio is Python's positive remainder, so this zero-pads up to the
        # ceil(n/ratio) multiple and the output rows stay ceil-aligned
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2.forward(F.gelu(self.w1.forward(x)))


class VisionTower(BaseOP):
    """The checkpoint's ``vision.*`` section: patch embed, block stack, final norm, the
    aligner and the three delimiter vectors."""

    def __init__(self, va: DeepseekV41VisionArgs, text_dim: int, *, prefix: str = "vision"):
        self.patch_embed = VisionPatchEmbed(va, prefix=f"{prefix}.patch_embed")
        self.blocks = OPList(
            [VisionBlock(va, prefix=f"{prefix}.blocks.{i}") for i in range(va.num_hidden_layers)]
        )
        self.norm = VisionRMSNorm(va.hidden_size)
        self.aligner = VisionAligner(va, text_dim, prefix=f"{prefix}.aligner")
        self.image_start = torch.empty(text_dim, dtype=torch.bfloat16)
        self.image_end = torch.empty(text_dim, dtype=torch.bfloat16)
        self.image_newline = torch.empty(text_dim, dtype=torch.bfloat16)
        self._rope_dim = va.hidden_size // va.num_attention_heads // 2
        self._theta = va.rope_theta
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """gpu: every tensor resident; host: block tensors in pinned banks streamed two
        blocks at a time (patch embed, aligner, norm and delimiters stay resident)."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.blocks.op_list, self.norm.weight.device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _blocks(self) -> Iterator[tuple[int, VisionBlock]]:
        if self._streamer is None:
            return enumerate(self.blocks.op_list)
        return self._streamer.blocks(self.blocks.op_list)

    @torch.inference_mode()
    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """One image's patches ``[n_h * n_w, 3, patch, patch]`` -> ViT features ``[n_h * n_w, dim]``."""
        x = self.patch_embed.forward(patches)
        cos, sin = _vision_cos_sin(n_h, n_w, self._rope_dim, self._theta, x.device)
        for _, block in self._blocks():
            x = block.forward(x, cos, sin)
        return self.norm.forward(x)

    @torch.inference_mode()
    def encode(self, item: MMItem) -> torch.Tensor:
        """The item's full soft-token span ``[2 + llm_h * (llm_w + 1), text_dim]``: the
        delimiter vectors composed around the aligner rows in prompt order."""
        n_h, n_w = item.vit_grid
        patches = item.feature.to(self.norm.weight.device, non_blocking=True)
        features = self.aligner.forward(self.forward(patches, n_h, n_w), n_h, n_w)
        llm_h, llm_w = item.llm_grid
        if features.shape[0] != llm_h * llm_w:
            raise ValueError(f"aligner produced {features.shape[0]} rows for a {llm_h}x{llm_w} llm grid")
        span = features.new_empty(2 + llm_h * (llm_w + 1), features.shape[1])
        span[0] = self.image_start
        span[-1] = self.image_end
        body = span[1:-1].view(llm_h, llm_w + 1, -1)
        body[:, :llm_w] = features.view(llm_h, llm_w, -1)
        body[:, llm_w] = self.image_newline
        return span


__all__ = ["VisionTower"]
