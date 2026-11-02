"""DeepSeek-V4.1-Flash model (engine-native FreeToken port of the reference model.py).

Single-stream port of the CSA2 backbone wired onto FreeToken's shared paged engine:

  - Single-pass manifold-constrained Hyper-Connections (4 residual streams): each of a
    block's two HC sites computes ONE mix whose ``post``/``comb`` merge that site's output
    back into the streams, while its ``pre`` is carried to the NEXT site (one-site delay):
    the attention site collapses its input with the incoming pre (one-hot stream 0 at the
    embedding), the FFN site with the attention site's pre, and the LM head collapses the
    final streams with the last FFN pre. Same sinkhorn/combine kernels as V4.
  - CSA2 attention (sliding window + ratio-1/2 compressed tiers + Lightning Indexer with
    candidate/reindex hierarchy) over the DSV4 paged pools, state threaded by
    :class:`~.compress.CSA2State` (see ``attention.py`` / ``compress.py``).
  - Engram n-gram retrieval writes into the streams at layers 1 / 14 (model-level hash).
  - Routed MXFP4 experts from :class:`OffloadMoeCache` (V4's MoE, hash routing off).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV41Args
from .attention import Attention
from .compress import CSA2State
from .moe import MoE


class Block(BaseOP):
    """Decoder block: engram -> attn-HC site -> CSA2 attention -> ffn-HC site -> MoE."""

    def __init__(self, layer_id: int, args: DeepseekV41Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.ffn = MoE(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # fp32 mixers, exactly as the checkpoint ships them (plain buffers, no head here --
        # V4.1 has no hc_head; the final collapse reuses the last site's pre).
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)
        # Attached by the Transformer when the checkpoint ships an engram layer here.
        self.engram = None
        self.engram_head_index = -1

    def _hc_mix(self, x, fn, scale, base):
        """One HC site's mix: ``(pre, post, comb)`` from the streams (fp32 sinkhorn)."""
        shape = x.size()
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(-1, mixes.size(-1)), scale, base,
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
        )
        M = shape[0] * shape[1]
        return pre.view(M, self.hc_mult), post.view(M, self.hc_mult), comb.view(M, self.hc_mult, self.hc_mult)

    def _collapse(self, x, pre):
        """Mix the residual streams into one hidden with the INCOMING site's pre."""
        shape, dtype = x.size(), x.dtype
        M = shape[0] * shape[1]
        xf = x.flatten(2).float().view(M, self.hc_mult, self.dim)
        return hc_pre_combine(xf, pre.reshape(M, self.hc_mult), dtype).view(*shape[:2], self.dim)

    def _expand(self, x, residual, post, comb):
        """Merge a site's output back into the streams (post scaling + comb mixing)."""
        shape = residual.size()
        M = shape[0] * shape[1]
        y = hc_post_combine(
            x.reshape(M, self.dim), residual.reshape(M, self.hc_mult, self.dim),
            post.reshape(M, self.hc_mult), comb.reshape(M, self.hc_mult, self.hc_mult),
        )
        return y.view(shape)

    def _engram_write(self, x, hashes):
        if self.engram is not None and hashes is not None:
            # Two-step by contract: the host-sync table gather cannot fuse with the
            # gate's projection, and the lookup output feeds forward as plain bf16.
            emb = self.engram.lookup(hashes[:, :, self.engram_head_index])
            return self.engram.forward(x, emb)
        return x

    def prefill_batched(self, x, pre_mix, input_ids, segments, flat_positions, state, hashes=None):
        x = self._engram_write(x, hashes)
        residual = x
        pre, post, comb = self._hc_mix(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm.forward(self._collapse(x, pre_mix))
        x = self.attn.forward_ragged(x, segments, flat_positions, state)
        x = self._expand(x, residual, post, comb)

        residual = x
        pre2, post2, comb2 = self._hc_mix(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(self._collapse(x, pre))
        x = self.ffn.forward(x, input_ids)
        x = self._expand(x, residual, post2, comb2)
        return x, pre2, state

    def decode_step(self, x, pre_mix, pos, rows, cmp_stage_cap, input_ids, wctx, state, hashes=None):
        x = self._engram_write(x, hashes)
        residual = x
        pre, post, comb = self._hc_mix(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm.forward(self._collapse(x, pre_mix))
        x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx, state)
        x = self._expand(x, residual, post, comb)

        residual = x
        pre2, post2, comb2 = self._hc_mix(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(self._collapse(x, pre))
        x = self.ffn.forward(x, input_ids)
        x = self._expand(x, residual, post2, comb2)
        return x, pre2, state


class Transformer(BaseOP):
    def __init__(self, args: DeepseekV41Args, quant_config=None, *, strategy: str = "offload", decode_target: str = "gpu", prefix: str = ""):
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_mult = hc_mult = args.hc_mult
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = OPList([Block(i, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.layers.{i}") for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelLMHead(args.vocab_size, args.dim, quant_config=quant_config, prefix=f"{prefix}.head")
        # Model-level n-gram hash + the per-layer engram writers (layers 1 / 14). The
        # ~384M-row embedding tables stay CPU resident inside the Engram module.
        self.engram_hash = None
        if args.engram_layer_ids:
            from .engram import EngramLayer, NgramHash

            self.engram_hash = NgramHash(args)
            for head_index, li in enumerate(args.engram_layer_ids):
                block = self.layers.op_list[li]
                block.engram = EngramLayer(
                    args, head_index, quant_config=quant_config, prefix=f"{prefix}.layers.{li}.engram"
                )
                block.engram_head_index = head_index

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers.op_list:
            layer.attn.bind(pool, device)

    def _embed_streams(self, input_ids: torch.Tensor):
        h = self.embed.forward(input_ids.reshape(-1))
        h = h.view(*input_ids.shape, self.args.dim).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        # Incoming pre at the embedding: one-hot stream 0 (fp32, the collapse's mixer).
        pre = h.new_zeros(*input_ids.shape, self.hc_mult, dtype=torch.float32)
        pre[..., 0] = 1
        return h, pre

    def prefill_batched(self, input_ids: torch.Tensor, segments, flat_positions: torch.Tensor) -> torch.Tensor:
        # Ragged batched prefill (bs >= 1); see V4's Transformer for the addressing contract.
        h, pre = self._embed_streams(input_ids.view(1, -1))
        state = CSA2State()
        hashes = self.engram_hash(input_ids.view(1, -1)) if self.engram_hash is not None else None
        for layer in self.layers.op_list:
            h, pre, state = layer.prefill_batched(h, pre, input_ids, segments, flat_positions, state, hashes)
        # No hc_head: the final streams collapse with the last site's pre.
        M = h.size(1)
        collapsed = hc_pre_combine(
            h.flatten(2).view(M, self.hc_mult, self.args.dim), pre.view(M, self.hc_mult), h.dtype
        )
        h = self.norm.forward(collapsed)
        # collapsed is [M, dim]: one row per token; the head picks the last index
        # of each request itself (see ParallelLMHead.forward).
        return self.head.forward(h)

    def decode(self, input_ids: torch.Tensor, pos: torch.Tensor, cmp_stage_cap: int) -> torch.Tensor:
        B = input_ids.size(0)
        rows = torch.arange(B, device=input_ids.device)
        h, pre = self._embed_streams(input_ids)
        state = CSA2State()
        hashes = self.engram_hash(input_ids) if self.engram_hash is not None else None
        # Layer-invariant window-ring slots hoisted once (see V4's decode contract).
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        for layer in self.layers.op_list:
            h, pre, state = layer.decode_step(h, pre, pos, rows, cmp_stage_cap, input_ids, wctx, state, hashes)
        collapsed = hc_pre_combine(
            h.view(B, self.hc_mult, self.args.dim), pre.view(B, self.hc_mult), h.dtype
        )
        h = self.norm.forward(collapsed)
        # collapsed is [B, dim]: one row per sequence, already the last token.
        return self.head.forward(h)


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine adapter: a registered :class:`BaseLLMModel` wrapping the V4.1 transformer.

    KV pools, rope constants and compressor state are bound on the first forward.
    """

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV41Args = config.dsv4_args
        self.model = Transformer(self._args, config.quant, strategy=config.moe_strategy, decode_target=config.decode_target, prefix="model")
        # Engram lookup syncs to the host (CPU-resident tables): decode CUDA graphs
        # must stay off, the engine reads this flag before building its GraphRunner.
        self.needs_eager_decode = bool(self._args.engram_layer_ids)
        self._bound = False

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self.model.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward (the model holds no pool reference; the
        per-bind scratch depends on the new pool's geometry)."""
        self._bound = False

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            return self.model.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long(),
            )
        B = batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        return self.model.decode(input_ids.view(B, 1), pos, cmp_stage_cap)


__all__ = ["Transformer", "DeepseekV41ForCausalLM"]
