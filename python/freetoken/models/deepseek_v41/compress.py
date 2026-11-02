"""CSA2 state machine: ratio-1/2 compressors + the Lightning Indexer + block candidates.

V4.1's compression is a per-ratio-group pipeline threaded through the layer stack by
:class:`CSA2State` (the model creates one per forward):

  - kv_source layers (2 / 8 / 14 / 20) own a compressor and publish compressed_kv into
    the group's cmp pool; consumers address the SOURCE layer's pool rows;
  - kv_source layers also publish the indexer's mxfp4 keys (``wk`` on the pooled latent);
  - index_source layers re-run the indexer and replace ``topk_indices`` (2 / 8 / 14 /
    20 / 24 / 28 / 32 / 36); other compressed layers reuse the most recent pick;
  - layer 20 (candidate_source) additionally publishes the candidate block set
    (block8 x 2048, newest block forced) that the Reindex layers (24 / 28 / 32 / 36)
    must pick their top-k within.

The compressor math is V4's non-overlap gated softmax pooling (no APE, no overlap
transform, no Hadamard), with the reference's cache QAT round-trips from ``quant.py``.
Decode keeps V4's per-window-page fp32 carry ring (ratio 2 only; ratio 1 has no state).
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.compress import gated_pool
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from ..deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode
from .quant import cache_qdq


class CSA2State:
    """Per-forward compression state threaded through the layer stack.

    ``topk_indices`` are compressed BLOCK indices (-1 = invalid) aligned with the
    query stream: ``[1, T, k]`` on prefill, ``[B, 1, k]`` on decode. ``candidates`` is
    layer 20's block mask -- prefill keeps one tensor per segment (their visible
    widths differ), decode a single ``[B, 1, n_stage]`` bool. The pool references are
    LAYER IDS (cmp pool / idx pool of the group's kv_source layer), not tensors.
    """

    __slots__ = ("topk_indices", "candidates", "cmp_layer", "keys_layer", "ratio")

    def __init__(self) -> None:
        self.topk_indices: torch.Tensor | None = None
        self.candidates: torch.Tensor | list[torch.Tensor] | None = None
        self.cmp_layer: int | None = None
        self.keys_layer: int | None = None
        self.ratio: int = 0


class Compressor(BaseOP):
    """Gated softmax pooling over 2 tokens (ratio 2) or a plain projection (ratio 1).

    Non-overlapping: every compressed position k pools tokens [k*ratio, (k+1)*ratio).
    Publishes the roped + nvfp4-quantized compressed_kv to the group's cmp pool and
    returns the pre-rope latent so the layer's indexer can project its keys from it.
    """

    def __init__(self, args, compress_ratio: int, head_dim: int, *, quant_config=None, prefix: str = ""):
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.max_batch_size = args.max_batch_size

        # bf16 in the checkpoint (the fp8 config does not cover the compressors);
        # matmul upcasts on-chip.
        self.wkv = LinearReplicated(self.dim, head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.wgate = (
            LinearReplicated(self.dim, head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wgate")
            if compress_ratio > 1 else None
        )
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # Ratio 2 carries the odd token of an incomplete block across steps in a
        # per-window-page fp32 ring (ring row = the block's even slot); ratio 1 is stateless.
        self.layer_id: int | None = None
        self.ring_size = compress_ratio  # 2 (state rows per window page) / unused for ratio 1
        self.item_size = head_dim
        self.P: int = 128
        self._freqs_cis: torch.Tensor | None = None
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_paged(self, pool, layer_id: int, freqs_cis, device) -> None:
        self.layer_id = layer_id
        self.P = pool.P
        self._freqs_cis = freqs_cis
        self._device = device
        if self.compress_ratio > 1:
            self._kv_state = torch.zeros(
                1, self.compress_ratio, self.head_dim, dtype=torch.float32, device=device
            )
            self._score_state = torch.full(
                (1, self.compress_ratio, self.head_dim),
                float("-inf"), dtype=torch.float32, device=device,
            )

    def reset(self) -> None:
        if self._kv_state is not None:
            self._kv_state.zero_()
            self._score_state.fill_(float("-inf"))

    def _publish(self, latent: torch.Tensor, start_pos: int, cutoff: int, ti: int) -> None:
        """Rope the latent at the compressed positions, nvfp4-quantize it, scatter to the
        group's cmp pool rows. ``latent`` is pre-rope; the caller keeps the original."""
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        freqs_cis = self._freqs_cis[start_pos:start_pos + cutoff:ratio]
        kv = latent.clone()
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        kv = cache_qdq(kv, "nvfp4", 16)
        block_starts = torch.arange(start_pos, start_pos + cutoff, ratio, device=self._device)
        self.attn.scatter_compressed(
            self.layer_id, "attn", self.attn.compress_rows_of(ti, block_starts, ratio), kv[0],
        )

    def _pool(self, kv: torch.Tensor, score: torch.Tensor, dtype) -> torch.Tensor:
        blocks = kv.unflatten(1, (-1, self.compress_ratio))
        scores = score.unflatten(1, (-1, self.compress_ratio))
        return (blocks * scores.softmax(dim=2)).sum(dim=2)

    def forward(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot=None, ti: int = 0):
        """Cold prefill (start_pos == 0) or radix-hit extend (start_pos > 0). Returns the
        post-norm pre-rope latent covering the completed compressed positions, or None."""
        if start_pos > 0:
            return self.extend(x, start_pos, window_slots, int(tail_window_slot), ti)
        bsz, seqlen, _ = x.size()
        ratio, d = self.compress_ratio, self.head_dim
        dtype = x.dtype
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        xf = x.float()
        kv = self.wkv.forward(xf)
        if self.wgate is None:
            # Ratio 1: every token is its own compressed position.
            latent = self.norm.forward(kv.to(dtype))
            self._publish(latent, 0, seqlen, ti)
            return latent
        score = self.wgate.forward(xf)
        if remainder:
            # Seed the rolling carry with the incomplete tail (decode continuation).
            self._kv_state[:bsz, :remainder] = kv[:, cutoff:]
            self._score_state[:bsz, :remainder] = score[:, cutoff:]
        self._write_boundary_carries(x, seqlen, window_slots)
        if cutoff == 0:
            return None
        latent = self.norm.forward(self._pool(kv[:, :cutoff], score[:, :cutoff], dtype).to(dtype))
        self._publish(latent, 0, cutoff, ti)
        return latent

    def extend(self, x, start_pos: int, window_slots: torch.Tensor, tail_window_slot: int, ti: int = 0):
        """Carry-aware extend for new tokens [start_pos, start_pos+seqlen); the boundary is
        128-aligned (hence block-aligned), so the first new block starts clean."""
        assert start_pos % self.P == 0, "radix re-prefill boundary must be 128-aligned"
        bsz, seqlen, _ = x.size()
        ratio, d = self.compress_ratio, self.head_dim
        dtype = x.dtype
        end = start_pos + seqlen
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if self.wgate is None:
            kv = self.wkv.forward(x.float())
            latent = self.norm.forward(kv.to(dtype))
            self._publish(latent, start_pos, seqlen, ti)
            return latent
        self._seed_carry_from_ring(tail_window_slot)
        xf = x.float()
        kv = self.wkv.forward(xf)
        score = self.wgate.forward(xf)
        # The register after this extend: empty (blocks start clean at the aligned boundary),
        # plus the incomplete tail.
        self._kv_state[:bsz].zero_()
        self._score_state[:bsz].fill_(float("-inf"))
        if remainder:
            self._kv_state[:bsz, :remainder] = kv[:, cutoff:]
            self._score_state[:bsz, :remainder] = score[:, cutoff:]
        self._write_boundary_carries(x, seqlen, window_slots, start_pos=start_pos)
        if end % self.P != 0:
            self._write_through_carry(int(window_slots[-1].item()))
        if cutoff == 0:
            return None
        latent = self.norm.forward(self._pool(kv[:, :cutoff], score[:, :cutoff], dtype).to(dtype))
        self._publish(latent, start_pos, cutoff, ti)
        return latent

    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor, rows: torch.Tensor,
    ):
        """Batched single-token compressor update. Always scatters (rows that did not
        complete a block route to their own scratch row); returns ``(latent, should)``
        where ``latent`` is the pre-rope post-norm latent [B, d] (garbage on incomplete
        rows) and ``should`` [B] marks completed blocks. No host syncs (graph safe)."""
        B = x.size(0)
        ratio, d, rd = self.compress_ratio, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x2 = x.view(B, -1)
        if self.wgate is None:
            # Ratio 1: every token is its own compressed position; stateless.
            kv = bf16_linear_fp32(x2, self.wkv.weight)
            latent = self.norm.forward(kv.to(dtype))
            should = torch.ones(B, dtype=torch.bool, device=x.device)
        else:
            kv = bf16_linear_fp32(x2, self.wkv.weight).view(B, 1, self.item_size)
            score = bf16_linear_fp32(x2, self.wgate.weight).view(B, 1, self.item_size)
            idx_mod = pos % ratio
            should = ((pos + 1) % ratio == 0)
            block = self.attn.read_carry_blocks(self.layer_id, "attn", prev_window_slots, self.ring_size)
            ks = block[..., :self.item_size].clone()
            ss = block[..., self.item_size:].clone()
            ks.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, self.item_size), kv)
            ss.scatter_(1, idx_mod.view(B, 1, 1).expand(B, 1, self.item_size), score)
            compressed = gated_pool(ks, ss, dtype)
            self.attn.write_carry_blocks(
                self.layer_id, "attn", window_slots, self.ring_size, torch.cat([ks, ss], dim=-1)
            )
            latent = self.norm.forward(compressed)
        # Publish the (possibly discarded) compressed_kv: completed rows -> their arithmetic
        # row, incomplete rows -> their OWN scratch row. Snapshot-read, graph safe.
        kv_q = latent.view(B, 1, d).clone()
        if self.wgate is None:
            freqs_t = self._freqs_cis.index_select(0, pos)
        else:
            freqs_t = self._freqs_cis.index_select(0, (pos + 1 - ratio).clamp_min(0))
        apply_rotary_emb_decode(kv_q[..., -rd:], freqs_t)
        kv_q = cache_qdq(kv_q, "nvfp4", 16)
        dst = self.attn.decode_compress_rows(rows, pos, ratio, self.layer_id, "attn", should)
        self.attn.scatter_compressed(self.layer_id, "attn", dst, kv_q.view(B, -1))
        return latent, should

    # ----- carry ring plumbing (V4's non-overlap conventions; ape-free) -----
    def _write_through_carry(self, window_slot: int) -> None:
        if self._kv_state is None:
            return
        self.attn.write_carry(
            self.layer_id, "attn", window_slot, self.ring_size,
            torch.cat([self._kv_state[0], self._score_state[0]], dim=-1),
        )

    def _seed_carry_from_ring(self, window_slot: int) -> None:
        if self._kv_state is None:
            return
        block = self.attn.read_carry(self.layer_id, "attn", window_slot, self.ring_size)
        self._kv_state[0].copy_(block[:, :self.item_size])
        self._score_state[0].copy_(block[:, self.item_size:])

    def _write_boundary_carries(self, x, seqlen: int, window_slots, start_pos: int = 0) -> None:
        # Non-overlap: the carry at a 128-page boundary is the RESET state (the next block
        # starts clean at a page-aligned boundary). The tail write-through carries the
        # in-progress register. Ratio-1 has no ring at all.
        if self._kv_state is None:
            return
        self.attn.write_boundary_carries(
            layer_id=self.layer_id, tier="attn", ratio=self.compress_ratio,
            overlap=False, ring_size=self.ring_size, ape=None,
            kv=x.new_zeros(0), score=x.new_zeros(0), lo=start_pos,
            hi=start_pos + seqlen, window_slots=window_slots,
        )


class Indexer(BaseOP):
    """Lightning Indexer: mxfp4 q/k scores -> relu-weighted head reduction -> top-512
    blocks, hierarchical selection (candidates from layer 20, Reindex layers re-pick)."""

    def __init__(self, args, compress_ratio: int, *, owns_keys: bool, is_candidate_source: bool, quant_config=None, prefix: str = ""):
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio
        self.owns_keys = owns_keys
        self.is_candidate_source = is_candidate_source
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.wq_b = LinearReplicated(self.q_lora_rank, self.n_heads * self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.weights_proj = LinearReplicated(self.dim, self.n_heads, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.weights_proj")
        # bf16 keys-side weights, present only on the group's kv_source layer. The
        # indexer projects the compressor's latent (attention head_dim) to index keys.
        if owns_keys:
            self.wk = LinearReplicated(args.head_dim, args.index_head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wk")
            self.k_norm = RMSNorm(args.index_head_dim, args.norm_eps)
        self.softmax_scale = self.head_dim ** -0.5
        self.layer_id: int | None = None
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind_paged(self, pool, layer_id: int, freqs_cis, device) -> None:
        self.layer_id = layer_id
        self._freqs_cis = freqs_cis
        self._device = device

    # ----- key publication (kv_source layers only) -----
    def publish_keys(self, latent: torch.Tensor, start_pos: int, ti: int) -> None:
        """Prefill/extend: project the group's pooled latent to index keys and scatter to
        the idx pool rows [start_pos/ratio, (start_pos+cutoff)/ratio)."""
        assert self.owns_keys
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        keys = self.k_norm.forward(self.wk.forward(latent))
        cutoff = latent.size(1) * ratio
        freqs_cis = self._freqs_cis[start_pos:start_pos + cutoff:ratio]
        apply_rotary_emb(keys[..., -rd:], freqs_cis)
        keys = cache_qdq(keys, "mxfp4", 32)
        block_starts = torch.arange(start_pos, start_pos + cutoff, ratio, device=self._device)
        self.attn.scatter_compressed(
            self.layer_id, "idx", self.attn.compress_rows_of(ti, block_starts, ratio), keys[0],
        )

    def publish_key_decode(
        self, latent: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, should: torch.Tensor,
    ) -> None:
        B = latent.size(0)
        rd = self.rope_head_dim
        keys = self.k_norm.forward(self.wk.forward(latent))
        apply_rotary_emb_decode(keys[..., -rd:], self._freqs_cis.index_select(0, pos))
        keys = cache_qdq(keys, "mxfp4", 32)
        dst = self.attn.decode_compress_rows(rows, pos, self.compress_ratio, self.layer_id, "idx", should)
        self.attn.scatter_compressed(self.layer_id, "idx", dst, keys.view(B, -1))

    # ----- selection -----
    def _select_candidate_blocks(self, scores: torch.Tensor, visible_lengths: torch.Tensor) -> torch.Tensor:
        """Layer 20's candidate block mask: block scores = max over block_size positions,
        the newest live block forced in, top-k blocks kept -> per-position bool mask."""
        bs = self.candidate_block_size
        width = scores.shape[-1]
        padded = torch.nn.functional.pad(scores, (0, (-width) % bs), value=float("-inf"))
        block_scores = padded.unflatten(-1, (-1, bs)).amax(-1)
        last_block = (visible_lengths - 1) // bs
        block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
        block_scores = block_scores.masked_fill(block_ids == last_block, float("inf"))
        topk = min(self.candidate_topk_blocks, block_scores.shape[-1])
        selected = block_scores.topk(topk, dim=-1)
        keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(
            -1, selected.indices, selected.values > float("-inf")
        )
        return keep.repeat_interleave(bs, dim=-1)[..., :width]

    def _pick_and_sort(self, scores: torch.Tensor) -> torch.Tensor:
        """Top-k with the reference's conventions: ascending sorted picks, invalid -> -1,
        then -1s partitioned to the END (the paged kernel's per-row count bounds a prefix)."""
        picks = scores.topk(min(self.index_topk, scores.shape[-1]), dim=-1)[1]
        picks = picks.sort(dim=-1)[0]
        valid = torch.isfinite(scores.gather(-1, picks))
        picks = torch.where(valid, picks, picks.new_full((), -1))
        # Stable partition: non-negative picks keep their relative order, -1s trail.
        reorder = (picks < 0).to(torch.uint8).argsort(dim=-1, stable=True)
        return picks.gather(-1, reorder)

    def _scores_prefill(self, x, qr, start_pos: int, ti: int, state: CSA2State):
        """One prefill segment's scores [1, n, width] over this group's keys."""
        ratio, rd = self.compress_ratio, self.rope_head_dim
        n = x.size(1)
        end = start_pos + n
        freqs_cis = self._freqs_cis[start_pos:end]
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = cache_qdq(q, "mxfp4", 32)
        weights = self.weights_proj.forward(x) * (self.softmax_scale * self.n_heads ** -0.5)
        keys_layer = self.layer_id if self.owns_keys else state.keys_layer
        keys = self.attn.indexer_keys(ti, end // ratio, ratio, keys_layer, 1)
        return self.attn.indexer_prefill_logits(q, keys, weights)

    def forward(self, x, qr, start_pos: int, seg_idx: int, ti: int, state: CSA2State):
        """One prefill segment: returns ``(topk_blocks [1, n, k], counts)``. The candidate
        source ALSO appends its per-segment mask to ``state`` (segment-ordered list)."""
        scores = self._scores_prefill(x, qr, start_pos, ti, state)
        n = scores.shape[1]
        live = ((start_pos + torch.arange(1, n + 1, device=scores.device)) // self.compress_ratio).unsqueeze(1)
        blk = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores + torch.where(blk >= live, float("-inf"), 0)
        if self.is_candidate_source:
            cands = self._select_candidate_blocks(scores, live)
            lst = state.candidates if isinstance(state.candidates, list) else []
            lst.append(cands)
            state.candidates = lst
        elif state.candidates is not None:
            # Reindex layers (24+) pick within layer 20's candidate set; the pre-20
            # index sources (2 / 8 / 14) score the full block range.
            scores = scores.masked_fill(~state.candidates[seg_idx], float("-inf"))
        picks = self._pick_and_sort(scores)
        return picks, (picks >= 0).sum(-1, dtype=torch.int32)

    def decode_step(self, x, qr, pos, rows, n_stage: int, state: CSA2State):
        """Batched single-token selection over the group's keys. Returns
        ``(topk_blocks [B, 1, k], counts)``; the candidate source also updates ``state``."""
        B = x.size(0)
        ratio, rd = self.compress_ratio, self.rope_head_dim
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb_decode(q[..., -rd:], self._freqs_cis.index_select(0, pos))
        q = cache_qdq(q, "mxfp4", 32)
        weights = self.weights_proj.forward(x) * (self.softmax_scale * self.n_heads ** -0.5)
        valid = (pos + 1) // ratio
        keys_layer = self.layer_id if self.owns_keys else state.keys_layer
        scores = self.attn.indexer_decode_scores(
            q.reshape(B, self.n_heads, self.head_dim), weights.reshape(B, self.n_heads),
            valid, n_stage, ratio, keys_layer,
        ).view(B, 1, n_stage)
        if self.is_candidate_source:
            state.candidates = self._select_candidate_blocks(scores, valid.view(B, 1, 1))
        elif state.candidates is not None:
            scores = scores.masked_fill(~state.candidates, float("-inf"))
        picks = self._pick_and_sort(scores)
        return picks, (picks >= 0).sum(-1, dtype=torch.int32)
