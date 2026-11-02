"""CSA2 attention over the paged DSV4 pools: 128-sliding window + the ratio group's
compressed top-k (window-first slot order, single paged launch). Owns the per-layer
Compressor / Indexer instances and the ratio-group state handoff.

Differences from V4's attention: no post-``wq_b`` head RMSNorm (the checkpoint's
``q_norm`` is on the 1280 latent only), the window KV round-trip is fp8 over the WHOLE
head_dim in 32-channel groups (rope channels included), the compressed tier is read
from the group's kv_source pool (``state.cmp_layer``), and the sparse picks come from
the CSA2 state machine (indexer re-runs / candidate blocks) instead of a per-layer
compressor+indexer pair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.dsv4.norm import rms_norm
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, LinearRowParallel, RMSNorm
from freetoken.utils import div_even

from ..deepseek_v4.layers import get_window_topk_idxs
from ..deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode, get_freqs_cis
from .compress import CSA2State, Compressor, Indexer
from .quant import cache_qdq


class Attention(BaseOP):
    def __init__(self, layer_id: int, args, *, quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        # The compressed tier + top-k of every downstream layer of the same ratio group
        # come from this layer; ratio-0 layers have no group.
        self.kv_source_layer = (
            max(s for s in args.kv_source_layer_ids if s <= layer_id and args.compress_ratios[s] == self.compress_ratio)
            if self.compress_ratio else None
        )

        # TP: latent projections and KV stay replicated; heads (wq_b / attn_sink) and output
        # groups (wo_a / wo_b) are sharded. Heads are group-contiguous (group g = heads
        # [g*heads_per_group, ...)), so a rank's head shard must not split a group.
        tp = get_tp_info()
        self.n_heads_local = div_even(self.n_heads, tp.size)
        self.n_groups_local = div_even(self.n_groups, tp.size)
        heads_per_group = self.n_heads // self.n_groups
        if self.n_heads_local % heads_per_group:
            raise ValueError(
                f"TP size {tp.size} splits the {self.n_heads} V4.1 heads at {self.n_heads_local} "
                f"per rank, which is not a multiple of the {heads_per_group} heads per output group"
            )

        self.attn_sink = torch.empty(self.n_heads_local, dtype=torch.float32)
        self.wq_a = LinearReplicated(self.dim, self.q_lora_rank, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_a")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = LinearColParallelMerged(self.q_lora_rank, [self.n_heads * self.head_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wq_b")
        self.wkv = LinearReplicated(self.dim, self.head_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a is one [o_lora_rank, K] matrix per output group, stacked on N and applied as a
        # bmm; the reference dequantizes it to bf16 and so does the reader. Under TP it shards
        # on N by group like wo_b shards on K.
        wo_a_rows = self.n_groups_local * args.o_lora_rank
        wo_a_k = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(self.n_groups * args.o_lora_rank, self.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wo_b")
        self.softmax_scale = self.head_dim ** -0.5

        if self.is_kv_source:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim, quant_config=quant_config, prefix=f"{prefix}.compressor")
        else:
            self.compressor = None
        if self.is_index_source:
            self.indexer = Indexer(
                args, self.compress_ratio,
                owns_keys=self.is_kv_source, is_candidate_source=self.is_candidate_source,
                quant_config=quant_config, prefix=f"{prefix}.indexer",
            )
        else:
            self.indexer = None

        # ratio>0 layers rope in the compressed regime (yarn @ compress_rope_theta);
        # ratio-0 layers use the plain table.
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, pool, device: torch.device) -> None:
        L = self.layer_id
        self.P = pool.P
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)
        if self.compressor is not None:
            self.compressor.bind_paged(pool, L, self._freqs_cis, device)
        if self.indexer is not None:
            self.indexer.bind_paged(pool, L, self._freqs_cis, device)

    def reset(self) -> None:
        if self.compressor is not None:
            self.compressor.reset()

    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups_local, -1)
        wo_a = self.wo_a.view(self.n_groups_local, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(o)

    def _qkv(self, x: torch.Tensor, freqs, decode: bool):
        """Shared q/kv projections: latent-normalized q (no per-head norm in V4.1) and the
        window KV with the reference's whole-head fp8 round-trip."""
        rd = self.rope_head_dim
        rope = apply_rotary_emb_decode if decode else apply_rotary_emb
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads_local, self.head_dim))
        rope(q[..., -rd:], freqs)
        kv = self.kv_norm.forward(self.wkv.forward(x))
        rope(kv[..., -rd:], freqs)
        kv = cache_qdq(kv, "fp8", 32)
        return qr, q, kv

    def _window_picks(self, ti: int, start_pos: int, n: int, slots: torch.Tensor):
        """The per-query 128-sliding window as GLOBAL pool slots (cold: natural width,
        extend: the sliding span)."""
        win, device = self.window_size, slots.device
        end = start_pos + n
        if start_pos == 0:
            win_cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            return self.attn.win_cols_to_global(win_cols, slots)
        w_lo = max(0, start_pos - win + 1)
        ws_pool = self.attn.window_slots_of(ti, w_lo, end)
        abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
        cand = (abs_p - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
        win_cols = torch.where(cand > abs_p, -1, cand - w_lo).unsqueeze(0)
        return self.attn.win_cols_to_global(win_cols, ws_pool)

    def forward_ragged(self, x, segments, flat_positions, state: CSA2State):
        """Ragged batched prefill (cu_seqlens). ``x`` is [1, T, dim]; per-request segments
        tile the flat stream. The group's kv_source layer computes/records the state
        (compressed_kv pool, indexer keys, top-k, candidates) inside its own segment loop;
        consumers read ``state``. Returns the layer output [1, T, dim]."""
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        _, T, _ = x.size()
        if len(segments) == 1:
            freqs = self._freqs_cis[segments[0][3]:segments[0][3] + T]
        else:
            freqs = self._freqs_cis.index_select(0, flat_positions)

        qr, q, kv = self._qkv(x, freqs, decode=False)

        if self.is_kv_source:
            state.cmp_layer = self.layer_id
            state.keys_layer = self.layer_id
            state.ratio = ratio
        win_parts: list[torch.Tensor] = []
        cmp_parts: list[torch.Tensor] = []
        topk_parts: list[torch.Tensor] = []
        for seg_idx, (off, n, ti, start_pos) in enumerate(segments):
            end = start_pos + n
            slots = self.attn.window_slots_of(ti, start_pos, end)
            self.attn.store_window(kv[0, off:off + n], self.layer_id, slots)
            win_global = self._window_picks(ti, start_pos, n, slots)
            win_parts.append(win_global)
            if not ratio:
                continue
            if self.is_kv_source:
                # Only the compressor/indexer read the matched tail page's slot; the host
                # sync (.item()) is paid after the ratio-0 layers never reach here.
                tail_ws = (
                    int(self.attn.window_slots_of(ti, start_pos - 1, start_pos).item())
                    if start_pos > 0 else None
                )
                if start_pos == 0:
                    self.compressor.reset()
                latent = self.compressor.forward(
                    x[:, off:off + n], start_pos, slots, tail_window_slot=tail_ws, ti=ti
                )
                if self.indexer is not None and latent is not None:
                    self.indexer.publish_keys(latent, start_pos, ti)
            if self.is_index_source:
                blocks, _counts = self.indexer.forward(
                    x[:, off:off + n], qr[:, off:off + n], start_pos, seg_idx, ti, state
                )
            else:
                blocks = state.topk_indices[:, off:off + n]
            topk_parts.append(blocks)
            cmp_parts.append(self.attn.blocks_to_global(blocks, ratio, ti=ti))

        if self.is_index_source and ratio:
            # picks are min(index_topk, that request's block count) wide, so a batch of
            # short and long prompts needs the -1 pad (the kernel masks -1) before the cat
            if len(topk_parts) == 1:
                state.topk_indices = topk_parts[0]
            else:
                max_k = max(t.shape[-1] for t in topk_parts)
                parts = [t if t.shape[-1] == max_k else F.pad(t, (0, max_k - t.shape[-1]), value=-1) for t in topk_parts]
                state.topk_indices = torch.cat(parts, dim=1)
        if ratio:
            max_c = max(c.shape[-1] for c in cmp_parts)
            flat = []
            for i, (_off, _n, _ti, _start) in enumerate(segments):
                wg = win_parts[i]
                if wg.shape[-1] < win:
                    wg = F.pad(wg, (0, win - wg.shape[-1]), value=-1)
                cg = cmp_parts[i]
                if cg.shape[-1] < max_c:
                    cg = F.pad(cg, (0, max_c - cg.shape[-1]), value=-1)
                flat.append(torch.cat([wg, cg], dim=-1))
            n_window = win
        else:
            # ratio-0: bs==1 keeps the natural width (see V4's padding note), bs>1 pads to win.
            n_window = win if len(segments) > 1 else win_parts[0].shape[-1]
            flat = []
            for i, (_off, _n, _ti, _start) in enumerate(segments):
                wg = win_parts[i]
                if wg.shape[-1] < n_window:
                    wg = F.pad(wg, (0, n_window - wg.shape[-1]), value=-1)
                flat.append(wg)
        topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=bool(ratio), cmp_layer_id=state.cmp_layer,
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, 1, T)

    def decode_step(self, x, pos, rows, cmp_stage_cap: int, wctx, state: CSA2State):
        """Batched single-token attention (EAGER/graph). ``wctx`` is the layer-invariant
        window context hoisted by the model; ``state`` carries the ratio group's picks."""
        B = x.size(0)
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        n_cmp_stage = (cmp_stage_cap + 1) // ratio if ratio else 0
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        freqs_t = self._freqs_cis.index_select(0, pos)

        qr, q, kv = self._qkv(x, freqs_t, decode=True)
        self.attn.store_window(kv.view(B, -1), self.layer_id, window_slots)

        cmp_counts = None
        if ratio:
            if self.is_kv_source:
                # same state contract as prefill: consumers address this group's
                # kv_source pools through state (cmp_layer/keys_layer/ratio)
                state.cmp_layer = self.layer_id
                state.keys_layer = self.layer_id
                state.ratio = ratio
                latent, should = self.compressor.decode_step(
                    x, pos, prev_window_slots, window_slots, rows
                )
                if self.indexer is not None:
                    self.indexer.publish_key_decode(latent, pos, rows, should)
            if self.is_index_source:
                blocks, counts = self.indexer.decode_step(
                    x, qr, pos, rows, n_cmp_stage, state
                )
                # same contract as prefill: the group's picks are threaded through
                # state for the ratio layers without their own indexer
                state.topk_indices = blocks
            else:
                blocks = state.topk_indices
                counts = (blocks >= 0).sum(-1, dtype=torch.int32)
            compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
            topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
            # Valid picks are a PREFIX of each row's list (-1s partitioned last by the
            # selector), so one device-side count per row bounds the kernel's loop.
            cmp_counts = counts
        else:
            topk_idxs = window_slots_topk
        topk_idxs = topk_idxs.int()

        o = self.attn.attend(
            q, self.layer_id, topk_idxs, win, self.attn_sink, self.softmax_scale,
            cmp_counts=cmp_counts, has_compression=bool(ratio), cmp_layer_id=state.cmp_layer,
        )
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
        return self._wo(o, B, 1)
