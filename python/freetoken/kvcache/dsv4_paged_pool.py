"""DSV4 paged KV pools (sglang-style management, FreeToken byte layout).

Four buffer families, sized from a budget (the cost model) not ``num_requests``:

* ``window_pool[L]``  -- all layers; the 128-sliding KV ring, page-granular.
* ``cmp_pool[L]``     -- ratio>0 layers; compressed KV, per-block.
* ``idx_pool[L]``     -- ratio==4 layers; Lightning-Indexer compressed keys.
* ``state_ring[L]``   -- ratio>0 layers; the per-window-page compress-state ring
  (fp32, ``kv|score`` split), with index ``-1`` a permanent scratch slot.

The KV/compressed/indexer pools are bf16 (the fp8/fp4 quant is an in-place round-trip already
baked into the bf16 value, so ``index_select`` staging is byte-exact); only the compress-state
ring is fp32.

``state_loc`` is DERIVED from a window slot, never stored:
    state_loc = where(ws < 0, -1, (ws // P) * ring_size + ws % ring_size)
``ring_size | P`` so distinct pages map to disjoint ring blocks.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

from .base import BaseKVCachePool
from .dsv4_cost_model import (
    DSV4PoolSizes,
    dsv4_kv_unit_bytes,
    dsv4_window_unit_bytes,
    ring_size_for_ratio,
)


logger = init_logger(__name__)


# LIFO free-list allocator for the window tier: one allocated unit spans ``page_unit = P``
# slots, so a unit base is always a multiple of ``P`` (the spec's ``G*P`` page-base
# invariant). The cmp/idx tiers have no allocator -- their rows are ``full_loc // ratio``
# pure arithmetic.
class FreeListAllocator:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        page_unit: int = 1,
    ) -> None:
        assert capacity % page_unit == 0, (
            f"capacity {capacity} must be a multiple of page_unit {page_unit}"
        )
        self._capacity = int(capacity)
        self._page_unit = int(page_unit)
        self._device = device
        # Unit base slots: [0, page_unit, 2*page_unit, ...]. LIFO -> pop the tail.
        self._free = self._fresh_free()

    def _fresh_free(self) -> torch.Tensor:
        n_units = self._capacity // self._page_unit
        return torch.arange(n_units, dtype=torch.int64, device=self._device) * self._page_unit

    def alloc(self, n_units: int) -> torch.Tensor:
        """Return ``n_units`` unit base slots (each a multiple of ``page_unit``)."""
        if n_units < 0:
            raise ValueError(f"n_units must be non-negative, got {n_units}")
        if n_units > self._free.numel():
            raise RuntimeError(
                f"FreeListAllocator out of slots: requested {n_units} units, "
                f"have {self._free.numel()} (capacity {self._capacity}, unit {self._page_unit})"
            )
        if n_units == 0:
            return self._free[:0].clone()
        taken = self._free[-n_units:].clone()
        self._free = self._free[:-n_units]
        return taken

    def free(self, units: torch.Tensor) -> None:
        """Return previously-allocated unit base slots to the pool (LIFO recycle)."""
        if units.numel() == 0:
            return
        units = units.to(device=self._device, dtype=torch.int64).reshape(-1)
        self._free = torch.cat([self._free, units])

    def available(self) -> int:
        """Free capacity in slots (units * page_unit)."""
        return int(self._free.numel()) * self._page_unit

    @property
    def capacity(self) -> int:
        return self._capacity

    def reset(self) -> None:
        self._free = self._fresh_free()



class CompressStateRing:
    """Per-layer fp32 compress-state ring: ``[n_slots + 1, 2*(1+overlap)*head_dim]``.

    Last row (index ``-1``) is a permanent scratch slot, re-cleared on every
    write. Last dim is split ``kv | score``; ``set_state`` writes both halves.
    """

    def __init__(
        self,
        n_slots: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.ring_size = ring_size
        self.head_dim = head_dim
        self._item_size = (1 + int(overlap)) * head_dim  # width of kv (== width of score)
        last_dim = 2 * self._item_size
        # +1 trailing scratch row at index -1.
        self.buffer = torch.zeros((n_slots + 1, last_dim), dtype=dtype, device=device)
        self.n_slots = n_slots
        self._clear_scratch()

    def _clear_scratch(self) -> None:
        self.buffer[-1, : self._item_size].zero_()
        self.buffer[-1, self._item_size :].fill_(float("-inf"))

    @property
    def item_size(self) -> int:
        return self._item_size

    def get(self, state_loc: torch.Tensor) -> torch.Tensor:
        """Gather ``kv_score`` rows at ``state_loc`` (``-1`` -> scratch row)."""
        return self.buffer[state_loc]

    def set(self, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        """Scatter ``kv_score`` rows to ``state_loc`` then re-clear the scratch row."""
        self.buffer[state_loc] = kv_score
        self._clear_scratch()

    def get_blocks(self, page_base: torch.Tensor) -> torch.Tensor:
        """Batched per-row carry-block read. ``page_base`` is ``[B]`` (the ring block base
        row per row, == ``(window_slot // P) * ring_size``). Returns ``[B, ring_size, 2*item]``
        — each row's whole rolling carry block. Distinct pages -> disjoint blocks (isolation)."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        return self.buffer[rows]

    def set_blocks(self, page_base: torch.Tensor, blocks: torch.Tensor) -> None:
        """Batched per-row carry-block write: ``buffer[base+arange(ring_size)] = blocks[row]``.
        ``page_base`` ``[B]``, ``blocks`` ``[B, ring_size, 2*item]``. Re-clears scratch."""
        rows = page_base[:, None] + torch.arange(self.ring_size, device=page_base.device)
        self.buffer[rows] = blocks
        self._clear_scratch()


class DSV4PagedKVCache(BaseKVCachePool):
    # DSV4 has a dedicated restore-segment adapter; the generic per-token tier does not apply.
    supports_prefix_tier: bool = False
    def __init__(
        self,
        sizes: DSV4PoolSizes,
        args,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        P: int = 128,
        n_scratch: int = 1,
    ) -> None:
        assert dtype == torch.bfloat16, "KV pools are bf16 (fp4/fp8 is an in-place round-trip)"
        self.args = args
        self.sizes = sizes
        self._device = device
        self._dtype = dtype
        self.P = P
        self._n_layers = args.n_layers
        self.head_dim = args.head_dim
        self.index_head_dim = args.index_head_dim
        # The checkpoint can ship one extra trailing ratio (44 for 43 layers); the model only uses
        # the first n_layers, so truncate to match.
        self.compress_ratios = tuple(args.compress_ratios)[: self._n_layers]
        assert len(self.compress_ratios) == self._n_layers
        # Scratch rows appended to each cmp/idx pool tensor BEYOND the allocator's capacity
        # (never handed out): batched decode routes each row whose compressed block did NOT
        # complete this step to its own scratch row ``cmp_scratch_base + row`` (a discarded
        # write), so the masked per-row scatter is graph-safe (no host sync, no -1 index, no
        # cross-row collision). One per running request row.
        self.n_scratch = int(n_scratch)

        # Logical full-loc currency: ONE mapping from the virtual full-token index space to window
        # slots; cmp/idx rows derive arithmetically (full_loc // ratio), the state ring off the
        # window slot. Unmapped = -1. The trailing row is a PERMANENT -1 sentinel, so a fancy-index
        # gather at -1 lands on it and returns -1 (gather-safe); scatter paths must never see a
        # negative (index_copy_ raises OOB, it does not wrap).
        for ratio in set(self.compress_ratios):
            assert ratio == 0 or P % ratio == 0, f"P={P} must be divisible by ratio {ratio}"
        self._paged_params: tuple[int, bool] | None = None  # (_init_paged_state args, for rebuild)
        self._alloc_buffers()

    def _alloc_buffers(self) -> None:
        """(Re)allocate every physical buffer for the CURRENT ``self.sizes``. Shared by __init__
        and the in-place ``rebuild`` (identity-preserving, like HybridSWAKVCache.rebuild -- the
        CacheManager/engine/ctx all hold THIS object)."""
        sizes, device, dtype = self.sizes, self._device, self._dtype
        self.cmp_scratch_base = []
        self.idx_scratch_base = []
        self.full_to_window = torch.full(
            (sizes.full_token + 1,), -1, dtype=torch.int64, device=device
        )
        # Pinned staging buffers for host gathers, recycled through a free list: an
        # async put's pinned copy may still be in flight (its consumer clones the
        # payload on the writer thread) when the next put is issued, so buffers must
        # not be reused until the consumer releases them.
        self._pinned_free: list[torch.Tensor] = []
        self._event_free: list[torch.cuda.Event] = []
        self._gpu_pack_bufs: dict[int, torch.Tensor] = {}
        # Per-shape pinned staging for the synchronous per-slice gather (the packed
        # single-copy variant wedged the driver; see gather_prefix_pages).
        self._pinned_bufs: dict[tuple, torch.Tensor] = {}
        # Consumed gather packs queued by the writer for the SCHEDULER to recycle:
        # freeing a pinned tensor touches PyTorch's CachingHostAllocator, which
        # submits CUDA calls (cudaEventCreate/Record for every stream recorded on
        # the storage -- the non_blocking D2H records the gather stream). A CUDA
        # submission from the writer deadlocks the driver against the engine's own
        # submissions (observed five times: engine wedged in cuMemcpyHtoDAsync while
        # the writer blocked in CachingHostAllocator free -> cuEventCreate).
        self._pack_recycle: list[dict] = []

        # The ONE slot map (table_idx, pos) -> full loc: the shared page_table, attached by the
        # engine policy. Window slots come from ``full_to_window``; cmp/idx rows are arithmetic.
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map: torch.Tensor | None = None

        # Window KV: every layer.
        self.window_pool: list[torch.Tensor] = [
            torch.zeros(sizes.n_win_slots, self.head_dim, device=device, dtype=dtype)
            for _ in range(self._n_layers)
        ]

        # Compressed KV / Indexer KV / compress-state ring: per ratio-class.
        # ``state_ring`` is the ATTENTION compressor's ring (head_dim). Ratio-4
        # layers additionally own an ``indexer_state_ring`` (index_head_dim) for
        # the indexer's own compressor -- a separate pool, no collision.
        self.cmp_pool: list[torch.Tensor | None] = []
        self.idx_pool: list[torch.Tensor | None] = []
        self.state_ring: list[CompressStateRing | None] = []
        self.indexer_state_ring: list[CompressStateRing | None] = []
        for L in range(self._n_layers):
            ratio = self.compress_ratios[L]
            if ratio == 0:
                self.cmp_pool.append(None)
                self.idx_pool.append(None)
                self.state_ring.append(None)
                self.indexer_state_ring.append(None)
                self.cmp_scratch_base.append(None)
                self.idx_scratch_base.append(None)
                continue

            self.cmp_scratch_base.append(sizes.cmp_blocks[L])
            self.cmp_pool.append(
                torch.zeros(
                    sizes.cmp_blocks[L] + self.n_scratch, self.head_dim, device=device, dtype=dtype
                )
            )
            if ratio == 4:
                self.idx_scratch_base.append(sizes.idx_blocks[L])
                self.idx_pool.append(
                    torch.zeros(
                        sizes.idx_blocks[L] + self.n_scratch,
                        self.index_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                )
                # Indexer compressor ring: index_head_dim, overlap, ring_size=8.
                self.indexer_state_ring.append(
                    CompressStateRing(
                        n_slots=sizes.idx_state_slots[L],
                        ring_size=ring_size_for_ratio(4),
                        overlap=True,
                        head_dim=self.index_head_dim,
                        device=device,
                    )
                )
            else:
                self.idx_pool.append(None)
                self.indexer_state_ring.append(None)
                self.idx_scratch_base.append(None)

            self.state_ring.append(
                CompressStateRing(
                    n_slots=sizes.state_slots[L],
                    ring_size=ring_size_for_ratio(ratio),
                    overlap=(ratio == 4),
                    head_dim=self.head_dim,
                    device=device,
                )
            )

    # ----- full-loc translation (gather-only -1 safety; see field comment) -----
    def translate_full_to_window(self, full_locs: torch.Tensor) -> torch.Tensor:
        # int64 gather indices: the shared page_table stores full locs as int32.
        return self.full_to_window[full_locs.to(dtype=torch.int64)]

    @staticmethod
    def cmp_rows(full_locs: torch.Tensor, ratio: int) -> torch.Tensor:
        return torch.div(full_locs.to(dtype=torch.int64), ratio, rounding_mode="floor")

    def bind_window_pages(self, full_page_base: int, window_page_base: int) -> None:
        assert full_page_base % self.P == 0 and window_page_base % self.P == 0
        self.full_to_window[full_page_base : full_page_base + self.P] = torch.arange(
            window_page_base, window_page_base + self.P, dtype=torch.int64, device=self._device
        )

    def unbind_window_pages(self, full_locs: torch.Tensor) -> None:
        self.full_to_window[full_locs[full_locs >= 0]] = -1

    # ----- generic swa_pool duck-type (the CacheManager plug-in surface) -----
    # ShadowRadix layering: the shared page_table is the virtual full-token coordinate; this pool
    # projects it into the physical tiers. The window tier is the managed "second currency" --
    # token-face signatures (what the generic CacheManager speaks), PAGE-ATOMIC internals (window
    # pages are 1:1 page-bound to full pages; the per-page state ring requires it). alloc_swa
    # receives whole ascending pages (allocate_paged's _page_to_token expansion) and free paths
    # are page-complete by construction (padded finish tails, align_down frontiers, page-aligned
    # tree nodes) -- asserted here, not assumed.
    swa_paged = True

    @property
    def sliding_window_size(self) -> int:
        return self.P

    @property
    def prefill_chunk_budget(self) -> int:
        return self._chunk_budget

    # ----- engine-facing rebuild surface -----
    # The tier buffers are bound into per-forward model scratch, invalid after a realloc.
    needs_rebind_on_rebuild = True

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_auto_cost_model

        dsv4_args = config.model_config.dsv4_args
        P = dsv4_args.window_size
        floor = _dsv4_window_floor_pages(config, P)
        per_page, fixed, min_reserve_tokens = dsv4_auto_cost_model(
            dsv4_args, _dsv4_swa_ratio(config), floor, P=P, n_scratch=config.max_running_req + 1
        )
        return per_page, fixed, config.page_size, min_reserve_tokens

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        # Solve the largest budget-respecting anchor with the exact per-tier byte model.
        # num_pages is in P (window) units and anchors full_token = num_pages*P (the FULL
        # cmp/idx tiers). The window working-set floor is honored in PAGES and the total is
        # byte-checked here. A budget too small for even the minimal working set raises a
        # graceful config error, not a late OOM.
        from freetoken.utils import mem_GB

        from .dsv4_cost_model import _dsv4_pool_sizes, _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_pool_bytes, dsv4_solve_num_pages

        dsv4_args = config.model_config.dsv4_args
        P = dsv4_args.window_size
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = dsv4_solve_num_pages(
                available_memory, dsv4_args, _dsv4_swa_ratio(config),
                floor_win_pages=_dsv4_window_floor_pages(config, P), P=P,
                n_scratch=config.max_running_req + 1,
            )
            # The solver fits PHYSICAL pages to memory; one is the dummy page, so the
            # usable (advertised) count is one less.
            num_pages = sizes.full_token // P - 1
        else:
            # Fail at config time with guidance: a below-floor pool would otherwise boot
            # (dsv4_pool_sizes caps the window at num_pages) and die at runtime alloc.
            floor = _dsv4_window_floor_pages(config, P)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the DSV4 "
                    f"window working-set floor {floor} pages ({floor * P} tokens); raise "
                    f"--num-pages or lower max_running_req/max_seq_len"
                )
            sizes = _dsv4_pool_sizes(config, num_pages + 1)  # +1 for dummy page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real = dsv4_pool_bytes(sizes, dsv4_args, config.max_running_req + 1)
        logger.info(
            f"Allocating {num_pages * P} tokens for DSV4 KV cache "
            f"({sizes.n_win_pages} window pages), total = {mem_GB(real)}"
        )
        return num_pages

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # The full anchor must cover the window working-set floor (full >= window always), so
        # that floor -- the value validate_rebuild enforces -- is the pool's floor in tokens.
        from .dsv4_cost_model import _dsv4_window_floor_pages

        P = config.model_config.dsv4_args.window_size
        return _dsv4_window_floor_pages(config, P) * P

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "",
        num_swa_pages: int | None = None, **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from .base import CacheRebuildRejected
        from .dsv4_cost_model import _dsv4_pool_sizes, _dsv4_window_floor_pages
        from .dsv4_cost_model import dsv4_pool_bytes

        dsv4_args = config.model_config.dsv4_args
        if num_pages is not None:
            floor = _dsv4_window_floor_pages(config, dsv4_args.window_size)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the DSV4 window working-set floor {floor} "
                    f"(max_running_req={config.max_running_req}); admission would deadlock"
                )
        if num_pages is not None or num_swa_pages is not None:
            # Size the pool a KV/window rebuild would build: the target anchor (or current) with
            # the target window (or current), computed BEFORE the config is mutated.
            target_pages = num_pages if num_pages is not None else current_num_pages
            kv_sizes = _dsv4_pool_sizes(
                config, target_pages + 1, num_swa_pages=num_swa_pages
            )  # +1 for dummy page
        else:
            # MoE-only rebuild keeps the CURRENT pool: budget-check against its live sizes
            # (reflects DSV4_FORCE_SMALL_POOL and the physical dummy page).
            kv_sizes = self.sizes
        # The rebuilds are free-before-alloc, so the whole budget is available (no fixed
        # cache term); an unfit request must still reject BEFORE the teardown.
        budget = net_cache_budget_bytes(config.memory_ratio, baseline_free, weights_bytes, 0)
        need = target_moe * per_expert_bytes + dsv4_pool_bytes(
            kv_sizes, dsv4_args, config.max_running_req + 1
        )
        if need > budget:
            kv_part = f"kv={num_pages} P-pages" if num_pages is not None else "kv=current pool"
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, {kv_part}) needs "
                f"{mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        from .dsv4_cost_model import _dsv4_pool_sizes

        # +1 for the dummy page
        self.rebuild(_dsv4_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        # The model reads full locs through full_loc_map; under the shared route that IS the
        # page table. The attention backend re-allocates its decode snapshot on re-capture.
        self.full_loc_map = page_table

    def unit_bytes(self) -> tuple[int, int]:
        # No measurable flat buffer (owned paged pool): the full (cmp/idx + mapping) and window
        # (sliding KV + state rings) per-token costs come from the per-tier cost model.
        return dsv4_kv_unit_bytes(self.args, self.P), dsv4_window_unit_bytes(self.args, self.P)

    def rebuild(self, sizes) -> None:
        """In-place resize to ``sizes`` (identity-preserving; free-before-alloc). The manager's
        tree/page bookkeeping reset is the scheduler's generic cache_manager.rebuild; the engine
        re-attaches the page table via attach_page_table afterwards."""
        import gc

        assert self._paged_params is not None, "rebuild before _init_paged_state"
        self.sizes = sizes
        self.window_pool = self.cmp_pool = self.idx_pool = None
        self.state_ring = self.indexer_state_ring = None
        self.full_to_window = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._alloc_buffers()
        self._init_paged_state(*self._paged_params)

    def _init_paged_state(self, max_running_req: int, radix: bool) -> None:
        """Build the pool-owned window free-list + the tail dummy binding. The LAST full page and
        LAST window page are the reserved dummy region: page_table's dummy row points at
        ``full_token - P`` (== the generic ``fill_(num_tokens)`` convention with num_tokens = the
        allocatable token count), permanently bound so graph-padded rows scatter to a real slot."""
        from .dsv4_cost_model import dsv4_reserved_window_pages
        P = self.P
        self._paged_params = (int(max_running_req), bool(radix))
        self.full_to_window.fill_(-1)
        self._win_alloc = FreeListAllocator(self.sizes.n_win_slots - P, self._device, page_unit=P)
        self.bind_window_pages(self.sizes.full_token - P, self.sizes.n_win_slots - P)
        # Chunk cap: a batched prefill holds the whole chunk's window live at once (sliding frees
        # only between chunks; peak ~2x the chunk), so reserve the concurrent working set and
        # halve the rest -- the same formula the bespoke manager used.
        n_win_pages = (self.sizes.n_win_slots // P) - 1
        reserved = dsv4_reserved_window_pages(max_running_req, radix)
        self._chunk_budget = max(P, (n_win_pages - reserved) // 2 * P)

    @property
    def swa_num_tokens(self) -> int:
        # Allocatable window slots + 1: the generic capacity convention reserves slot 0 as a
        # sentinel (cap == swa_num_tokens - 1); DSV4's reserved unit is the tail dummy page,
        # already excluded from the free-list, so +1 re-encodes the same cap.
        return (self.sizes.n_win_slots - self.P) + 1

    def swa_available_size(self) -> int:
        return int(self._win_alloc.available())

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        """Bind one window page per incoming FULL page. ``full_indices`` must be whole ascending
        pages (the ``_page_to_token`` expansion); the in-page offsets are preserved
        (``window_slot = wbase + pos % P``), which the state ring's page-block layout requires."""
        n = int(full_indices.numel())
        if n == 0:
            return
        P = self.P
        assert n % P == 0, f"alloc_swa needs whole pages, got {n} slots"
        fi = full_indices.to(device=self._device, dtype=torch.int64).view(-1, P)
        fbases = fi[:, 0]
        assert torch.equal(fi, fbases[:, None] + torch.arange(P, device=self._device)), (
            "alloc_swa pages must be contiguous ascending"
        )
        wbases = self._win_alloc.alloc(fbases.numel())  # raises when exhausted (caller gated)
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = (
            wbases[:, None] + offsets
        ).flatten()

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the window pages backing these FULL locs and unbind the mapping. Page-atomic:
        the incoming locs must cover each touched page completely (guaranteed by the padded
        finish tails / aligned frontiers / page-aligned tree values). Idempotent over already
        unbound (slid/tombstoned) pages."""
        if full_indices.numel() == 0:
            return
        P = self.P
        fi = full_indices.to(device=self._device, dtype=torch.int64)
        fi = fi[fi >= 0]
        if fi.numel() == 0:
            return
        fbases, counts = torch.unique(
            torch.div(fi, P, rounding_mode="floor") * P, return_counts=True
        )
        assert bool((counts == P).all()), (
            f"free_swa got partial pages (counts {counts[counts != P].tolist()[:4]})"
        )
        ws = self.full_to_window[fbases]
        live = ws[ws >= 0]
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = -1
        if live.numel():
            self._win_alloc.free(torch.div(live, P, rounding_mode="floor") * P)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_window[kv_indices.to(dtype=torch.int64)]

    # ----- state_loc derivation (vectorized, LongTensor in/out) -----
    @staticmethod
    def state_loc(window_slot: torch.Tensor, ring_size: int, P: int) -> torch.Tensor:
        pages = torch.div(window_slot, P, rounding_mode="floor")
        loc = pages * ring_size + (window_slot % ring_size)
        return torch.where(window_slot < 0, torch.full_like(loc, -1), loc)

    def ring_size(self, layer_id: int) -> int:
        return ring_size_for_ratio(self.compress_ratios[layer_id])

    # ----- compress-state ring accessors -----
    def get_state(self, layer_id: int, state_loc: torch.Tensor) -> torch.Tensor:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} (ratio 0) has no compress-state ring"
        return ring.get(state_loc)

    def set_state(self, layer_id: int, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} (ratio 0) has no compress-state ring"
        ring.set(state_loc, kv_score)

    # ----- specialized writes -----
    def store_window(self, k: torch.Tensor, layer_id: int, window_slot: torch.Tensor) -> None:
        self.window_pool[layer_id].index_copy_(0, window_slot, k.to(self._dtype))

    def store_compressed(self, kv: torch.Tensor, layer_id: int, cmp_slot: torch.Tensor) -> None:
        pool = self.cmp_pool[layer_id]
        assert pool is not None, f"layer {layer_id} (ratio 0) has no compressed pool"
        pool.index_copy_(0, cmp_slot, kv.to(self._dtype))

    def store_indexer(self, k: torch.Tensor, layer_id: int, idx_slot: torch.Tensor) -> None:
        pool = self.idx_pool[layer_id]
        assert pool is not None, f"layer {layer_id} has no indexer pool (only ratio-4)"
        pool.index_copy_(0, idx_slot, k.to(self._dtype))

    def total_bytes(self) -> int:
        n = self.full_to_window.numel() * self.full_to_window.element_size()
        n += sum(t.numel() * t.element_size() for t in self.window_pool)
        n += sum(t.numel() * t.element_size() for t in self.cmp_pool if t is not None)
        n += sum(t.numel() * t.element_size() for t in self.idx_pool if t is not None)
        n += sum(
            r.buffer.numel() * r.buffer.element_size()
            for r in self.state_ring
            if r is not None
        )
        n += sum(
            r.buffer.numel() * r.buffer.element_size()
            for r in self.indexer_state_ring
            if r is not None
        )
        return int(n)

    # ----- HiCache restore segments -----
    # One payload is enough to resume a page-aligned prefix: window KV, all compressed
    # KV/indexer KV blocks in the page, and both compressor carry blocks for that page.
    # A page whose window slot was already tombstoned stores None for window KV; restore
    # zero-fills it and the SWA radix tombstones it again after insertion.

    def gather_prefix_page(self, full_locs: torch.Tensor) -> dict:
        P = self.P
        fi = full_locs.detach().to(device=self._device, dtype=torch.int64)
        assert fi.numel() == P
        base = int(fi[0].item())
        assert bool(torch.equal(fi, base + torch.arange(P, device=self._device))), (
            "DSV4 HiCache pages must be contiguous full-token pages"
        )
        window_slot = int(self.full_to_window[base].item())
        window = None
        if window_slot >= 0:
            assert window_slot % P == 0
            window = [
                self.window_pool[layer_id][window_slot:window_slot + P].detach().cpu().clone()
                for layer_id in range(self._n_layers)
            ]

        cmp: dict[int, torch.Tensor] = {}
        idx: dict[int, torch.Tensor] = {}
        rings: dict[int, dict[str, torch.Tensor | None]] = {}
        for layer_id, ratio in enumerate(self.compress_ratios):
            if not ratio:
                continue
            # One compressed row per block, not one per token position.
            cmp_rows = torch.arange(
                base // ratio, (base + P) // ratio, device=self._device, dtype=torch.int64
            )
            cmp[layer_id] = self.cmp_pool[layer_id][cmp_rows].detach().cpu().clone()
            if ratio == 4:
                idx[layer_id] = self.idx_pool[layer_id][cmp_rows].detach().cpu().clone()
            rings[layer_id] = {}
            for tier, attr in (("attn", "state_ring"), ("idx", "indexer_state_ring")):
                ring = getattr(self, attr)[layer_id]
                if ring is None:
                    rings[layer_id][tier] = None
                    continue
                ring_page = window_slot // P if window_slot >= 0 else None
                if ring_page is None:
                    rings[layer_id][tier] = None
                else:
                    block_base = ring_page * ring.ring_size
                    rings[layer_id][tier] = (
                        ring.buffer[block_base:block_base + ring.ring_size]
                        .detach().cpu().clone()
                    )
        return {"window": window, "cmp": cmp, "idx": idx, "rings": rings}

    def gather_prefix_pages(self, full_locs: torch.Tensor) -> list[dict]:
        """Synchronous gather of n whole pages: per-slice non_blocking D2H into
        per-shape pinned buffers on the current stream, then ONE stream sync and a
        clone into regular host memory. This is the implementation that passed the
        full 3-pass benchmark 60/60 before packing; the packed single-staging-copy
        variant wedged the driver six times (engine thread blocked forever inside
        cuMemcpyHtoDAsync for an unrelated pageable H2D, L8192xC8 only) and was
        reverted -- do not reintroduce GPU->GPU packing or cross-thread event
        handoff here without re-running that cell as the gate."""
        locs = full_locs.detach().to("cpu", dtype=torch.int64).contiguous()
        ticket = self._plan_prefix_run(locs)
        if self._device.type != "cuda":
            host = {tag: src.detach().cpu().clone() for src, tag in ticket["slices"]}
            return self._build_prefix_pages(host, ticket)
        host: dict[str, torch.Tensor] = {}
        for src, tag in ticket["slices"]:
            key = (tuple(src.shape), src.dtype)
            buf = self._pinned_bufs.get(key)
            if buf is None:
                if len(self._pinned_bufs) > 32:
                    self._pinned_bufs.clear()  # rare: many run lengths; drop and rebuild
                buf = torch.empty(tuple(src.shape), dtype=src.dtype, pin_memory=True)
                self._pinned_bufs[key] = buf
            buf.copy_(src.detach(), non_blocking=True)
            host[tag] = buf
        torch.cuda.current_stream(self._device).synchronize()
        return self._build_prefix_pages(host, ticket)

    def issue_gather_pages(self, full_locs: torch.Tensor) -> dict:
        """UNUSED (kept for reference only): the packed issue/complete split wedged
        the driver six times -- the serving path now calls gather_prefix_pages, the
        synchronous per-slice gather that passed the 3-pass benchmark 60/60.
        plan the run, slice the sources, pack them GPU->GPU into one staging tensor,
        enqueue ONE pinned D2H and record the done event. Returns a ticket the writer
        thread completes with build_gather_pages. No host wait except the single tiny
        ``full_to_window`` D2H inside _plan_prefix_run."""
        ticket = self._plan_prefix_run(
            full_locs.detach().to("cpu", dtype=torch.int64).contiguous()
        )
        if self._device.type != "cuda":
            ticket["host"] = {tag: src.detach().cpu().clone()
                              for src, tag in ticket["slices"]}
            return ticket
        ticket["pack"] = self._issue_pack(ticket["slices"])
        return ticket

    def build_gather_pages(self, ticket: dict) -> list[dict]:
        """COMPLETE half (writer thread): wait for the done event, reassemble the
        per-page payloads from the pinned pack, and queue the pack for the scheduler
        to recycle (see _pack_recycle -- the writer must not free pinned memory, let
        alone submit CUDA). Pure CPU except the cudaEventSynchronize, which is a
        wait-side call and never takes the submission locks the engine thread holds."""
        pack = ticket.get("pack")
        if pack is not None:
            try:
                pack["done"].synchronize()
                host = {tag: pack["pinned"][off:off + nbytes].view(dtype).view(shape)
                        for off, nbytes, shape, dtype, tag in pack["plan"]}
                return self._build_prefix_pages(host, ticket)
            finally:
                # Queue BEFORE anything can raise further: the payloads are CPU
                # clones, so recycling the storage once drained is always safe.
                self._pack_recycle.append(pack)
        return self._build_prefix_pages(ticket["host"], ticket)

    def drain_pack_recycle(self) -> None:
        """SCHEDULER thread: return consumed gather packs to the pinned/event free
        lists. Any pinned tensor that drops off here (or off a full free list) is
        deallocated on the scheduler thread, the process's only CUDA submitter."""
        while self._pack_recycle:
            pack = self._pack_recycle.pop()
            self._release_pinned(pack["pinned"])
            self._release_event(pack["done"])

    def _plan_prefix_run(self, locs: torch.Tensor) -> dict:
        """Slice a contiguous run of n whole pages into per-layer sources.

        One contiguity check on CPU, one indexed read of ``full_to_window`` (the ONLY
        device sync), then per-layer slices over the whole run -- ~2*n_layers+2 GPU
        ops instead of ~15 per page. The caller passes CPU int64 locs. Falls back to
        per-page slices when the run's window slots are not one contiguous bound range
        (rare: freshly committed suffixes allocate full+window pages as one
        sequential run) -- the fallback still needs no further device sync because the
        per-page window slots are already in hand."""
        P = self.P
        n = locs.numel() // P
        assert n >= 1 and locs.numel() % P == 0
        base0 = int(locs[0])
        if any(int(v) != base0 + i for i, v in enumerate(locs.tolist())):
            raise AssertionError("DSV4 HiCache pages must be contiguous full-token pages")
        bases = torch.arange(base0, base0 + n * P, P, device=self._device)
        ws_list = self.full_to_window[bases].cpu().tolist()
        if any(w % P for w in ws_list if w >= 0):
            raise AssertionError("DSV4 HiCache window slot is not page-aligned")
        if not all(0 <= ws_list[i] == ws_list[0] + i * P for i in range(n)) or any(
            ratio and (P % ratio or base0 % ratio) for ratio in self.compress_ratios
        ):
            # Scattered/unbound window pages or unaligned compressed rows: per-page slices.
            slices: list[tuple[torch.Tensor, str]] = []
            layouts: list[dict] = []
            for i in range(n):
                base = base0 + i * P
                ws = ws_list[i]
                layout: dict = {"window": ws >= 0, "cmp": {}, "idx": {}, "rings": {}}
                if ws >= 0:
                    for layer_id in range(self._n_layers):
                        slices.append(
                            (self.window_pool[layer_id][ws:ws + P], f"p{i}:win:{layer_id}"))
                for layer_id, ratio in enumerate(self.compress_ratios):
                    if not ratio:
                        continue
                    r0, r1 = base // ratio, (base + P) // ratio
                    layout["cmp"][layer_id] = r1 - r0
                    slices.append((self.cmp_pool[layer_id][r0:r1], f"p{i}:cmp:{layer_id}"))
                    if ratio == 4:
                        layout["idx"][layer_id] = r1 - r0
                        slices.append((self.idx_pool[layer_id][r0:r1], f"p{i}:idx:{layer_id}"))
                    for tier, attr in (("attn", "state_ring"), ("idx", "indexer_state_ring")):
                        ring = getattr(self, attr)[layer_id]
                        if ring is None or ws < 0:
                            layout["rings"][(layer_id, tier)] = None
                            continue
                        layout["rings"][(layer_id, tier)] = ring.ring_size
                        block_base = (ws // P) * ring.ring_size
                        slices.append(
                            (ring.buffer[block_base:block_base + ring.ring_size],
                             f"p{i}:ring:{layer_id}:{tier}"))
                layouts.append(layout)
            return {"slices": slices, "fast": False, "n": n, "layouts": layouts}

        span = n * P
        ws0 = ws_list[0]
        slices = []
        for layer_id in range(self._n_layers):
            slices.append((self.window_pool[layer_id][ws0:ws0 + span], f"win:{layer_id}"))
        per_page: dict[int, int] = {}
        ring_sizes: dict[tuple[int, str], int] = {}
        for layer_id, ratio in enumerate(self.compress_ratios):
            if not ratio:
                continue
            per_page[layer_id] = P // ratio
            row0 = base0 // ratio
            rows_n = (base0 + span) // ratio - row0
            slices.append((self.cmp_pool[layer_id][row0:row0 + rows_n], f"cmp:{layer_id}"))
            if ratio == 4:
                slices.append((self.idx_pool[layer_id][row0:row0 + rows_n], f"idx:{layer_id}"))
            for tier, attr in (("attn", "state_ring"), ("idx", "indexer_state_ring")):
                ring = getattr(self, attr)[layer_id]
                if ring is None:
                    continue
                ring_sizes[(layer_id, tier)] = ring.ring_size
                block_base = (ws0 // P) * ring.ring_size
                slices.append((ring.buffer[block_base:block_base + n * ring.ring_size],
                               f"ring:{layer_id}:{tier}"))
        return {"slices": slices, "fast": True, "n": n, "per_page": per_page,
                "ring_sizes": ring_sizes, "n_layers": self._n_layers}

    def _build_prefix_pages(self, host: dict[str, torch.Tensor],
                            ticket: dict) -> list[dict]:
        """Reassemble per-page payloads from host views (writer thread, pure CPU).
        Every returned tensor is a fresh clone -- the pinned source may be recycled
        as soon as this returns."""
        P = self.P
        pages: list[dict] = []
        if ticket["fast"]:
            n = ticket["n"]
            per_page = ticket["per_page"]
            ring_sizes = ticket["ring_sizes"]
            n_layers = ticket["n_layers"]
            for i in range(n):
                cmp_i = {layer: host[f"cmp:{layer}"][i * per_page[layer]:(i + 1) * per_page[layer]].clone()
                         for layer in per_page}
                idx_i = {layer: host[f"idx:{layer}"][i * per_page[layer]:(i + 1) * per_page[layer]].clone()
                         for layer in per_page if f"idx:{layer}" in host}
                rings_i: dict[int, dict[str, torch.Tensor | None]] = {}
                for (layer_id, tier), ring_size in ring_sizes.items():
                    h = host[f"ring:{layer_id}:{tier}"]
                    rings_i.setdefault(layer_id, {})[tier] = (
                        h[i * ring_size:(i + 1) * ring_size].clone())
                for layer_id, ratio in enumerate(self.compress_ratios):
                    if not ratio:
                        continue
                    tiers = rings_i.setdefault(layer_id, {})
                    for tier in ("attn", "idx"):
                        tiers.setdefault(tier, None)
                pages.append({
                    "window": [host[f"win:{layer_id}"][i * P:(i + 1) * P].clone()
                               for layer_id in range(n_layers)],
                    "cmp": cmp_i, "idx": idx_i, "rings": rings_i,
                })
            return pages
        for i, layout in enumerate(ticket["layouts"]):
            window = ([host[f"p{i}:win:{layer_id}"].clone()
                       for layer_id in range(self._n_layers)]
                      if layout["window"] else None)
            cmp_i = {layer: host[f"p{i}:cmp:{layer}"].clone() for layer in layout["cmp"]}
            idx_i = {layer: host[f"p{i}:idx:{layer}"].clone() for layer in layout["idx"]}
            rings_i: dict[int, dict[str, torch.Tensor | None]] = {}
            for (layer_id, tier), size in layout["rings"].items():
                tiers = rings_i.setdefault(layer_id, {})
                if size is None:
                    tiers[tier] = None
                else:
                    tiers[tier] = host[f"p{i}:ring:{layer_id}:{tier}"].clone()
            pages.append({"window": window, "cmp": cmp_i, "idx": idx_i,
                          "rings": rings_i})
        return pages

    def _issue_pack(self, slices: list[tuple[torch.Tensor, str]]) -> dict:
        """Pack every slice into one GPU uint8 staging tensor with ONE pinned D2H at
        the end, all on the CURRENT stream (the put gather stream), all async.

        The GPU->GPU packing intentionally does NOT go through the copy engines:
        ``mul(src, 1, out=dst)`` runs an elementwise kernel on the SMs instead. At
        KV-pool pressure >0.96 the copy-engine path wedged the driver three times
        (the scheduler thread blocked forever inside cuMemcpyHtoDAsync while another
        thread was in cuLaunchKernel/cuEventDestroy on the same device), while the
        pre-packing builds that used pageable D2Hs ran 60/60. One CE entry for the
        final pinned D2H stays; the big staging copies cost ~1ms of SM time instead.

        Returns the plan/pinned/done ticket for build_gather_pages."""
        ALIGN = 256  # keeps every byte offset element-alignable for .view(dtype)
        plan: list[tuple[int, int, tuple, torch.dtype, str]] = []
        off = 0
        for src, tag in slices:
            nbytes = src.numel() * src.element_size()
            plan.append((off, nbytes, tuple(src.shape), src.dtype, tag))
            off += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        stream = torch.cuda.current_stream(self._device)
        staging = self._gpu_pack_bufs.get(off)
        if staging is None:
            if len(self._gpu_pack_bufs) > 8:
                self._gpu_pack_bufs.clear()  # rare: many run lengths; drop and rebuild
            staging = torch.empty(off, dtype=torch.uint8, device=self._device)
            self._gpu_pack_bufs[off] = staging
        # the cache holds the only ref; if evicted while copies are in flight, the
        # allocator must not hand the blocks to another stream's allocation
        staging.record_stream(stream)
        for (off_i, nbytes, _shape, _dtype, _tag), (src, _) in zip(plan, slices):
            src = src.detach()
            if not src.is_contiguous():
                src = src.contiguous()
            dst = staging[off_i:off_i + nbytes].view(_dtype)
            # SM copy: x*1 == x bit-exact for every pool dtype (bf16/fp32/int)
            torch.mul(src.reshape(-1), 1, out=dst.reshape(-1))
        pinned = self._acquire_pinned(off)
        # copy_ requires equal shapes: a recycled buffer may be larger than this
        # pack, so slice it to the pack size first
        pinned[:off].copy_(staging[:off], non_blocking=True)
        done = self._acquire_event()
        done.record(stream)
        return {"plan": plan, "pinned": pinned, "done": done}

    def _acquire_pinned(self, nbytes: int) -> torch.Tensor:
        for i, buf in enumerate(self._pinned_free):
            if buf.numel() >= nbytes:
                return self._pinned_free.pop(i)
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def _release_pinned(self, buf: torch.Tensor) -> None:
        if len(self._pinned_free) < 8:
            self._pinned_free.append(buf)

    def _acquire_event(self) -> "torch.cuda.Event":
        """Recycled done events: an Event object GC'd on the writer thread calls
        cuEventDestroy -- a driver call we want off that thread entirely (and the
        reclaim race was another suspect in the driver wedges)."""
        if self._event_free:
            return self._event_free.pop()
        return torch.cuda.Event()

    def _release_event(self, ev: "torch.cuda.Event") -> None:
        # Deliberately UNBOUNDED: capping the free list made every release past the
        # cap drop the event, so it was GC'd -- and cuEventDestroy'd -- on the writer
        # thread, one driver call there too many during the channel-starvation wedge.
        # Events are ~100 bytes of host state; the pool only grows to peak in-flight
        # puts. An event still tracked in the tier's _gather_inflight deque can be
        # recycled and re-recorded: the tier prunes its deque before acquiring, and a
        # re-recorded alias only makes a deque wait more conservative, never shorter.
        self._event_free.append(ev)

    def scatter_prefix_page(self, full_locs: torch.Tensor, payload: dict) -> None:
        P = self.P
        fi = full_locs.detach().to(device=self._device, dtype=torch.int64)
        base = int(fi[0].item())
        assert bool(torch.equal(fi, base + torch.arange(P, device=self._device))), (
            "DSV4 HiCache restore pages must be contiguous full-token pages"
        )
        window_slot = int(self.full_to_window[base].item())
        bound = window_slot >= 0
        if bound:
            assert window_slot % P == 0

        window = payload.get("window")
        # An unbound page is restored compressed-only: its raw window KV no longer exists
        # (the snapshot stores None) and the SWA tree tombstones the page right after the
        # insert. Rings are page-block addressed through the window slot -> skipped too.
        assert window is None or bound, "window payload without a bound swa slot"
        if bound:
            for layer_id in range(self._n_layers):
                if window is not None:
                    self.window_pool[layer_id][window_slot:window_slot + P].copy_(
                        window[layer_id].to(device=self._device, dtype=self._dtype)
                    )
                else:
                    self.window_pool[layer_id][window_slot:window_slot + P].zero_()

        for layer_id, ratio in enumerate(self.compress_ratios):
            if not ratio:
                continue
            rows = torch.arange(
                base // ratio, (base + P) // ratio, device=self._device, dtype=torch.int64
            )
            self.cmp_pool[layer_id][rows] = payload["cmp"][layer_id].to(
                device=self._device, dtype=self.cmp_pool[layer_id].dtype
            )
            if ratio == 4:
                self.idx_pool[layer_id][rows] = payload["idx"][layer_id].to(
                    device=self._device, dtype=self.idx_pool[layer_id].dtype
                )
            rings = payload["rings"].get(layer_id, {})
            for tier, attr in (("attn", "state_ring"), ("idx", "indexer_state_ring")):
                ring = getattr(self, attr)[layer_id]
                block = rings.get(tier)
                if ring is None or block is None or not bound:
                    continue
                ring_page = window_slot // P
                block_base = ring_page * ring.ring_size
                ring.buffer[block_base:block_base + ring.ring_size] = block.to(
                    device=self._device, dtype=ring.buffer.dtype
                )
                ring._clear_scratch()

    # ----- BaseKVCachePool interface -----
    def k_cache(self, index: int) -> torch.Tensor:
        return self.window_pool[index]

    def v_cache(self, index: int) -> torch.Tensor:  # MLA: K == V (single latent)
        return self.window_pool[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        # Thin window-write shim for ABC compat; DSV4 writes via the specialized
        # setters above. K == V (single latent), so the window slot is out_loc.
        self.store_window(k, layer_id, out_loc)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._n_layers


__all__ = ["CompressStateRing", "DSV4PagedKVCache"]
