import threading

import pytest
import torch

from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.dsv4_prefix_tier import DSV4PrefixTier
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.kvcache.dsv4_cost_model import DSV4PoolSizes


P = 128


def _args():
    args = DeepseekV4Args()
    args.n_layers = 3
    args.head_dim = 8
    args.index_head_dim = 4
    args.window_size = P
    args.compress_ratios = (0, 4, 128)
    return args


def _pool():
    sizes = DSV4PoolSizes(
        P=P, swa_ratio=1.0, full_token=4 * P, n_win_slots=4 * P, n_win_pages=4,
        cmp_blocks=[None, 128, 4], idx_blocks=[None, 128, None],
        state_slots=[None, 32, 512], ring_sizes=[None, 8, 128],
        idx_state_slots=[None, 32, None],
    )
    pool = DSV4PagedKVCache(
        sizes=sizes, args=_args(), device=torch.device("cpu"), dtype=torch.bfloat16,
        P=P, n_scratch=1,
    )
    pool._init_paged_state(max_running_req=1, radix=True)
    return pool


def test_dsv4_prefix_page_roundtrip():
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 7
    pool.cmp_pool[2][torch.arange(2)] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 9
    for tier_name, ring in (("attn", pool.state_ring[1]), ("idx", pool.indexer_state_ring[1])):
        ring.buffer[:8] = torch.arange(8 * ring.buffer.shape[-1], dtype=torch.float32).view(8, -1) + (1 if tier_name == "idx" else 0)
    pool.state_ring[2].buffer[:128] = torch.arange(
        128 * pool.state_ring[2].buffer.shape[-1], dtype=torch.float32
    ).view(128, -1)

    payload = pool.gather_prefix_page(full[:P])
    assert payload["window"][0].shape == (P, 8)
    assert torch.equal(payload["cmp"][1], pool.cmp_pool[1][:32].cpu())

    # Destroy all live pool state, then restore the exact page from the payload.
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.state_ring[1].buffer.fill_(float("nan")); pool.indexer_state_ring[1].buffer.fill_(float("nan"))
    pool.state_ring[2].buffer.fill_(float("nan"))
    pool.scatter_prefix_page(full[:P], payload)

    window_slots = pool.translate_full_to_window(full[:P])
    assert torch.equal(pool.window_pool[0][window_slots.long()].cpu(), payload["window"][0])
    assert torch.equal(pool.cmp_pool[1][:32].cpu(), payload["cmp"][1])
    assert torch.equal(pool.idx_pool[1][:4].cpu(), payload["idx"][1][:4])
    ring_page = int(pool.translate_full_to_window(full[:1])[0].item()) // P
    ring_base = ring_page * pool.state_ring[1].ring_size
    assert torch.equal(
        pool.state_ring[1].buffer[ring_base:ring_base + 8].cpu(), payload["rings"][1]["attn"]
    )
    assert torch.equal(
        pool.indexer_state_ring[1].buffer[ring_base:ring_base + 8].cpu(), payload["rings"][1]["idx"]
    )
    ring_page128 = int(pool.translate_full_to_window(full[:1])[0].item()) // P
    ring_base128 = ring_page128 * pool.state_ring[2].ring_size
    assert torch.equal(
        pool.state_ring[2].buffer[ring_base128:ring_base128 + 128].cpu(),
        payload["rings"][2]["attn"],
    )


def test_dsv4_prefix_batched_gather_matches_per_page():
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 7
    pool.cmp_pool[2][torch.arange(2)] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 9
    pool.state_ring[2].buffer[:128] = torch.arange(
        128 * pool.state_ring[2].buffer.shape[-1], dtype=torch.float32
    ).view(128, -1)

    batched = pool.gather_prefix_pages(full)
    assert len(batched) == 2
    per_page = [pool.gather_prefix_page(full[i * P:(i + 1) * P]) for i in range(2)]
    for b, s in zip(batched, per_page):
        assert (b["window"] is None) == (s["window"] is None)
        if s["window"] is not None:
            for layer in range(3):
                assert torch.equal(b["window"][layer], s["window"][layer])
        for layer_id, ratio in enumerate(pool.compress_ratios):
            if not ratio:
                continue
            assert torch.equal(b["cmp"][layer_id], s["cmp"][layer_id])
            if ratio == 4:
                assert torch.equal(b["idx"][layer_id], s["idx"][layer_id])
            for tier in ("attn", "idx"):
                got = b["rings"].get(layer_id, {}).get(tier)
                want = s["rings"].get(layer_id, {}).get(tier)
                if want is None:
                    assert got is None
                else:
                    assert torch.equal(got, want)

    # Scatter the window slots so the run is no longer one contiguous bound range:
    # the batched gather must fall back to the per-page path and stay correct.
    pool.full_to_window[full.long()] = -1
    pool.full_to_window[full[P:].long()] = torch.arange(2 * P, 3 * P, dtype=torch.int64)
    scattered = pool.gather_prefix_pages(full)
    assert scattered[0]["window"] is None
    for layer in range(3):
        assert torch.equal(
            scattered[1]["window"][layer],
            pool.gather_prefix_page(full[P:])["window"][layer],
        )


def test_dsv4_prefix_tier_spills_and_restores(tmp_path):
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 7
    pool.cmp_pool[2][torch.arange(2)] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 9
    pool.state_ring[1].buffer[:8] = torch.arange(
        8 * pool.state_ring[1].buffer.shape[-1], dtype=torch.float32
    ).view(8, -1)
    pool.state_ring[2].buffer[:128] = torch.arange(
        128 * pool.state_ring[2].buffer.shape[-1], dtype=torch.float32
    ).view(128, -1)

    expected_window = pool.window_pool[0][window_slots.long()].cpu().clone()
    ring_bases = (pool.translate_full_to_window(full[::P]).long() // P) * pool.state_ring[1].ring_size
    expected_ring1 = pool.state_ring[1].buffer[ring_bases].cpu().clone()
    expected_ring2 = pool.state_ring[2].buffer[
        (pool.translate_full_to_window(full[::P]).long() // P) * pool.state_ring[2].ring_size
    ].cpu().clone()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert tier.put(ids, full) == 2 * P
    tier.flush()
    assert tier.host_pages_count == 1 and tier.disk_pages_count == 1

    # Reuse the same full pages after unbinding and poisoning every pool buffer.
    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.state_ring[1].buffer.fill_(float("nan")); pool.indexer_state_ring[1].buffer.fill_(float("nan"))
    pool.state_ring[2].buffer.fill_(float("nan"))
    pool.alloc_swa(full)

    assert tier.get(ids, full) == 2 * P
    window_slots = pool.translate_full_to_window(full)
    assert torch.equal(pool.window_pool[0][window_slots.long()].cpu(), expected_window)
    assert torch.equal(pool.cmp_pool[1][:32].cpu(),
                       torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7)
    assert torch.equal(pool.idx_pool[1][:32].cpu(),
                       torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9)
    restored_ring_bases = (
        (pool.translate_full_to_window(full[::P]).long() // P) * pool.state_ring[1].ring_size
    )
    assert torch.equal(pool.state_ring[1].buffer[restored_ring_bases].cpu(), expected_ring1)
    restored_ring_bases2 = (
        (pool.translate_full_to_window(full[::P]).long() // P) * pool.state_ring[2].ring_size
    )
    assert torch.equal(pool.state_ring[2].buffer[restored_ring_bases2].cpu(), expected_ring2)


def test_cache_manager_restores_dsv4_host_prefix(tmp_path):
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    # 4 pages: the restore must clear the capacity gate with room for the prompt tail
    # after the seeding cycle hands its pages back.
    cm = CacheManager(
        num_pages=4, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    # match_req excludes the final prompt token; use one extra token so two full
    # host pages are eligible for restore.
    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))
    handle = cm.match_req(pending).cuda_handle
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=handle,
    )
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)

    window_slots = pool.translate_full_to_window(pt[0, :2 * P])
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9
    expected_window = pool.window_pool[0][window_slots.long()].cpu().clone()
    with cm.lazy_free_region():
        cm.cache_req(req, finished=False)
    cm.flush_host_puts()  # async write-through: apply the queued put before poisoning

    # Force both pages out of the GPU SWA tree, then poison every live pool buffer.
    cm.unlock(req.cache_handle)
    evicted = cm.prefix_cache.evict_full(len(ids))
    cm._free_swa(evicted.swa_indices)
    cm._free(evicted.kv_indices)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))

    restored = cm.match_req(pending).cuda_handle
    assert restored.cached_len == 2 * P
    restored_locs = restored.get_matched_indices()
    restored_windows = pool.translate_full_to_window(restored_locs)
    # The restore path tombstones window KV strictly below the trailing window; only the
    # live tail is required to resume prefill.
    assert torch.equal(
        pool.window_pool[0][restored_windows[-P:].long()].cpu(), expected_window[-P:]
    )
    assert torch.equal(
        pool.cmp_pool[1][:32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )


def test_scatter_prefix_page_allows_unbound_head():
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    # Head page slid out of the window before the gather: its snapshot must carry
    # window=None and no ring blocks, exactly like a finish-time put of a long prompt.
    pool.free_swa(full[:P])
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9

    head = pool.gather_prefix_page(full[:P])
    assert head["window"] is None
    assert all(block is None for tier_blocks in head["rings"].values() for block in tier_blocks.values())
    tail = pool.gather_prefix_page(full[P:])
    assert tail["window"] is not None

    # Restore with only the tail bound: the head page is restored compressed-only and
    # needs no swa slot (it had none when the snapshot was taken).
    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.state_ring[1].buffer.fill_(float("nan"))
    pool.alloc_swa(full[P:])
    pool.scatter_prefix_page(full[:P], head)
    pool.scatter_prefix_page(full[P:], tail)

    assert bool((pool.translate_full_to_window(full[:P]) < 0).all())
    assert torch.equal(
        pool.cmp_pool[1][:32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )
    tail_windows = pool.translate_full_to_window(full[P:])
    assert torch.equal(
        pool.window_pool[0][tail_windows.long()].cpu()[:, 0],
        torch.arange(P, 2 * P, dtype=torch.bfloat16),
    )


def test_cache_manager_restores_with_scarce_swa_slots(tmp_path):
    """A long-prompt restore must not demand one swa slot per restored page: only the
    trailing snapshots carry window KV, so restoring 2 pages with a single free window
    page has to succeed (the old bind-every-page gate silently killed it)."""
    from types import SimpleNamespace

    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9
    expected_tail = pool.window_pool[0][window_slots[-P:].long()].cpu().clone()

    # Head page out of the window at put time (finish-time retention shape).
    pool.free_swa(full[:P])
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    assert tier.put(ids, full) == 2 * P

    # Poison the live state and leave exactly ONE free window page: the restore depth
    # is 2 pages but only the tail needs a swa slot.
    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.alloc_swa(torch.arange(2 * P, 4 * P, dtype=torch.int32))

    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=3, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )
    ids_req = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(
        input_ids=torch.tensor(ids_req, dtype=torch.int32), input_len=len(ids_req))
    restored = cm.match_req(pending).cuda_handle
    assert restored.cached_len == 2 * P
    locs = restored.get_matched_indices()
    assert torch.equal(
        pool.cmp_pool[1][:32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )
    windows = pool.translate_full_to_window(locs)
    assert bool((windows[:P] < 0).all())  # head stays unbound (compressed-only)
    assert torch.equal(pool.window_pool[0][windows[-P:].long()].cpu(), expected_tail)
    # Request-level hit-rate counters: one eligible request restored 2 pages.
    assert tier.stats.restore_eligible == 1
    assert tier.stats.restore_requests == 1
    assert tier.stats.restore_tokens == 2 * P
    assert tier.stats.restore_eligible_tokens == 2 * P
    # Global KV hit-rate counters: counted once even when match_req repeats.
    assert cm.kv_stat_requests == 1
    assert cm.kv_stat_prompt_tokens == 2 * P
    assert cm.kv_stat_cached_tokens == 2 * P
    snap = cm.kv_stat_snapshot()
    assert snap["hit_rate_pct"] == 100.0
    cm.match_req(pending)
    assert cm.kv_stat_requests == 1


def test_restore_pins_pages_against_concurrent_allocation(tmp_path):
    """Page ownership: restored pages are pinned from publish until the owning handle is
    unlocked. Before the pin, a concurrent admission's allocator reclaimed the freshly
    published pages through evict_full (unlocked leaves), forcing a repeat restore."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=4, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))

    # Seed the tier: one request cycle puts both pages into host, then everything is
    # evicted from the GPU tree so the next match_req restores from host.
    seed_handle = cm.match_req(pending).cuda_handle
    seed_req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=seed_handle,
    )
    seed_req.device_len = len(ids)
    cm.lock(seed_handle)
    cm.allocate_paged([seed_req])
    seed_req.cached_len = len(ids)
    with cm.lazy_free_region():
        cm.cache_req(seed_req, finished=False)
    cm.flush_host_puts()  # async write-through: apply the queued put before evicting
    cm.unlock(seed_req.cache_handle)
    evicted = cm.prefix_cache.evict_full(len(ids))
    cm._free_swa(evicted.swa_indices)
    cm._free(evicted.kv_indices)

    handle = cm.match_req(pending).cuda_handle
    assert getattr(handle, "restore_locked", False)
    assert cm.prefix_cache.full_protected == 2 * P

    # A concurrent admission squeezes the pool: eviction must NOT reach the pinned prefix
    # (1 free page left, 1 more needed -> the old code evicted the restore and cascaded
    # the tombstone head away; the pin now starves the allocator instead).
    with pytest.raises(AssertionError, match="Eviction did not free enough space"):
        cm._allocate(2)
    m = cm.prefix_cache.match_prefix(torch.tensor(ids[: 2 * P], dtype=torch.int32))
    assert m.cached_len == 2 * P
    assert cm.prefix_cache.full_protected == 2 * P

    # lock() on the pinned handle is idempotent (admission re-lock path).
    cm.lock(handle)
    assert cm.prefix_cache.full_protected == 2 * P

    # The admission-failure path unlocks symmetrically: pin released, pages evictable again.
    cm.unlock(handle)
    assert cm.prefix_cache.full_protected == 0
    evicted = cm.prefix_cache.evict_full(2 * P)
    assert int(evicted.kv_indices.numel()) == 2 * P


def test_restore_skipped_when_prefill_tail_would_not_fit(tmp_path):
    """The capacity gate must cover the prompt tail: a restore that fits only its own pages
    dooms the admission, which then requeues unlocked and re-exposes the published pages."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=3, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))

    # Seed the tier with the 2-page prefix, then hand every page back: after the evict
    # exactly 2 pages are free -- the 2-page restore fits, but restore + 1-page prompt
    # tail does not, so the gate must decline the restore entirely.
    seed_handle = cm.match_req(pending).cuda_handle
    seed_req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=seed_handle,
    )
    seed_req.device_len = len(ids)
    cm.lock(seed_handle)
    cm.allocate_paged([seed_req])
    seed_req.cached_len = len(ids)
    with cm.lazy_free_region():
        cm.cache_req(seed_req, finished=False)
    cm.flush_host_puts()  # async write-through: apply the queued put before evicting
    cm.unlock(seed_req.cache_handle)
    evicted = cm.prefix_cache.evict_full(len(ids))
    cm._free_swa(evicted.swa_indices)
    cm._free(evicted.kv_indices)

    handle = cm.match_req(pending).cuda_handle
    # Restore declined: no pages published, no pin, counters still record the attempt.
    assert handle.cached_len == 0
    assert cm.prefix_cache.full_evictable == 0
    assert not getattr(handle, "restore_locked", False)
    assert tier.stats.restore_eligible == 1
    assert tier.stats.restore_requests == 0


def test_dsv4_prefix_tier_restarts_from_disk_metadata(tmp_path):
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9
    expected_window = pool.window_pool[0][window_slots.long()].cpu().clone()

    first = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert first.put(ids, full) == 2 * P
    first.close()

    second = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    assert second.match_len(ids) == 2 * P
    assert second.restorable_len(ids) == 2 * P

    # A restarted tier owns only disk pages; poison GPU state before restore.
    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.alloc_swa(full)
    assert second.get(ids, full) == 2 * P
    assert torch.equal(
        pool.window_pool[0][pool.translate_full_to_window(full).long()].cpu(),
        expected_window,
    )


def test_dsv4_prefix_tier_rejects_corrupt_payload(tmp_path):
    pool = _pool()
    full = torch.arange(P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + P, dtype=torch.int32)
    first = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert first.put(ids, full) == P
    first.close()

    path = next((tmp_path / "host").glob("page-*.pt"))
    wrapper = torch.load(path, map_location="cpu", weights_only=True)
    wrapper["checksum"] = "0" * 64
    torch.save(wrapper, path)

    second = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        second.get(ids, full)


def test_dsv4_prefix_tier_prefetch_warms_disk_page(tmp_path):
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9

    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert tier.put(ids, full) == 2 * P
    tier.flush()
    assert tier.disk_pages_count == 1

    assert tier.prefetch(ids) == 1
    tier.wait_prefetch(ids)
    assert tier.stats.prefetch_scheduled == 1

    # Poison the GPU state; both pages must still restore, including the warmed one.
    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.alloc_swa(full)
    assert tier.get(ids, full) == 2 * P
    assert torch.equal(
        pool.cmp_pool[1][:32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )


def test_dsv4_prefix_tier_disk_quota_evicts_prefix(tmp_path):
    pool = _pool()
    tier = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host", disk_capacity_pages=2
    )
    for sequence in range(4):
        ids = torch.arange(1000 + sequence * P, 1000 + (sequence + 1) * P, dtype=torch.int32)
        full = torch.arange(sequence * P, (sequence + 1) * P, dtype=torch.int32)
        pool.alloc_swa(full)
        tier.put(ids, full)
        pool.free_swa(full)
    assert tier.disk_pages_count == 2
    matched = sum(
        tier.match_len(torch.arange(1000 + seq * P, 1000 + (seq + 1) * P, dtype=torch.int32)) == P
        for seq in range(4)
    )
    assert matched == 3



def test_admission_gate_failure_skips_unlock_on_plain_match(tmp_path):
    """Regression: the kv_pages admission gate used to unlock the matched handle even when
    the restore had been declined by the tier capacity gate, leaving a plain GPU-match
    handle nobody locked. dec_lock then drove the matched node's ref_count negative and
    took the scheduler down (seen live as a TP0/TP1 crash on the L8192xC8 cell)."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=3, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending_ns = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))

    # Seed the tier with the 2-page prefix, then hand every page back.
    seed_handle = cm.match_req(pending_ns).cuda_handle
    seed_req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=seed_handle,
    )
    seed_req.device_len = len(ids)
    cm.lock(seed_handle)
    cm.allocate_paged([seed_req])
    seed_req.cached_len = len(ids)
    with cm.lazy_free_region():
        cm.cache_req(seed_req, finished=False)
    cm.flush_host_puts()  # async write-through: apply the queued put before evicting
    cm.unlock(seed_req.cache_handle)
    evicted = cm.prefix_cache.evict_full(len(ids))
    cm._free_swa(evicted.swa_indices)
    cm._free(evicted.kv_indices)

    # Leave a one-page GPU match live (non-root node, unlocked, ref 0) so the next match
    # returns a handle that looks partially cached: the restore capacity gate then declines
    # (256 restore + 1 tail > 128 free) and hands back that unlocked handle.
    head_loc = cm._page_to_token(cm._allocate(1))
    cm.prefix_cache.insert(torch.tensor(ids[:P], dtype=torch.int32), head_loc)

    tm = TableManager(max_running_reqs=4, page_table=pt)
    # reserved_size stands in for the sibling admissions of the same pass: with it the
    # FIRST gate (kv_pages, before any lock) refuses, which is the crashing call site.
    adder = PrefillAdder(token_budget=4096, reserved_size=P, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.tensor(ids, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending) is None
    assert adder.last_fail == "kv_pages"
    assert cm.prefix_cache.full_protected == 0


def test_put_async_applies_and_releases(tmp_path):
    """put_async gathers on the writer thread, stores the pages, and invokes the release
    callback exactly once -- the contract the CacheManager's tree pin relies on."""
    from collections import deque as _deque

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=4, disk_dir=tmp_path / "host")
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    for layer in range(3):
        pool.window_pool[layer][pool.translate_full_to_window(full).long()] = 1.0
    ids = torch.arange(2 * P, dtype=torch.int32)

    releases: _deque = _deque()
    returned = tier.put_async(ids, full, release=lambda: releases.append(1))
    assert returned == 2 * P            # every page was new
    assert tier.pending_puts == 1       # queued, not yet applied
    tier.wait_puts()
    assert tier.pending_puts == 0
    assert len(releases) == 1           # release ran exactly once, after the job
    assert tier.stats.put_requests == 1 and tier.stats.put_pages == 2 * P // P

    # A repeat put of the same prefix is a tier no-op and still releases.
    releases.clear()
    assert tier.put_async(ids, full, release=lambda: releases.append(1)) == 0
    tier.wait_puts()
    assert len(releases) == 1


def test_put_async_drops_stale_generation(tmp_path):
    """A queued put whose tier was cleared meanwhile must not resurrect stale pages."""
    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=4)
    full = torch.arange(P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(P, dtype=torch.int32)
    # Direct writer-side entry with a generation from a future clear: dropped.
    job = tier._issue_put(ids.clone(), full, None, None, tier._generation + 1)
    assert tier._complete_put(job) == 0
    assert tier.stats.put_requests == 0 and tier.host_pages_count == 0
    # Current generation stores normally.
    job = tier._issue_put(ids.clone(), full, None, None, tier._generation)
    assert tier._complete_put(job) == P
    assert tier.host_pages_count == 1


def test_cache_manager_async_put_pins_and_restores(tmp_path):
    """The CacheManager async path: cache_req pins the committed path, the writer gathers
    under the pin, the release queue hands the dec_lock back to the scheduler thread, and
    the stored pages still restore after the GPU tree is evicted."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2)
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=4, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))
    handle = cm.match_req(pending).cuda_handle
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=handle,
    )
    req.device_len = len(ids)
    # Hold the writer mid-put so the in-flight assertions below are deterministic
    # (the put itself completes in microseconds on the CPU-device pool).
    gate = threading.Event()
    real_complete = tier._complete_put

    def gated_complete(job):
        gate.wait(timeout=30)
        return real_complete(job)

    tier._complete_put = gated_complete
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    with cm.lazy_free_region():
        cm.cache_req(req, finished=False)

    # While the put is in flight the committed path is pinned against eviction...
    assert tier.pending_puts == 1
    assert cm.prefix_cache.full_protected >= 2 * P
    # ...and once the writer finished and the request's own handle unlocked, the
    # scheduler-thread drain leaves no protected ref behind.
    gate.set()
    # ...and once the writer finished and the request's own handle unlocked, the
    # scheduler-thread drain leaves no protected ref behind.
    cm.unlock(req.cache_handle)
    cm.flush_host_puts()
    assert tier.pending_puts == 0
    assert len(cm._host_releases) == 0
    assert cm.prefix_cache.full_protected == 0
    assert tier.stats.put_pages == 2

    # Evict the GPU tree, poison the pools, and restore purely from the async-stored tier.
    evicted = cm.prefix_cache.evict_full(len(ids))
    cm._free_swa(evicted.swa_indices)
    cm._free(evicted.kv_indices)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))

    restored = cm.match_req(pending).cuda_handle
    assert restored.cached_len == 2 * P
    assert tier.stats.restore_requests == 1
    assert tier.stats.restore_failures == 0


def test_match_prefix_full_returns_deepest_node():
    """match_prefix_full covers the whole key path (no swa windowed truncation) and
    node-aligns a partial match -- the boundary the async put pin relies on."""
    from freetoken.kvcache.swa_radix_cache import SWARadixCache

    tree = SWARadixCache(torch.device("cpu"), page_size=P, sliding_window_size=P)
    ids = torch.arange(3 * P, dtype=torch.int32)
    kv = torch.arange(3 * P, dtype=torch.int64)
    tree.insert(ids, kv)

    node, matched = tree.match_prefix_full(ids)
    assert matched == 3 * P and not node.is_root()
    assert node.length == 3 * P  # one insert call coalesces into a single node

    # A diverging tail stops at the deepest shared page boundary.
    other = ids.clone()
    other[-1] += 1
    node2, matched2 = tree.match_prefix_full(other)
    assert matched2 == (3 * P - 1) // P * P  # page-aligned down: 2 pages

    # A partial match inside a longer node splits it and stops at the query boundary.
    node3, matched3 = tree.match_prefix_full(ids[:2 * P])
    assert matched3 == 2 * P and node3.length == 2 * P


def _fill_pool_for_prefix(pool, full, ids_len):
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            len(full), dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 7
    pool.cmp_pool[2][torch.arange(2)] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][torch.arange(32)] = torch.arange(32, dtype=torch.bfloat16).unsqueeze(1) + 9
    pool.state_ring[1].buffer[:8] = torch.arange(
        8 * pool.state_ring[1].buffer.shape[-1], dtype=torch.float32
    ).view(8, -1)
    pool.state_ring[2].buffer[:128] = torch.arange(
        128 * pool.state_ring[2].buffer.shape[-1], dtype=torch.float32
    ).view(128, -1)


def test_dsv4_prefix_tier_raw_page_roundtrip(tmp_path):
    from freetoken.kvcache import gds_io

    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    _fill_pool_for_prefix(pool, full, 2 * P)
    expected_window = pool.window_pool[0][
        pool.translate_full_to_window(full).long()
    ].cpu().clone()

    first = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host", use_gds=True
    )
    assert first.put(ids, full) == 2 * P
    first.close()
    paths = sorted((tmp_path / "host").glob("page-*.pt"))
    assert len(paths) == 2
    assert all(gds_io.is_raw_page(p) for p in paths)

    second = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host", use_gds=True
    )
    assert second.match_len(ids) == 2 * P

    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.state_ring[1].buffer.fill_(float("nan")); pool.indexer_state_ring[1].buffer.fill_(float("nan"))
    pool.state_ring[2].buffer.fill_(float("nan"))
    pool.alloc_swa(full)

    assert second.get(ids, full) == 2 * P
    assert torch.equal(
        pool.window_pool[0][pool.translate_full_to_window(full).long()].cpu(),
        expected_window,
    )
    assert second.stats.disk_reads == 2
    assert second.stats.gds_reads == 0  # CPU pool: the GDS reader never engages
    second.close()


def test_dsv4_prefix_tier_raw_rejects_corruption(tmp_path):
    from freetoken.kvcache import gds_io

    pool = _pool()
    full = torch.arange(P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + P, dtype=torch.int32)
    _fill_pool_for_prefix(pool, full, P)

    first = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host", use_gds=True
    )
    assert first.put(ids, full) == P
    first.close()

    path = next((tmp_path / "host").glob("page-*.pt"))
    header = gds_io.read_header(path)
    raw = bytearray(path.read_bytes())
    raw[header["block_offset"] + 3] ^= 0xFF
    path.write_bytes(bytes(raw))

    second = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host", use_gds=True
    )
    assert second.match_len(ids) == P
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        second.get(ids, full)


def test_dsv4_prefix_tier_reads_mixed_page_formats(tmp_path):
    from freetoken.kvcache import gds_io

    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)
    _fill_pool_for_prefix(pool, full, 2 * P)
    expected_window = pool.window_pool[0][
        pool.translate_full_to_window(full).long()
    ].cpu().clone()

    # Page 0 is written in the legacy torch.save format, page 1 in raw format.
    legacy = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert legacy.put(ids[:P], full[:P]) == P
    legacy.close()
    raw = DSV4PrefixTier(
        pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host", use_gds=True
    )
    assert raw.put(ids, full) == 2 * P
    raw.close()
    paths = sorted((tmp_path / "host").glob("page-*.pt"))
    assert [gds_io.is_raw_page(p) for p in paths] == [False, True]

    third = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    assert third.match_len(ids) == 2 * P

    pool.free_swa(full)
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))
    pool.alloc_swa(full)
    assert third.get(ids, full) == 2 * P
    assert torch.equal(
        pool.window_pool[0][pool.translate_full_to_window(full).long()].cpu(),
        expected_window,
    )


def test_bytes_per_page_matches_payload_size():
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    for layer in range(3):
        pool.window_pool[layer][pool.translate_full_to_window(full).long()] = 1.0
    pool.cmp_pool[1][:32] = 1.0
    pool.cmp_pool[2][:2] = 1.0
    pool.idx_pool[1][:32] = 1.0
    pool.state_ring[1].buffer[:8] = 1.0
    pool.state_ring[2].buffer[:128] = 1.0

    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=1)
    payload = pool.gather_prefix_page(full[:P])
    from freetoken.kvcache.dsv4_prefix_tier import _payload_to_tensors
    actual = sum(t.numel() * t.element_size()
                 for t in _payload_to_tensors(payload).values())
    assert tier.bytes_per_page() == actual


def test_dsv4_prefix_tier_warm_does_not_respill_disk_pages(tmp_path):
    pool = _pool()
    n = 3
    full = torch.arange(n * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + n * P, dtype=torch.int32)
    window_slots = pool.translate_full_to_window(full)
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            n * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer

    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    assert tier.put(ids, full) == n * P
    tier.flush()
    # page 0 spilled; pages 1,2 still host-resident
    assert tier.disk_pages_count == 1
    writes_after_put = tier.stats.disk_writes

    # Three warm passes rotate all pages through the 2-page host tier: pages 1,2
    # spill once each (legit first-time writes), and the final pass evicts an
    # already-spilled page from host, which must not write again.
    for _ in range(3):
        tier.prefetch(ids)
        tier.wait_prefetch(ids)
    tier.flush()
    assert tier.stats.disk_writes == writes_after_put + 2


def test_dsv4_prefix_tier_prefetch_skips_gpu_resident_pages(tmp_path):
    pool = _pool()
    full = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(full)
    ids = torch.arange(1000, 1000 + 2 * P, dtype=torch.int32)

    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=1, disk_dir=tmp_path / "host")
    assert tier.put(ids, full) == 2 * P
    tier.flush()

    # A GPU radix match covering both pages must schedule no disk warming.
    assert tier.prefetch(ids, start=2 * P) == 0
    assert tier.stats.prefetch_scheduled == 0
    assert tier.stats.prefetch_gpu_resident == 2

    # Move page 1 to disk so the partial match below has something to warm.
    assert tier.prefetch(ids[:P]) == 1
    tier.wait_prefetch(ids[:P])
    assert tier.stats.prefetch_scheduled == 1

    # A partial GPU match warms only the uncovered tail page.
    assert tier.prefetch(ids, start=P) == 1
    tier.wait_prefetch(ids[P:])
    assert tier.stats.prefetch_scheduled == 2
    assert tier.stats.prefetch_gpu_resident == 3
def test_full_host_tier_without_disk_dir_evicts_drop_style():
    """SGLang semantics: with no disk backing, a full host tier evicts the oldest
    prefix (LRU, drop) instead of failing the put -- the write path never blocks."""
    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2)

    first = torch.arange(2 * P, dtype=torch.int32)
    pool.alloc_swa(first)
    assert tier.put(first, first.clone()) == 2 * P
    tier.flush()
    assert tier.host_pages_count == 2
    first_tokens = tier.page_metadata[0]["token_ids"]
    pool.free_swa(first)  # the tier holds its own copies; recycle the GPU slots

    # A different prefix: capacity is full, no disk dir -- the oldest pages must
    # be dropped (not spilled) and the new put must succeed.
    second = torch.arange(100, 100 + 2 * P, dtype=torch.int32)
    pool.alloc_swa(second)
    assert tier.put(second, second.clone()) == 2 * P
    tier.flush()
    assert tier.host_pages_count == 2
    assert tier.disk_pages_count == 0
    assert tier.stats.evicted_pages >= 1  # the radix evicts whole leaves
    assert tier.page_metadata[next(iter(tier.host_pages))]["token_ids"] != first_tokens

    # The dropped prefix no longer matches.
    assert tier.match_len(first) == 0
    assert tier.match_len(second) == 2 * P


def _wb_pool(num_pages: int):
    """A DSV4 pool sized for ``num_pages`` full pages (the _pool shape, scaled)."""
    from freetoken.kvcache.dsv4_cost_model import dsv4_pool_sizes

    sizes = dsv4_pool_sizes(num_pages=num_pages, args=_args(), swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(
        sizes=sizes, args=_args(), device=torch.device("cpu"), dtype=torch.bfloat16,
        P=P, n_scratch=1,
    )
    pool._init_paged_state(max_running_req=1, radix=True)
    return pool


def test_write_back_defers_host_put_until_gpu_radix_eviction(tmp_path):
    """write_back policy: a committed prefix NEVER touches the host tier at prefill
    time; the copy happens on the eviction path (CacheManager._allocate -> evict_full),
    after which a re-match restores from the host tier (sglang HiCache write_back)."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _wb_pool(8)
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2)
    pt = torch.zeros(4, 8 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=8, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
        host_write_policy="write_back",
    )
    assert cm._host_write_back

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))
    handle = cm.match_req(pending).cuda_handle
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=handle,
    )
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)

    window_slots = pool.translate_full_to_window(pt[0, :2 * P])
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    pool.cmp_pool[2][:2] = torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16)
    pool.idx_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 4) + 9
    expected_window = pool.window_pool[0][window_slots.long()].cpu().clone()
    with cm.lazy_free_region():
        cm.cache_req(req, finished=False)

    # The core write_back promise: the commit wrote NOTHING to the host tier.
    assert cm._host_writeback_pages == 0
    assert tier.host_pages_count == 0

    # Free the tree pin, then apply allocation pressure (one page more than free) so
    # _allocate must evict the tree's 2-page leaf; the write-back hook must copy it
    # to the host tier BEFORE the swa/full slots go back to the free lists.
    cm.unlock(req.cache_handle)
    forced = cm._allocate(6)   # 1 page over free -> must evict the 2-page leaf
    assert tier.host_pages_count == 2
    assert cm._host_writeback_pages == 2
    tier.flush()
    assert tier.match_len(torch.tensor(ids[:2 * P], dtype=torch.int32)) == 2 * P

    # The forced allocation was a phantom consumer: hand its pages (plus the
    # abandoned request's tail page) back so the restore capacity gate
    # (available_size >= restore depth + prompt tail) clears -- the same idle-but-
    # full-pool state the write-through test's seeding cycle leaves behind.
    cm.free_slots = torch.cat([cm.free_slots, forced, pt[0, 2 * P:3 * P][::P].clone()])

    # Poison every live pool buffer: a successful re-match can only come from the
    # host tier's write-back copies.
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan")); pool.cmp_pool[2].fill_(float("nan"))
    pool.idx_pool[1].fill_(float("nan"))

    restored = cm.match_req(pending).cuda_handle
    assert restored.cached_len == 2 * P
    restored_locs = restored.get_matched_indices()
    restored_windows = pool.translate_full_to_window(restored_locs)
    assert torch.equal(
        pool.window_pool[0][restored_windows[-P:].long()].cpu(), expected_window[-P:]
    )
    # The restored pages may land anywhere in the pool: check the seeded layer-1/2
    # state at wherever the first/second restored page actually landed.
    q0 = int(restored_locs[0].item()) // P
    q1 = int(restored_locs[P].item()) // P
    assert torch.equal(
        pool.cmp_pool[1][q0 * 32:(q0 + 1) * 32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )
    assert torch.equal(
        pool.cmp_pool[2][[q0, q1]].cpu(),
        torch.tensor([[31] * 8, [63] * 8], dtype=torch.bfloat16),
    )


def test_write_back_reports_cascaded_tombstone_leaves():
    """evict_full(leaves_out=...) reports (key, value) for LIVE leaves AND for the
    swa-tombstone leaves the unlink cascade exposes -- write-back backs up their
    full KV too (window pages are sentinel-safe: they snapshot as window=None)."""
    from freetoken.kvcache.swa_radix_cache import SWARadixCache

    tree = SWARadixCache(torch.device("cpu"), page_size=P, sliding_window_size=P)
    n = 3 * P
    ids = torch.arange(n, dtype=torch.int32)
    locs = torch.arange(n, dtype=torch.int32)
    # root - A(ids[:2P], internal) - B(ids[2P:], live leaf): two tree nodes.
    tree.insert(ids[:2 * P], locs[:2 * P])
    tree.insert(ids, locs)

    # Tombstone the internal head the way trim/free-swa does: swa freed, full KV kept.
    freed_swa = tree.trim_head_swa(ids, 2 * P)
    assert len(freed_swa) == 2 * P
    assert tree._all_nodes()[0].swa_tombstone  # A is now a swa-tombstone internal node

    leaves_out: list = []
    res = tree.evict_full(n, leaves_out=leaves_out)
    assert len(res.kv_indices) == n
    # Popping B must cascade through the exposed tombstone leaf A -> 2 reported leaves
    # (report order is pop order: B first, then A).
    assert len(leaves_out) == 2
    (k_b, v_b), (k_a, v_a) = leaves_out
    assert torch.equal(k_b, ids[2 * P:]) and torch.equal(v_b, locs[2 * P:])
    assert torch.equal(k_a, ids[:2 * P]) and torch.equal(v_a, locs[:2 * P])


def test_write_back_windowless_restore_recomputes_tail(tmp_path):
    """The production write_back shape: by eviction time the prefix's window KV has
    slid out of the window pool, so the snapshot is window-less and restorable_len
    reports 0. The restore must still land [0, n-P) as an all-tombstone chain with
    cached_len = n-P (only the tail page is recomputed)."""
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _wb_pool(8)
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2)
    pt = torch.zeros(4, 8 * P, dtype=torch.int32)
    cm = CacheManager(
        num_pages=8, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
        host_write_policy="write_back",
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))
    handle = cm.match_req(pending).cuda_handle
    req = Req(
        input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0,
        output_len=0, uid=0, sampling_params=SamplingParams(), cache_handle=handle,
    )
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    orig_locs = pt[0, :2 * P].clone()

    window_slots = pool.translate_full_to_window(pt[0, :2 * P])
    for layer in range(3):
        pool.window_pool[layer][window_slots.long()] = torch.arange(
            2 * P, dtype=torch.bfloat16).unsqueeze(1) + 100 * layer
    pool.cmp_pool[1][:32] = torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7
    with cm.lazy_free_region():
        cm.cache_req(req, finished=False)
    cm.unlock(req.cache_handle)

    # The window slid out of the pool before eviction: tombstone the tree leaf and
    # release its swa slots (the evict_swa-on-internal + child-unlink end state).
    node, matched = cm.prefix_cache.match_prefix_full(torch.tensor(ids[:2 * P], dtype=torch.int32))
    assert matched == 2 * P and not node.swa_tombstone
    pool.free_swa(orig_locs)
    node.swa_tombstone = True
    cm.prefix_cache.swa_evictable -= node.length

    # Evict -> the write-back put must snapshot window-less pages.
    forced = cm._allocate(6)
    assert tier.host_pages_count == 2
    tier.flush()
    assert tier.match_len(torch.tensor(ids[:2 * P], dtype=torch.int32)) == 2 * P
    assert tier.restorable_len(torch.tensor(ids[:2 * P], dtype=torch.int32)) == 0

    cm.free_slots = torch.cat([cm.free_slots, forced])
    for layer in range(3):
        pool.window_pool[layer].fill_(float("nan"))
    pool.cmp_pool[1].fill_(float("nan"))

    restored = cm.match_req(pending).cuda_handle
    # Only the tail page is recomputed: cached_len = 2P - P.
    assert restored.cached_len == P
    locs = restored.get_matched_indices()
    assert len(locs) == P
    # No swa slot was bound (all-tombstone restore) and the compressed state landed.
    assert bool((pool.translate_full_to_window(locs) < 0).all())
    q0 = int(locs[0].item()) // P
    assert torch.equal(
        pool.cmp_pool[1][q0 * 32:(q0 + 1) * 32].cpu(),
        torch.arange(32, dtype=torch.bfloat16).view(32, 1).expand(32, 8) + 7,
    )
    # The published chain is all-tombstone with the canonical locs.
    n2, m2 = cm.prefix_cache.match_prefix_full(torch.tensor(ids[:P], dtype=torch.int32))
    assert m2 == P and n2.swa_tombstone
    assert cm.prefix_cache.check_integrity() is None


