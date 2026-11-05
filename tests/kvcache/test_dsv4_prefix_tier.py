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
    cm = CacheManager(
        num_pages=3, page_size=P, page_table=pt, type="swa_radix",
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

    from freetoken.scheduler.cache import CacheManager

    pool = _pool()
    tier = DSV4PrefixTier(pool, page_size=P, capacity_pages=2, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 4 * P, dtype=torch.int32)
    # 2 free pages: the 2-page restore fits, but restore + 1-page prompt tail does not.
    cm = CacheManager(
        num_pages=2, page_size=P, page_table=pt, type="swa_radix",
        swa_pool=pool, sliding_window_size=P, host_prefix_cache=tier,
    )

    ids = list(range(1000, 1000 + 2 * P + 1))
    pending = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))
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
