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
