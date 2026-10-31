import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.prefix_tier import HostPrefixTier


def _pool(num_pages=16, page_size=2):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return MHAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=4,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def _ids(values):
    return torch.tensor(values, dtype=torch.int32)


def test_host_prefix_tier_roundtrip_and_disk_spill(tmp_path):
    pool = _pool()
    ids = _ids([11, 12, 13, 14, 15, 16, 17, 18])
    locs = torch.arange(8, dtype=torch.int32)
    for dense in range(2):
        pool.k_cache(dense).view(-1, 2, 4)[locs] = (
            torch.randn(8, 2, 4, dtype=torch.bfloat16) + dense
        )
        pool.v_cache(dense).view(-1, 2, 4)[locs] = (
            torch.randn(8, 2, 4, dtype=torch.bfloat16) + 10 + dense
        )

    tier = HostPrefixTier(pool, page_size=2, capacity_tokens=4, disk_dir=tmp_path / "host")
    assert tier.put(ids, locs) == 8
    assert tier.host_tokens == 4 and tier.disk_tokens == 4
    tier.flush()
    assert len(list((tmp_path / "host").glob("page-*.pt"))) == 2

    # Poison every GPU row; only the host/disk tier can restore them.
    for dense in range(2):
        pool.k_cache(dense).fill_(float("nan"))
        pool.v_cache(dense).fill_(float("nan"))
    restored = tier.get(ids, locs)
    assert restored == 8
    for dense in range(2):
        assert not torch.isnan(pool.k_cache(dense).view(-1, 2, 4)[locs.long()]).any()
        assert not torch.isnan(pool.v_cache(dense).view(-1, 2, 4)[locs.long()]).any()
    assert tier.stats.disk_reads == 2  # two spilled pages


def test_host_prefix_tier_evicts_by_prefix(tmp_path):
    pool = _pool()
    tier = HostPrefixTier(pool, page_size=2, capacity_tokens=8, disk_dir=tmp_path / "host")
    first = _ids([1, 2, 3, 4])
    second = _ids([5, 6, 7, 8])
    locs = torch.arange(4, dtype=torch.int32)
    tier.put(first, locs)
    tier.put(second, locs)
    assert tier.cached_tokens == 8
    assert tier.evict(4) == 4
    assert tier.match_len(first) == 0
    assert tier.match_len(second) == 4


def test_cache_manager_restores_host_prefix(tmp_path):
    from types import SimpleNamespace

    from freetoken.core import Req, SamplingParams
    from freetoken.scheduler.cache import CacheManager

    pool = _pool(num_pages=16, page_size=2)
    tier = HostPrefixTier(pool, page_size=2, capacity_tokens=8, disk_dir=tmp_path / "host")
    pt = torch.zeros(4, 16, dtype=torch.int32)
    cm = CacheManager(16, 2, pt, "radix", host_prefix_cache=tier)

    ids = [1, 2, 3, 4, 5, 6, 7, 8]
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

    # Give every slab row a distinctive value, then commit it into GPU radix + host tier.
    values = torch.arange(32, dtype=torch.bfloat16).view(32, 1, 1).expand(32, 2, 4).contiguous()
    for dense in range(2):
        pool.k_cache(dense).view(-1, 2, 4).copy_(values + 100 * dense)
        pool.v_cache(dense).view(-1, 2, 4).copy_(values + 200 * dense)
    with cm.lazy_free_region():
        cm.cache_req(req, finished=False)

    # Force the GPU prefix out, then poison every physical slab row.
    cm.unlock(req.cache_handle)
    cm.prefix_cache.evict(len(ids))
    for dense in range(2):
        pool.k_cache(dense).fill_(float("nan"))
        pool.v_cache(dense).fill_(float("nan"))

    restored_handle = cm.match_req(pending).cuda_handle
    # match_req excludes the prompt's final token, so only page-aligned [0, 6) is restored.
    assert restored_handle.cached_len == 6
    restored = restored_handle.get_matched_indices()
    for dense in range(2):
        got_k = pool.k_cache(dense).view(-1, 2, 4)[restored.long()]
        got_v = pool.v_cache(dense).view(-1, 2, 4)[restored.long()]
        assert torch.equal(got_k.squeeze(1), values[:6] + 100 * dense)
        assert torch.equal(got_v.squeeze(1), values[:6] + 200 * dense)
