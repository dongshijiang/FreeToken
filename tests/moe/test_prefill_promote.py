"""Active-set promotion + startup preload: prefill-routed experts must land in
persistent LRU slots with source-identical bytes, free-slot-only (never evicting a
resident or stealing a borrowed prefill-buffer slot), and the startup preload must
seed the pool the same way."""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # borrowed buffer = slots [0, 16), pool = [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=False,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(
    cache: OffloadMoeCache,
    sources: dict[str, list[torch.Tensor]],
    layer_id: int,
    expert_id: int,
    slot: int,
) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    cache.usage[slot] = 1  # real residents always carry usage; usage-0 means free
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


def _routes(cache: OffloadMoeCache, counts: dict[int, int]) -> torch.Tensor:
    ids = [e for e, c in counts.items() for _ in range(c)]
    return torch.tensor(ids, dtype=torch.int32, device=cache.device).reshape(1, -1)


def _assert_resident(cache, sources, layer_id: int, expert_id: int) -> None:
    slot = int(cache.slot_for_id[layer_id, expert_id])
    assert slot >= 2 * E, f"expert {expert_id} of layer {layer_id} not in the pool (slot {slot})"
    assert int(cache.id_of_slot[slot]) == layer_id * E + expert_id
    for name, per_layer in sources.items():
        assert torch.equal(cache.bank_caches[name][slot].cpu(), per_layer[layer_id][expert_id].cpu())


@CUDA
def test_promote_budget_is_the_free_share():
    cache, _ = _make_cache()
    cache.begin_prefill()
    # share = 24 // 3 = 8, free = 24 - 16 buffer slots = 8 -> per-layer budget 2
    assert cache._promote_budget == 2

    cache2, sources = _make_cache()
    _seed_resident(cache2, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache2, sources, layer_id=1, expert_id=2, slot=18)
    _seed_resident(cache2, sources, layer_id=1, expert_id=4, slot=19)
    cache2.begin_prefill()
    assert cache2._promote_budget == 1  # (24 - 16 - 4) // 3


@CUDA
def test_promote_admits_top_routed_from_buffer():
    cache, sources = _make_cache()
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    torch.cuda.synchronize()  # the real path fences via wait_prefill_layer's ready event
    cache.promote_prefill_layer(0, _routes(cache, {2: 5, 6: 3, 0: 1}))
    torch.cuda.synchronize()
    # the two most-routed experts are resident with source-identical bytes; the
    # tail of the histogram stays out (budget 2)
    _assert_resident(cache, sources, 0, 2)
    _assert_resident(cache, sources, 0, 6)
    assert int(cache.slot_for_id[0, 0]) == -1
    # the borrowed buffer slots must stay metadata-free
    assert bool((cache.id_of_slot[: 2 * E] == -1).all())


@CUDA
def test_promote_never_evicts_residents():
    cache, sources = _make_cache()
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    cache.begin_prefill()  # budget = (24 - 16 - 2) // 3 = 2
    cache.prefetch_prefill_layer(0)
    torch.cuda.synchronize()
    cache.promote_prefill_layer(0, _routes(cache, {5: 7, 2: 2, 1: 1, 4: 1}))
    torch.cuda.synchronize()
    # the pre-existing residents keep their slots and bytes
    for e, slot in ((1, 17), (3, 20)):
        assert int(cache.slot_for_id[1, e]) == slot
        assert int(cache.id_of_slot[slot]) == E + e
        for name, per_layer in sources.items():
            assert torch.equal(cache.bank_caches[name][slot].cpu(), per_layer[1][e].cpu())
    _assert_resident(cache, sources, 0, 5)
    _assert_resident(cache, sources, 0, 2)
    assert bool((cache.id_of_slot[: 2 * E] == -1).all())


@CUDA
def test_promote_is_skipped_without_budget_or_buffer():
    cache, _ = _make_cache()
    cache.prefetch_prefill_layer(0)
    torch.cuda.synchronize()
    # no begin_prefill -> budget still 0 -> admission must be a no-op
    cache.promote_prefill_layer(0, _routes(cache, {2: 5}))
    torch.cuda.synchronize()
    assert bool((cache.slot_for_id == -1).all())


@CUDA
def test_preload_seeds_first_experts_of_every_layer(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MOE_PRELOAD", "2")
    cache, sources = _make_cache()
    assert cache._preload_count == 2
    cache.preload_experts()
    for layer_id in range(NUM_LAYERS):
        for e in range(2):
            _assert_resident(cache, sources, layer_id, e)
            slot = int(cache.slot_for_id[layer_id, e])
            assert int(cache.usage[slot]) >= 1
    # nothing landed in the borrowed buffer region, and only the seeded experts
    assert bool((cache.id_of_slot[: 2 * E] == -1).all())
    assert int((cache.id_of_slot >= 0).sum()) == NUM_LAYERS * 2


@CUDA
def test_preload_auto_default_stays_within_the_pool():
    cache, _ = _make_cache()
    # usable pool = 8 slots; auto preload caps at 8 // (2 * 3) = 1 per layer so the
    # three layers cannot evict each other's seeds
    assert cache._preload_count == 1
    cache.preload_experts()
    assert int((cache.id_of_slot >= 0).sum()) == NUM_LAYERS
