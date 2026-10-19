"""CPU core partition across TP ranks (``--moe-strategy hybrid`` with TP > 1).

Pure helpers of ``cpu_executor.py``: no CUDA, no torch.distributed collective
(the grouping is exercised on real multi-rank boots), no Linux-only syscalls
beyond ``os.sched_getaffinity`` -- skipped on platforms without it.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="needs Linux sched_getaffinity"
)


@pytest.fixture
def fake_cores(monkeypatch):
    """Pin ``physical_core_cpus`` to a deterministic 8-core node."""

    def _cores() -> list[int]:
        return list(range(8))

    import freetoken.moe.cpu_executor as ce

    monkeypatch.setattr(ce, "physical_core_cpus", _cores)
    return ce


def test_single_rank_keeps_every_core(fake_cores):
    assert fake_cores.rank_shared_physical_cores(0, 1) == list(range(8))
    assert fake_cores.rank_shared_physical_cores() == list(range(8))


def test_even_split(fake_cores):
    assert fake_cores.rank_shared_physical_cores(0, 2) == [0, 1, 2, 3]
    assert fake_cores.rank_shared_physical_cores(1, 2) == [4, 5, 6, 7]


def test_uneven_split_gives_early_shares_the_extra_core(fake_cores):
    # 8 cores / 3 ranks -> 3+3+2
    assert fake_cores.rank_shared_physical_cores(0, 3) == [0, 1, 2]
    assert fake_cores.rank_shared_physical_cores(1, 3) == [3, 4, 5]
    assert fake_cores.rank_shared_physical_cores(2, 3) == [6, 7]


def test_shares_are_disjoint_and_complete(fake_cores):
    seen: list[int] = []
    for ordinal in range(5):
        share = fake_cores.rank_shared_physical_cores(ordinal, 5)
        assert share  # 8 >= 5, every rank gets at least one core
        seen.extend(share)
    assert sorted(seen) == list(range(8))


def test_machine_smaller_than_rank_count_falls_back_to_full_set(fake_cores):
    # 8 physical cores but 16 ranks: oversubscription beats an empty share
    assert fake_cores.rank_shared_physical_cores(3, 16) == list(range(8))


def test_auto_sizing_uses_only_the_rank_share(fake_cores):
    cores = fake_cores.rank_shared_physical_cores(1, 2)
    n, core_ids = fake_cores.resolve_threads_and_affinity(0, cores=cores)
    assert n == 4
    assert core_ids == [4, 5, 6, 7]


def test_explicit_count_spreads_the_share_first(fake_cores):
    cores = fake_cores.rank_shared_physical_cores(1, 2)
    n, core_ids = fake_cores.resolve_threads_and_affinity(6, cores=cores)
    assert n == 6
    assert set(core_ids[:4]) == {4, 5, 6, 7}  # the share serves the first four threads
    assert len(set(core_ids)) == 6  # distinct hardware threads before any doubling
