"""DSV4 host/SSD HiCache tier.

A DSV4 prefix is not a flat KV page: resuming it needs the compressed KV, the
Lightning-indexer KV, and the compressor carry ring at every page boundary. This tier
therefore stores one immutable *restore segment* per full window page. The segment is
self-contained enough to bind a freshly allocated full page, restore its state, and
continue prefill from the next page boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import dataclasses
import threading
import time
from collections import deque
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from freetoken.utils import align_down, init_logger

from .gds_io import GdsPageReader, GdsPageWriter, gds_available
from . import gds_io
from .radix_cache import RadixPrefixCache

logger = init_logger(__name__)


@dataclass
class DSV4PrefixTierStats:
    put_requests: int = 0
    put_pages: int = 0
    restore_pages: int = 0
    restore_failures: int = 0
    # Request-level hit-rate counters: eligible = GPU miss with a longer host prefix,
    # requests = eligible requests whose restore was published into the GPU tree.
    restore_eligible: int = 0
    restore_requests: int = 0
    restore_tokens: int = 0
    restore_eligible_tokens: int = 0
    # Re-restores of an already-counted request (its restored pages were reclaimed
    # by concurrent restores and it fell back to host again). Real extra work,
    # excluded from the hit-rate numerator so it stays <= eligible.
    restore_repeats: int = 0
    disk_writes: int = 0
    disk_reads: int = 0
    gds_reads: int = 0
    evicted_pages: int = 0
    prefetch_scheduled: int = 0
    prefetch_already_hot: int = 0
    # Matched pages skipped because the GPU radix already covers them; warming
    # those would only burn disk reads (and, on a full host tier, re-spill).
    prefetch_gpu_resident: int = 0
    prefetch_failures: int = 0
    put_ms_total: float = 0.0
    restore_ms_total: float = 0.0
    prefetch_ms_total: float = 0.0
    wait_ms_total: float = 0.0


def _payload_to_tensors(payload: dict) -> dict[str, torch.Tensor]:
    """Flatten a restore-segment payload into zero-padded raw-page names."""
    out: dict[str, torch.Tensor] = {}
    window = payload.get("window")
    if window is not None:
        for layer, t in enumerate(window):
            out[f"window.{layer:04d}"] = t
    for layer, t in payload.get("cmp", {}).items():
        out[f"cmp.{layer:04d}"] = t
    for layer, t in payload.get("idx", {}).items():
        out[f"idx.{layer:04d}"] = t
    for layer, rings in payload.get("rings", {}).items():
        for tier_name, t in rings.items():
            if t is not None:
                out[f"ring.{tier_name}.{layer:04d}"] = t
    return out


def _payload_from_flat(flat: dict[str, torch.Tensor], n_layers: int) -> dict:
    """Rebuild the restore-segment payload structure from raw-page names."""
    window = None
    if any(name.startswith("window.") for name in flat):
        window = [flat[f"window.{layer:04d}"] for layer in range(n_layers)]
    cmp = {
        int(name.split(".")[1]): t for name, t in flat.items() if name.startswith("cmp.")
    }
    idx = {
        int(name.split(".")[1]): t for name, t in flat.items() if name.startswith("idx.")
    }
    rings: dict[int, dict[str, torch.Tensor | None]] = {}
    for name, t in flat.items():
        if not name.startswith("ring."):
            continue
        _, tier_name, layer = name.split(".")
        rings.setdefault(int(layer), {})[tier_name] = t
    return {"window": window, "cmp": cmp, "idx": idx, "rings": rings}


def _timed(name: str):
    """Record wall latency for a HiCache operation without nesting its locking logic."""
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return method(self, *args, **kwargs)
            finally:
                self._record_latency(name, started)
        return wrapper
    return decorator


class DSV4PrefixTier:
    """Host/disk backing store for page-aligned DSV4 prefix restore segments."""

    supports_dsv4 = True
    supports_async = True
    # Backpressure bound: above this many queued puts the caller falls back to a
    # synchronous write-through. Each queued job pins its source GPU pages via the
    # SWA tree, so the bound also caps how much eviction space a burst can hold back.
    # Bursts of simultaneous commits (one per running request) must absorb into the
    # async queue: falling back to a synchronous put stalls the scheduler behind the
    # writer's DMA. 16 covers max-running-requests waves with margin; the pins held
    # while queued are bounded by the same number of request paths.
    ASYNC_PUT_QUEUE_LIMIT = 16

    def __init__(self, pool, *, page_size: int, capacity_pages: int,
                 disk_dir: str | Path | None = None,
                 disk_capacity_pages: int | None = None,
                 use_gds: bool = False) -> None:
        from .dsv4_paged_pool import DSV4PagedKVCache

        if not isinstance(pool, DSV4PagedKVCache):
            raise TypeError("DSV4PrefixTier requires DSV4PagedKVCache")
        if page_size != pool.P:
            raise ValueError(f"DSV4 tier page_size must equal pool.P ({pool.P})")
        self.pool = pool
        self.page_size = int(page_size)
        self.capacity_pages = max(1, int(capacity_pages))
        self.disk_dir = Path(disk_dir).resolve() if disk_dir is not None else None
        self.disk_capacity_pages = (
            max(1, int(disk_capacity_pages)) if disk_capacity_pages is not None else None
        )
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.index = RadixPrefixCache(torch.device("cpu"), page_size=self.page_size)
        self.host_pages: OrderedDict[int, dict] = OrderedDict()
        self.disk_pages: dict[int, Path] = {}
        self.page_metadata: dict[int, dict] = {}
        self.checksums: dict[int, str] = {}
        self.free_page_ids: list[int] = []
        self.next_page_id = 0
        self.stats = DSV4PrefixTierStats()
        self._latency_samples: dict[str, deque[float]] = {
            name: deque(maxlen=256) for name in ("put", "restore", "prefetch", "wait")
        }
        self._warned_restore_failure = False
        self._disk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsv4-prefix-tier")
        self._disk_writes: dict[int, Future] = {}
        # Keep the writers serial: parallel workers shrink the per-put pin window but
        # multiply the concurrent put DMA, and the put's D2H contending with the
        # engine's expert-fetch H2D is exactly what delays first tokens (C8: TTFT
        # +17% with 1 worker, +29% with 4). The pin-hold cost is bounded by the queue
        # limit instead.
        self._put_pool = ThreadPoolExecutor(
            max_workers=int(os.environ.get("FREETOKEN_PUT_WORKERS", "1")),
            thread_name_prefix="dsv4-prefix-put",
        )
        self._put_futures: set[Future] = set()
        self._pending_puts = 0
        # Bumped by clear(); an in-flight put whose generation no longer matches is
        # dropped instead of resurrecting stale pages into the fresh index.
        self._generation = 0
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dsv4-prefix-prefetch"
        )
        self._prefetch_writes: dict[int, Future] = {}
        self._closed = False
        self._lock = threading.RLock()
        # Raw pages keep the GDS restore path available after a restart even when this
        # process cannot use GDS; the CPU reader handles the format either way.
        self.write_raw = bool(use_gds)
        self._gds: GdsPageReader | None = None
        self._gds_writer: GdsPageWriter | None = None
        if use_gds:
            if gds_available() and pool._device.type == "cuda":
                self._gds = GdsPageReader(pool._device)
                self._gds_writer = GdsPageWriter(pool._device)
            else:
                logger.warning(
                    "GPUDirect Storage requested but unavailable (nvidia_fs module or "
                    "torch support missing); DSV4 prefix restores use the host path"
                )
        # The put writer thread gathers pool reads on its OWN CUDA stream, ordered
        # against the engine's commit writes by a per-put ready event (recorded by the
        # scheduler when the put is enqueued) -- not by riding the engine stream, which
        # serialized the gather behind the whole engine queue and inflated TTFT.
        # One shared low-priority gather stream, created lazily on first put.
        self._gather_stream: torch.cuda.Stream | None = None
        # Recent put timing breakdown (queue/event/lock/gather/insert), for diagnosing
        # write-through overhead; bounded ring, read from stats_snapshot.
        self._put_breakdown: deque = deque(maxlen=512)
        self._load_persisted_index()

    @property
    def host_pages_count(self) -> int:
        return len(self.host_pages)

    @property
    def disk_pages_count(self) -> int:
        return len(self.disk_pages)

    def bytes_per_page(self) -> int:
        """Byte size of a fully populated restore segment."""
        P, device = self.page_size, self.pool._device
        n = 0
        for layer_id in range(self.pool._n_layers):
            win = self.pool.window_pool[layer_id]
            n += win.element_size() * win[0].numel() * P
            ratio = self.pool.compress_ratios[layer_id]
            if ratio:
                cmp_t = self.pool.cmp_pool[layer_id]
                n += cmp_t.element_size() * cmp_t[0].numel() * (P // ratio)
                ring = self.pool.state_ring[layer_id]
                n += ring.buffer[0].numel() * ring.ring_size * ring.buffer[0].element_size()
                if ratio == 4:
                    idx_t = self.pool.idx_pool[layer_id]
                    n += idx_t.element_size() * idx_t[0].numel() * (P // ratio)
                    ring = self.pool.indexer_state_ring[layer_id]
                    n += ring.buffer[0].numel() * ring.ring_size * ring.buffer[0].element_size()
        return int(n)

    def stats_snapshot(self) -> dict:
        bytes_per_page = self.bytes_per_page()
        put_p50, put_p95 = self._percentiles("put")
        restore_p50, restore_p95 = self._percentiles("restore")
        prefetch_p50, prefetch_p95 = self._percentiles("prefetch")
        wait_p50, wait_p95 = self._percentiles("wait")
        return {
            **dataclasses.asdict(self.stats),
            "put_ms_p50": round(put_p50, 3),
            "put_ms_p95": round(put_p95, 3),
            "restore_ms_p50": round(restore_p50, 3),
            "restore_ms_p95": round(restore_p95, 3),
            "prefetch_ms_p50": round(prefetch_p50, 3),
            "prefetch_ms_p95": round(prefetch_p95, 3),
            "wait_ms_p50": round(wait_p50, 3),
            "wait_ms_p95": round(wait_p95, 3),
            "disk_read_bytes": self.stats.disk_reads * bytes_per_page,
            "disk_write_bytes": self.stats.disk_writes * bytes_per_page,
            "host_bytes": len(self.host_pages) * bytes_per_page,
            "disk_bytes": len(self.disk_pages) * bytes_per_page,
            "cached_pages": self.index.size_info.total_size // self.page_size,
            "host_pages": len(self.host_pages),
            "disk_pages": len(self.disk_pages),
            "gds_reads": self.stats.gds_reads,
            "pending_puts": self._pending_puts,
            "capacity_pages": self.capacity_pages,
            "disk_capacity_pages": self.disk_capacity_pages,
            **self._put_breakdown_stats(),
        }

    def _put_breakdown_stats(self) -> dict:
        """Mean of each put stage over the recent ring (queue/event/lock/gather/insert)."""
        rows = list(self._put_breakdown)
        if not rows:
            return {}
        out: dict = {"put_samples": len(rows)}
        for key in ("queue_ms", "event_ms", "lock_ms", "gather_ms", "insert_ms", "total_ms"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                out[f"put_{key}_mean"] = round(sum(vals) / len(vals), 1)
        out["put_pages_mean"] = round(
            sum(r["pages"] for r in rows) / len(rows), 1)
        return out

    def _record_latency(self, name: str, started: float) -> None:
        if name not in self._latency_samples:
            name = {"get": "restore", "wait_prefetch": "wait"}.get(name, name)
        ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self._latency_samples[name].append(ms)
        if name == "put":
            self.stats.put_ms_total += ms
        elif name == "restore":
            self.stats.restore_ms_total += ms
        elif name == "prefetch":
            self.stats.prefetch_ms_total += ms
        elif name == "wait":
            self.stats.wait_ms_total += ms

    def _percentiles(self, name: str) -> tuple[float, float]:
        values = sorted(self._latency_samples[name])
        if not values:
            return 0.0, 0.0
        def pct(percent: float) -> float:
            index = min(len(values) - 1, max(0, round((len(values) - 1) * percent)))
            return values[index]
        return pct(0.50), pct(0.95)

    def match_len(self, token_ids: torch.Tensor) -> int:
        return int(self.index.match_prefix(self._cpu(token_ids)).cuda_handle.cached_len)

    def restorable_len(self, token_ids: torch.Tensor) -> int:
        """Deepest page boundary whose segment still carries the window tail.

        Old DSV4 window pages may already have been tombstoned before the commit write;
        their compressed/indexer/ring state remains useful, but they cannot become the
        resumption tail. The last page with window KV defines the safe restore boundary.
        """
        ids = self._cpu(token_ids)
        handle = self.index.match_prefix(ids).cuda_handle
        matched = int(handle.cached_len)
        if matched == 0:
            return 0
        self.index.lock_handle(handle)
        try:
            rows = handle.get_matched_indices()
            last_good = -1
            for page_idx, page_id in enumerate(self._page_ids(rows, matched, self.page_size)):
                if self._page_has_window(page_id):
                    last_good = page_idx
            return (last_good + 1) * self.page_size if last_good >= 0 else 0
        finally:
            self.index.lock_handle(handle, unlock=True)

    def window_page_positions(self, token_ids: torch.Tensor, target: int) -> list[int]:
        """Positions (0-based, page units) of pages within [0, target) whose snapshot still
        carries window KV.

        Restore binds a swa slot only for these pages: head snapshots store window=None
        (the raw KV was discarded when it slid out of the window), so the swa-slot demand
        is the trailing live window, a small constant, not the restore depth.
        """
        if target <= 0:
            return []
        ids = self._cpu(token_ids)[:target]
        handle = self.index.match_prefix(ids).cuda_handle
        matched = min(int(handle.cached_len), target)
        if matched == 0:
            return []
        self.index.lock_handle(handle)
        try:
            rows = handle.get_matched_indices()
            out = []
            for page_idx, page_id in enumerate(self._page_ids(rows, matched, self.page_size)):
                if self._page_has_window(page_id):
                    out.append(page_idx)
            return out
        finally:
            self.index.lock_handle(handle, unlock=True)

    @_timed("prefetch")
    def prefetch(self, token_ids: torch.Tensor, start: int = 0) -> int:
        """Warm disk-resident prefix pages into host RAM before admission.

        Scheduling is best-effort and never blocks the caller on disk I/O. A single worker
        keeps page order stable; `get()` remains synchronous and will wait on a prefetch
        future if it has not completed yet. Pages fully covered by ``start`` (the length
        already served by the GPU radix) are skipped: they need no disk read.
        """
        ids = self._cpu(token_ids)
        aligned = align_down(len(ids), self.page_size)
        if aligned == 0:
            return 0
        # FT_PREFIX_TIER_PREFETCH=0 keeps disk reads off the async prefetch
        # worker so a restore pays its own disk latency synchronously; the GDS
        # A/B benchmark needs that, since the prefetch otherwise wins the race
        # and decides which pages go through cuFile.
        if os.environ.get("FT_PREFIX_TIER_PREFETCH", "1") == "0":
            return 0
        gpu_through = align_down(int(start), self.page_size)
        scheduled = 0
        with self._lock:
            if self._closed:
                return 0
            handle = self.index.match_prefix(ids[:aligned]).cuda_handle
            matched = int(handle.cached_len)
            if matched == 0:
                return 0
            self.index.lock_handle(handle)
            try:
                rows = handle.get_matched_indices()
                for lo in range(0, matched, self.page_size):
                    if lo + self.page_size <= gpu_through:
                        self.stats.prefetch_gpu_resident += 1
                        continue
                    page_id = int(rows[lo].item()) // self.page_size
                    if page_id in self.host_pages or page_id in self._prefetch_writes:
                        self.stats.prefetch_already_hot += 1
                        continue
                    future = self._prefetch_pool.submit(self._warm_page, page_id)
                    self._prefetch_writes[page_id] = future
                    scheduled += 1
                self.stats.prefetch_scheduled += scheduled
                return scheduled
            finally:
                self.index.lock_handle(handle, unlock=True)

    @_timed("wait_prefetch")
    def wait_prefetch(self, token_ids: torch.Tensor | None = None) -> None:
        """Wait for prefetches along a prefix (or all pending prefetches when None)."""
        with self._lock:
            if token_ids is None:
                futures = list(self._prefetch_writes.values())
                self._prefetch_writes.clear()
            else:
                ids = self._cpu(token_ids)
                handle = self.index.match_prefix(ids).cuda_handle
                rows = handle.get_matched_indices() if handle.cached_len else torch.empty(0)
                page_ids = {
                    int(rows[lo].item()) // self.page_size
                    for lo in range(0, len(rows), self.page_size)
                }
                futures = [
                    self._prefetch_writes.pop(page_id)
                    for page_id in page_ids if page_id in self._prefetch_writes
                ]
        # Never hold the tier lock while waiting: _warm_page needs it to insert the page.
        for future in futures:
            future.result()

    @property
    def pending_puts(self) -> int:
        return self._pending_puts

    @property
    def saturated(self) -> bool:
        return self._pending_puts >= self.ASYNC_PUT_QUEUE_LIMIT

    @_timed("put")
    def put(self, token_ids: torch.Tensor, full_locs: torch.Tensor) -> int:
        """Synchronous write-through (the put_async backpressure fallback): issue on
        the CALLER's thread and block until the writer has applied it. The caller is
        the scheduler -- the only CUDA-submitting thread besides the engine itself
        (two submitting threads deadlocked inside libcuda under KV-pool pressure)."""
        ids = self._cpu(token_ids)
        aligned = align_down(len(ids), self.page_size)
        if aligned == 0:
            return 0
        with self._lock:
            if self._closed:
                return 0
            generation = self._generation
            self._pending_puts += 1
        try:
            job = self._issue_put(ids[:aligned].clone(), full_locs,
                                  None, None, generation)
            if job is None:
                return 0
            fut = self._put_pool.submit(self._complete_put, job)
            self._track_put(fut)
            return int(fut.result())
        except RuntimeError:
            with self._lock:
                self._pending_puts -= 1
            return 0  # pool already shut down

    def put_async(self, token_ids: torch.Tensor, full_locs: torch.Tensor,
                  release: Callable[[], None] | None = None,
                  ready: "torch.cuda.Event | None" = None) -> int:
        """Asynchronous write-through: issue the gather inline and return immediately.

        ISSUE runs here on the scheduler thread (async CUDA submissions only, on the
        shared low-priority gather stream); COMPLETE runs on the writer thread and
        only waits on events and touches host memory. Two threads submitting CUDA
        calls to one device deadlocked inside libcuda (cuLaunchKernel from the writer
        vs cuMemcpyHtoDAsync from the engine -- a driver-internal lock inversion) at
        KV-pool pressure >0.96, so the writer must never submit.

        The CALLER guarantees the source GPU slots stay un-evicted until the writer
        has gathered them (the CacheManager pins the committed tree path and drops
        the pin via ``release``, which the writer invokes once the job has run --
        success or failure). ``ready`` is a CUDA event recorded on the engine stream
        AFTER the commit writes it covers; the gather stream waits on it instead of
        the whole engine queue.
        """
        ids = self._cpu(token_ids)
        aligned = align_down(len(ids), self.page_size)
        if aligned == 0:
            if release is not None:
                release()
            return 0
        with self._lock:
            if self._closed:
                if release is not None:
                    release()
                return 0
            matched = int(self.index.match_prefix(ids[:aligned]).cuda_handle.cached_len)
            generation = self._generation
            self._pending_puts += 1
        if matched >= aligned:
            with self._lock:
                self._pending_puts -= 1
            if release is not None:
                release()
            return 0
        try:
            # Freeze the source page locations on the CALLER's thread (the scheduler,
            # on the engine stream): the gather slices by VALUE from these CPU ints.
            job = self._issue_put(ids[:aligned].clone(), full_locs,
                                  release, ready, generation)
        except Exception:
            with self._lock:
                self._pending_puts -= 1
            if release is not None:
                release()
            raise
        if job is None:
            with self._lock:
                self._pending_puts -= 1
            if release is not None:
                release()
            return 0
        try:
            fut = self._put_pool.submit(self._complete_put, job)
            self._track_put(fut)
        except RuntimeError:
            with self._lock:
                self._pending_puts -= 1
            if release is not None:
                release()
            raise
        return aligned - matched

    def _track_put(self, fut: "Future") -> None:
        """Remember an in-flight put so wait_puts can drain them all; parallel workers
        broke the old submit-a-sentinel trick (the sentinel could beat queued jobs)."""
        with self._lock:
            self._put_futures.add(fut)
        fut.add_done_callback(self._untrack_put)

    def _untrack_put(self, fut: "Future") -> None:
        with self._lock:
            self._put_futures.discard(fut)

    def wait_puts(self) -> None:
        """Block until every enqueued async put has been applied."""
        with self._lock:
            if self._closed:
                return
            futures = list(self._put_futures)
        for fut in futures:
            try:
                fut.result()
            except Exception:
                pass  # _complete_put already logged the failure

    def _ensure_gather_stream(self) -> "torch.cuda.Stream | None":
        """One shared low-priority gather stream: it yields the copy engines to the
        engine stream (put DMA jumping the engine's queue delays first tokens;
        measured C8 TTFT +31% with a high-priority stream, +29% with per-worker
        streams). All of its work is submitted from the scheduler thread; the writer
        only waits on its events."""
        if self.pool._device.type != "cuda" or not torch.cuda.is_available():
            return None
        with self._lock:
            if self._gather_stream is None:
                self._gather_stream = torch.cuda.Stream(priority=1)
            return self._gather_stream

    def _issue_put(self, ids: torch.Tensor, full_locs: torch.Tensor,
                   release: Callable[[], None] | None,
                   ready: "torch.cuda.Event | None", generation: int | None) -> dict | None:
        """Runs on the SCHEDULER thread. Matches the tier index (CPU-only snapshot for
        the gather range; the writer re-matches under the tier lock before inserting,
        so puts that landed in between are honored), freezes the source locs, and
        issues the whole gather as async CUDA ops on the shared low-priority gather
        stream. Returns the job for _complete_put, or None when everything is already
        cached (``release`` already handled)."""
        t0 = time.perf_counter()
        aligned = len(ids)
        # Phase 1: snapshot the gather range. No lock held past this point: the
        # matched span is only a lower bound for the gather; insert-time re-matching
        # under the tier lock keeps the index consistent.
        with self._lock:
            t_lock0 = time.perf_counter()
            matched = int(self.index.match_prefix(ids).cuda_handle.cached_len)
            t_lock_ms = (time.perf_counter() - t_lock0) * 1e3
        locs = full_locs[:aligned].detach().to("cpu", dtype=torch.int64).contiguous()
        if len(locs) < aligned:
            raise ValueError("DSV4 prefix tier put got fewer full locs than token ids")
        page_starts = [
            start for start in range(matched, aligned, self.page_size)
            if min(start + self.page_size, aligned) - start == self.page_size
        ]
        stream = self._ensure_gather_stream()
        payloads: list[dict] = []
        t_gather0 = time.perf_counter()
        if stream is not None:
            with torch.cuda.stream(stream):
                if ready is not None:
                    # Order the gather after exactly the commit writes the event
                    # covers; the engine queue keeps running in parallel.
                    stream.wait_event(ready)
                else:
                    ev = torch.cuda.Event()
                    ev.record()
                    stream.wait_event(ev)
                run0 = 0
                while run0 < len(page_starts):
                    base = int(locs[page_starts[run0]])
                    run1 = run0 + 1
                    while (
                        run1 < len(page_starts)
                        and int(locs[page_starts[run1]])
                        == base + (run1 - run0) * self.page_size
                    ):
                        run1 += 1
                    run_locs = locs[
                        page_starts[run0]:page_starts[run1 - 1] + self.page_size
                    ]
                    try:
                        # Synchronous per-slice gather (stream-syncs at the end):
                        # the packed single-staging-copy variant wedged the driver
                        # six times -- see pool.gather_prefix_pages.
                        payloads.extend(self.pool.gather_prefix_pages(run_locs))
                    except AssertionError:
                        seg = run_locs.tolist()
                        seg_base = seg[0]
                        bad = [i for i, v in enumerate(seg) if v != seg_base + i]
                        logger.error(
                            "DSV4 HiCache put non-contiguous run pages [%d,%d) "
                            "matched=%d: base=%d first16=%s first_bad_off=%s",
                            page_starts[run0], page_starts[run1 - 1] + self.page_size,
                            matched, seg_base, seg[:16], bad[:4],
                        )
                        raise
                    run0 = run1
        else:
            # CPU pool: no streams to order; gather synchronously right here.
            run0 = 0
            while run0 < len(page_starts):
                base = int(locs[page_starts[run0]])
                run1 = run0 + 1
                while (
                    run1 < len(page_starts)
                    and int(locs[page_starts[run1]])
                    == base + (run1 - run0) * self.page_size
                ):
                    run1 += 1
                payloads.extend(self.pool.gather_prefix_pages(
                    locs[page_starts[run0]:page_starts[run1 - 1] + self.page_size]))
                run0 = run1
        return {
            "ids": ids,
            "aligned": aligned,
            "matched": matched,
            "page_starts": page_starts,
            "payloads": payloads,
            "generation": generation,
            "release": release,
            "t_issued": time.perf_counter(),
            "lock_ms": round(t_lock_ms, 1),
            "issue_ms": round((time.perf_counter() - t_gather0) * 1e3, 1),
            "total_ms": round((time.perf_counter() - t0) * 1e3, 1),
        }

    def _complete_put(self, job: dict) -> int:
        """Writer thread: the payloads were already gathered on the scheduler thread
        (synchronous per-slice D2H, see _issue_put); this only does the tier-lock
        insert. NO CUDA work here whatsoever -- any driver call or submission from
        this thread risks the libcuda lock inversion against the engine thread."""
        result = 0
        try:
            started = time.perf_counter()
            try:
                t_done0 = time.perf_counter()
                payloads: list[dict] = job["payloads"]
                t_event_ms = (time.perf_counter() - t_done0) * 1e3
                ids = job["ids"]
                aligned = job["aligned"]
                # Phase 3 (tier lock): host bookkeeping + index insert. Re-match under
                # the lock: the issue-time snapshot may be shorter than the tree now
                # is (earlier puts landed while this gather was in flight);
                # insert_prefix's own tree walk dedups whatever already exists. Match
                # and insert in ONE critical section, so no lock_handle is needed to
                # keep matched_rows valid against eviction.
                with self._lock:
                    if job["generation"] is not None and job["generation"] != self._generation:
                        return 0  # the tier was cleared while this put was in flight
                    self.stats.put_requests += 1
                    handle = self.index.match_prefix(ids).cuda_handle
                    tree_len = int(handle.cached_len)
                    matched_rows = (
                        handle.get_matched_indices().clone().to(torch.int64)
                        if tree_len else torch.empty(0, dtype=torch.int64)
                    )
                    parent_page_id = (
                        int(matched_rows[-1].item()) // self.page_size if tree_len else None
                    )
                    t_ins0 = time.perf_counter()
                    new_rows: list[int] = []
                    for start, payload in zip(job["page_starts"], payloads):
                        if start < tree_len:
                            continue  # an earlier put already inserted this page
                        end = start + self.page_size
                        payload["token_ids"] = ids[start:end].clone()
                        self._ensure_host_capacity(1, protect=len(new_rows) // self.page_size)
                        page_id = self._allocate_page_ids(1)[0]
                        self.host_pages[page_id] = payload
                        self.page_metadata[page_id] = {
                            "token_ids": ids[start:end].tolist(),
                            "parent_page_id": parent_page_id,
                            "has_window": payload.get("window") is not None,
                        }
                        new_rows.extend(page_id * self.page_size + offset
                                        for offset in range(self.page_size))
                        parent_page_id = page_id
                        self.stats.put_pages += 1
                    if new_rows:
                        rows = torch.tensor(new_rows, dtype=torch.int64)
                        result = int(self.index.insert_prefix(
                            ids, torch.cat((matched_rows, rows))[:aligned]
                        ).handle.cached_len)
                    else:
                        result = tree_len
                t_insert_ms = (time.perf_counter() - t_ins0) * 1e3
                self._last_put_stages = {
                    "lock_ms": job["lock_ms"],
                    "gather_ms": job["issue_ms"],
                    "insert_ms": round(t_insert_ms, 1),
                }
                logger.info(
                    "DSV4 HiCache put: pages=%d lock=%.0fms gather=%.0fms "
                    "insert=%.0fms total=%.0fms",
                    len(job["page_starts"]), job["lock_ms"], job["issue_ms"],
                    t_insert_ms, job["total_ms"] + (time.perf_counter() - started) * 1e3,
                )
                return result
            finally:
                self._record_latency("put", started)
                stages = getattr(self, "_last_put_stages", {})
                self._put_breakdown.append({
                    "queue_ms": round((started - job["t_issued"]) * 1e3, 1),
                    "event_ms": round(t_event_ms, 1),
                    **stages,
                    "total_ms": round((time.perf_counter() - started) * 1e3, 1),
                    "pages": len(job["page_starts"]),
                })
        except Exception as exc:
            logger.warning("DSV4 HiCache async put failed: %r", exc)
        finally:
            with self._lock:
                self._pending_puts -= 1
            release = job["release"]
            if release is not None:
                try:
                    release()
                except Exception as exc:
                    logger.warning("DSV4 HiCache put release failed: %r", exc)
        return result

    @_timed("get")
    def get(self, token_ids: torch.Tensor, full_locs: torch.Tensor) -> int:
        """Restore all complete host/disk pages into freshly bound full/window pages."""
        ids = self._cpu(token_ids)
        target = min(self.match_len(ids), align_down(len(full_locs), self.page_size))
        if target == 0:
            return 0
        # Wait outside the lock: a prefetch worker needs the lock to insert the warm page.
        self.wait_prefetch(ids)
        pages = target // self.page_size
        with self._lock:
            handle = self.index.match_prefix(ids[:target]).cuda_handle
            self.index.lock_handle(handle)
            try:
                rows = handle.get_matched_indices()
                for page_idx, page_id in enumerate(
                        self._page_ids(rows, pages * self.page_size, self.page_size)):
                    lo = page_idx * self.page_size
                    payload = self._restore_page(page_id)
                    self.pool.scatter_prefix_page(full_locs[lo:lo + self.page_size], payload)
                    self.stats.restore_pages += 1
                return target
            finally:
                self.index.lock_handle(handle, unlock=True)

    @staticmethod
    def _page_ids(rows: torch.Tensor, ntokens: int, page_size: int) -> list[int]:
        """Page id per page-start row, gathered with a single device sync.

        Calling ``rows[lo].item()`` per page costs a CPU-GPU round trip each
        iteration; one strided slice + tolist syncs once for the whole span.
        """
        return (rows[:ntokens:page_size] // page_size).tolist()

    def _restore_page(self, page_id: int) -> dict:
        """A restore-segment payload, preferring the GPUDirect read when possible.

        An I/O error falls back to the CPU read, which verifies the same
        checksums - corrupted bytes raise on both paths and never reach the GPU.
        """
        if self._gds is not None and page_id not in self.host_pages:
            path = self.disk_pages.get(page_id)
            if path is not None:
                try:
                    flat = self._gds.read_page(path)
                    self.stats.gds_reads += 1
                    return _payload_from_flat(flat, self.pool._n_layers)
                except Exception as exc:
                    logger.warning(
                        "GDS restore of DSV4 prefix page %d failed, falling back to CPU: %r",
                        page_id, exc,
                    )
        return self._load_page(page_id)

    def evict(self, pages: int) -> int:
        pages = max(0, min(int(pages), self.index.size_info.evictable_size // self.page_size))
        if pages == 0:
            return 0
        with self._lock:
            row_ids = self.index.evict(pages * self.page_size)
            for row in row_ids[:: self.page_size]:
                self._drop_page(int(row.item()) // self.page_size)
            self.stats.evicted_pages += pages
            return pages

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self.index = RadixPrefixCache(torch.device("cpu"), page_size=self.page_size)
            for page_id in list(self.host_pages):
                self._drop_page(page_id)
            for page_id in list(self.disk_pages):
                self._drop_page(page_id)
            self.host_pages.clear()
            self.disk_pages.clear()
            self.free_page_ids.clear()
            self.next_page_id = 0

    def flush(self) -> None:
        self.wait_puts()
        with self._lock:
            for future in list(self._prefetch_writes.values()):
                future.result()
            self._prefetch_writes.clear()
            for future in list(self._disk_writes.values()):
                future.result()
            self._disk_writes.clear()

    def close(self) -> None:
        self._closed = True
        # Drain in-flight async puts first: their pages must land in host_pages (and
        # their pins release) before the spill pass below snapshots host contents.
        try:
            self._put_pool.shutdown(wait=True)
        except RuntimeError:
            pass
        # Drain prefetches first: a worker may still insert a page after the loop below.
        with self._lock:
            futures = list(self._prefetch_writes.values())
            self._prefetch_writes.clear()
        for future in futures:
            future.result()
        # Host-resident pages are not crash-durable. On orderly shutdown, spill them too so
        # a restarted server can reconstruct the full disk-backed index. Pages already
        # spilled keep their (immutable) disk copy; rewriting them is pure waste.
        if self.disk_dir is not None:
            for page_id, payload in list(self.host_pages.items()):
                self.host_pages.pop(page_id, None)
                if page_id in self.disk_pages:
                    continue
                path = self.disk_dir / f"page-{page_id:016x}.pt"
                self.disk_pages[page_id] = path
                self._disk_writes[page_id] = self._disk_pool.submit(
                    self._write_page, page_id, path, payload
                )
        self.flush()
        self._prefetch_pool.shutdown(wait=True)
        self._disk_pool.shutdown(wait=True)
        if self._gds is not None:
            self._gds.close()
            self._gds = None
        if self._gds_writer is not None:
            self._gds_writer.close()
            self._gds_writer = None

    def pool_signature(self) -> dict:
        """Layout fingerprint persisted with every page; prevents cross-model restores."""
        pool, args = self.pool, self.pool.args
        return {
            "type": "dsv4",
            "page_size": self.page_size,
            "n_layers": pool._n_layers,
            "head_dim": pool.head_dim,
            "index_head_dim": pool.index_head_dim,
            "compress_ratios": list(pool.compress_ratios),
            "n_win_slots": pool.sizes.n_win_slots,
            "cmp_blocks": list(pool.sizes.cmp_blocks),
            "idx_blocks": list(pool.sizes.idx_blocks),
            "state_slots": list(pool.sizes.state_slots),
            "idx_state_slots": list(pool.sizes.idx_state_slots),
            "scale_fmt": getattr(args, "scale_fmt", None),
        }

    def _load_persisted_index(self) -> None:
        """Rebuild the CPU radix index from parent-linked page sidecars."""
        if self.disk_dir is None:
            return
        signature = self.pool_signature()
        metas: dict[int, dict] = {}
        for metadata_path in sorted(self.disk_dir.glob("page-*.json")):
            try:
                meta = json.loads(metadata_path.read_text())
                if meta.get("version") not in (2, gds_io.FORMAT_VERSION):
                    raise ValueError("unsupported metadata version")
                if meta.get("page_size") != self.page_size:
                    raise ValueError("page size mismatch")
                if meta.get("pool_signature") != signature:
                    raise ValueError("pool signature mismatch")
                page_id = int(meta["page_id"])
                if len(meta["token_ids"]) != self.page_size:
                    raise ValueError("token id length mismatch")
                if not metadata_path.with_suffix(".pt").is_file():
                    raise ValueError("payload file missing")
                metas[page_id] = meta
            except Exception as exc:
                logger.warning(
                    "ignoring DSV4 HiCache page metadata %s: %r", metadata_path, exc
                )

        reconstructed: dict[int, list[int]] = {}

        def tokens_for(page_id: int) -> list[int] | None:
            if page_id in reconstructed:
                return reconstructed[page_id]
            meta = metas.get(page_id)
            if meta is None:
                return None
            parent_id = meta.get("parent_page_id")
            if parent_id is None:
                prefix: list[int] = []
            else:
                parent_tokens = tokens_for(int(parent_id))
                if parent_tokens is None:
                    return None
                prefix = list(parent_tokens)
            prefix.extend(int(x) for x in meta["token_ids"])
            reconstructed[page_id] = prefix
            return prefix

        next_page_id = 0
        loaded = 0
        for page_id, meta in sorted(metas.items()):
            token_ids = tokens_for(page_id)
            if token_ids is None:
                logger.warning("skipping DSV4 HiCache orphan page %s", page_id)
                continue
            expected_row = page_id * self.page_size
            # Radix values address the full token prefix, not just this page.
            rows = torch.arange(expected_row + self.page_size, dtype=torch.int64)
            self.index.insert_prefix(
                torch.tensor(token_ids, dtype=torch.int64), rows
            )
            self.disk_pages[page_id] = self.disk_dir / f"page-{page_id:016x}.pt"
            self.page_metadata[page_id] = {
                "token_ids": meta["token_ids"],
                "parent_page_id": meta.get("parent_page_id"),
                "has_window": bool(meta.get("has_window", False)),
            }
            self.checksums[page_id] = meta.get("checksum")
            next_page_id = max(next_page_id, page_id + 1)
            loaded += 1
        self.next_page_id = max(self.next_page_id, next_page_id)
        if loaded:
            logger.info("loaded %d DSV4 HiCache pages from %s", loaded, self.disk_dir)

    def _touch_page(self, page_id: int) -> None:
        if page_id in self.host_pages:
            self.host_pages.move_to_end(page_id)

    def _warm_page(self, page_id: int) -> None:
        try:
            # Load outside the lock: this is a full disk read, and holding the
            # tier lock across it would stall get()/put() on the only prefetch
            # worker's I/O.
            payload = self._load_page(page_id)
            with self._lock:
                if self._closed or page_id in self.host_pages:
                    return
                self._ensure_host_capacity(1)
                self.host_pages[page_id] = payload
                self._touch_page(page_id)
        except Exception as exc:
            self.stats.prefetch_failures += 1
            logger.warning("DSV4 HiCache prefetch failed for page %s: %r", page_id, exc)

    def _page_has_window(self, page_id: int) -> bool:
        pending = self._disk_writes.get(page_id)
        if pending is not None:
            pending.result()
            self._disk_writes.pop(page_id, None)
        meta = self.page_metadata.get(page_id)
        if meta is not None:
            return bool(meta.get("has_window", False))
        return False

    def _cpu(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.detach().cpu().to(torch.int64).contiguous()

    def _allocate_page_ids(self, pages: int) -> list[int]:
        page_ids: list[int] = []
        for _ in range(pages):
            if self.free_page_ids:
                page_ids.append(self.free_page_ids.pop())
            else:
                page_ids.append(self.next_page_id)
                self.next_page_id += 1
        return page_ids

    def _ensure_disk_capacity(self, extra_pages: int) -> None:
        """Make room in the SSD tier by evicting radix LRU leaves. Raises when full."""
        if self.disk_capacity_pages is None:
            return
        while len(self.disk_pages) + extra_pages > self.disk_capacity_pages:
            evicted = self.evict(1)
            if evicted == 0:
                raise RuntimeError(
                    f"DSV4 HiCache disk tier is full ({len(self.disk_pages)}/"
                    f"{self.disk_capacity_pages} pages) and has no evictable prefix"
                )

    def _ensure_host_capacity(self, extra_pages: int, protect: int = 0) -> None:
        """``protect``: pages the CALLER has already inserted for its own in-flight
        put -- drop-eviction must never eat the prefix it is currently building
        (only reachable when capacity < pages-per-put)."""
        while len(self.host_pages) + extra_pages > self.capacity_pages:
            if not self.host_pages:
                raise RuntimeError("DSV4 host prefix tier has no evictable page")
            if self.disk_dir is None:
                # No disk backing: evict drop-style, SGLang HiCache semantics. The
                # host tier is a pure cache -- dropping the oldest prefix costs a
                # recompute on its next use but never blocks (or poisons) the write
                # path. Refusing to evict made every put past a full tier pay the
                # full gather and then fail at insert, inflating TTFT on replay
                # workloads (pass2ON small cells +40-160%).
                evictable = self.index.size_info.evictable_size // self.page_size
                if evictable <= protect or self.evict(extra_pages) == 0:
                    raise RuntimeError("DSV4 host prefix tier has no evictable page")
                continue
            self._ensure_disk_capacity(extra_pages)
            page_id, payload = self.host_pages.popitem(last=False)
            if page_id in self.disk_pages:
                # A spilled page's bytes are immutable and its disk copy is
                # current, so evicting the host copy needs no write. Re-spilling
                # turned every warm cycle of a full host tier into a full extra
                # pass of disk writes over pages that were already on disk.
                continue
            path = self.disk_dir / f"page-{page_id:016x}.pt"
            self.disk_pages[page_id] = path
            self._disk_writes[page_id] = self._disk_pool.submit(
                self._write_page, page_id, path, payload
            )

    def _write_page(self, page_id: int, path: Path, payload: dict) -> Path:
        meta = self.page_metadata[page_id]
        if self.write_raw:
            self._write_raw_page(page_id, path, payload, meta)
            self.stats.disk_writes += 1
            return path
        checksum = _checksum_payload(payload)
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.{page_id}")
        torch.save(
            {"version": 2, "checksum": checksum, "payload": payload}, tmp
        )
        os.replace(tmp, path)
        metadata = {
            "version": 2,
            "page_size": self.page_size,
            "pool_signature": self.pool_signature(),
            "page_id": page_id,
            "token_ids": meta["token_ids"],
            "parent_page_id": meta.get("parent_page_id"),
            "has_window": bool(meta.get("has_window", False)),
            "checksum": checksum,
        }
        self._write_metadata(path, metadata)
        self.checksums[page_id] = checksum
        self.stats.disk_writes += 1
        return path

    def _write_raw_page(self, page_id: int, path: Path, payload: dict, meta: dict) -> None:
        tensors = _payload_to_tensors(payload)
        if self._gds_writer is not None:
            try:
                self._gds_writer.write(path, tensors)
            except Exception as exc:
                # A failed cuFile write falls back to the CPU path so the page
                # still lands on disk; the reader consumes either format.
                logger.warning("GDS page write failed for %s, using CPU path: %r",
                               path, exc)
                gds_io.write_page(path, tensors)
        else:
            gds_io.write_page(path, tensors)
        metadata = {
            "version": gds_io.FORMAT_VERSION,
            "format": "raw",
            "page_size": self.page_size,
            "pool_signature": self.pool_signature(),
            "page_id": page_id,
            "token_ids": meta["token_ids"],
            "parent_page_id": meta.get("parent_page_id"),
            "has_window": bool(meta.get("has_window", False)),
        }
        self._write_metadata(path, metadata)

    def _write_metadata(self, path: Path, metadata: dict) -> None:
        metadata_tmp = path.with_suffix(f".meta.tmp.{os.getpid()}.{threading.get_ident()}")
        metadata_tmp.write_text(json.dumps(metadata, sort_keys=True))
        metadata_path = path.with_suffix(".json")
        os.replace(metadata_tmp, metadata_path)

    def _load_page(self, page_id: int) -> dict:
        pending = self._disk_writes.get(page_id)
        if pending is not None:
            pending.result()
            self._disk_writes.pop(page_id, None)
        if page_id in self.host_pages:
            return self.host_pages[page_id]
        path = self.disk_pages.get(page_id)
        if path is None or not path.is_file():
            raise RuntimeError(f"DSV4 prefix tier page {page_id} is missing")
        if gds_io.is_raw_page(path):
            flat = gds_io.read_page_cpu(path)
            self.stats.disk_reads += 1
            return _payload_from_flat(flat, self.pool._n_layers)
        wrapper = torch.load(path, map_location="cpu", weights_only=True)
        if wrapper.get("version") != 2:
            raise RuntimeError("unsupported DSV4 HiCache payload version")
        payload = wrapper["payload"]
        actual = _checksum_payload(payload)
        expected = wrapper.get("checksum") or self.checksums.get(page_id)
        if actual != expected:
            raise RuntimeError("DSV4 HiCache payload checksum mismatch")
        self.checksums[page_id] = actual
        self.stats.disk_reads += 1
        return payload

    def _drop_page(self, page_id: int) -> None:
        self.host_pages.pop(page_id, None)
        prefetch = self._prefetch_writes.pop(page_id, None)
        if prefetch is not None:
            prefetch.result()
        pending = self._disk_writes.pop(page_id, None)
        if pending is not None:
            pending.result()
        path = self.disk_pages.pop(page_id, None)
        if path is not None:
            path.unlink(missing_ok=True)
        if path is not None:
            path.with_suffix(".json").unlink(missing_ok=True)
        self.page_metadata.pop(page_id, None)
        self.checksums.pop(page_id, None)
        self.free_page_ids.append(page_id)


def _checksum_payload(payload) -> str:
    """Content hash over the restore segment without relying on torch.save layout."""
    digest = hashlib.sha256()

    def visit(value, path: str) -> None:
        if value is None:
            digest.update(b"\x00N" + path.encode())
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"\x00T" + path.encode())
            digest.update(f"{tensor.dtype},{tuple(tensor.shape)},".encode())
            digest.update(tensor.flatten().view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"\x00D" + path.encode())
            for key in sorted(value, key=repr):
                digest.update(repr(key).encode())
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            digest.update(b"\x00L" + path.encode())
            digest.update(struct.pack("<Q", len(value)))
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        else:
            digest.update(b"\x00S" + path.encode())
            digest.update(repr(value).encode())

    visit(payload, "")
    return digest.hexdigest()


def create_dsv4_prefix_tier(pool, *, capacity_gib: float,
                            disk_dir: str | None = None,
                            disk_capacity_gib: float = 0.0,
                            use_gds: bool = False) -> DSV4PrefixTier | None:
    """Create a DSV4 tier from host/SSD budgets. Page byte size is exact for the pool."""
    if capacity_gib <= 0:
        return None
    probe = DSV4PrefixTier(
        pool, page_size=pool.P, capacity_pages=1, disk_dir=disk_dir, use_gds=use_gds
    )
    bytes_per_page = probe.bytes_per_page()
    probe.capacity_pages = max(
        1, int(float(capacity_gib) * (1 << 30) // max(1, bytes_per_page))
    )
    if disk_dir is not None and disk_capacity_gib > 0:
        probe.disk_capacity_pages = max(
            1, int(float(disk_capacity_gib) * (1 << 30) // max(1, bytes_per_page))
        )
    return probe


__all__ = ["DSV4PrefixTier", "DSV4PrefixTierStats", "create_dsv4_prefix_tier"]
