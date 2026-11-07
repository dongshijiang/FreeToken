from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from freetoken.utils import align_down, div_ceil, init_logger

from . import gds_io
from .gds_io import GdsPageReader, GdsPageWriter
from .radix_cache import RadixPrefixCache

logger = init_logger(__name__)


@dataclass
class PrefixTierStats:
    put_requests: int = 0
    put_tokens: int = 0
    host_restore_tokens: int = 0
    disk_restore_tokens: int = 0
    evicted_tokens: int = 0
    restore_failures: int = 0
    disk_writes: int = 0
    disk_reads: int = 0
    gds_reads: int = 0


class HostPrefixTier:
    """Host/disk backing store for page-aligned prefix-cache segments.

    The radix ``index`` maps token ids to synthetic host row ids. A page id owns
    ``page_size`` consecutive row ids, so restore can copy each pool slab page-by-page.
    Host pages are LRU; when the host budget is exhausted they spill to one ``torch.save``
    file per page. The first version is synchronous and safe: GPU pages are read before
    the caller can free or reuse them, and disk writes happen after the bytes are on host.
    """

    def __init__(
        self,
        pool,
        *,
        page_size: int,
        capacity_tokens: int,
        disk_dir: str | Path | None = None,
        use_gds: bool = False,
    ) -> None:
        if page_size <= 0:
            raise ValueError("host prefix tier page_size must be positive")
        if not getattr(pool, "supports_prefix_tier", False):
            raise ValueError(f"{type(pool).__name__} does not support the host prefix tier")
        self.pool = pool
        self.page_size = int(page_size)
        self.capacity_pages = max(1, div_ceil(int(capacity_tokens), self.page_size))
        self.disk_dir = Path(disk_dir).resolve() if disk_dir is not None else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.index = RadixPrefixCache(torch.device("cpu"), page_size=self.page_size)
        self.slabs = tuple(pool.prefix_tier_slabs())
        self.host_pages: OrderedDict[int, list[torch.Tensor]] = OrderedDict()
        self.disk_pages: dict[int, Path] = {}
        # Disk writes are backgrounded after the GPU bytes are already on host. The single
        # worker preserves order; pending page reads wait on its future.
        self._disk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-tier")
        self._disk_writes: dict[int, Future] = {}
        self.free_page_ids: list[int] = []
        self.next_page_id = 0
        self.stats = PrefixTierStats()
        self._warned_restore_failure = False
        # Raw pages keep the GDS restore path available after a restart even when this
        # process cannot use GDS; the CPU reader handles the format either way.
        self.write_raw = bool(use_gds)
        self._gds: GdsPageReader | None = None
        self._gds_writer: GdsPageWriter | None = None
        if use_gds:
            if gds_io.gds_available():
                device = torch.device(getattr(pool, "device", "cuda"))
                if device.type == "cuda":
                    self._gds = GdsPageReader(device)
                    self._gds_writer = GdsPageWriter(device)
            else:
                logger.warning(
                    "GPUDirect Storage requested but unavailable (nvidia_fs module or "
                    "torch support missing); prefix restores use the host path"
                )

    @property
    def capacity_tokens(self) -> int:
        return self.capacity_pages * self.page_size

    @property
    def host_tokens(self) -> int:
        return len(self.host_pages) * self.page_size

    @property
    def disk_tokens(self) -> int:
        return len(self.disk_pages) * self.page_size

    @property
    def cached_tokens(self) -> int:
        return self.index.size_info.total_size

    def bytes_per_token(self) -> int:
        return sum(
            torch.Size(slab.shape).numel() * slab.dtype.itemsize for slab in self.slabs
        )

    def stats_snapshot(self) -> dict:
        return {
            "put_requests": self.stats.put_requests,
            "put_tokens": self.stats.put_tokens,
            "host_restore_tokens": self.stats.host_restore_tokens,
            "disk_restore_tokens": self.stats.disk_restore_tokens,
            "evicted_tokens": self.stats.evicted_tokens,
            "restore_failures": self.stats.restore_failures,
            "disk_writes": self.stats.disk_writes,
            "disk_reads": self.stats.disk_reads,
            "gds_reads": self.stats.gds_reads,
            "cached_tokens": self.index.size_info.total_size,
            "host_tokens": self.host_tokens,
            "disk_tokens": self.disk_tokens,
            "capacity_tokens": self.capacity_tokens,
        }

    def match_len(self, token_ids: torch.Tensor) -> int:
        return int(self.index.match_prefix(self._cpu_ids(token_ids)).cuda_handle.cached_len)

    def put(self, token_ids: torch.Tensor, gpu_locs: torch.Tensor) -> int:
        """Write a committed GPU prefix into the tier. Returns the stored token length."""
        ids = self._cpu_ids(token_ids)
        aligned = align_down(len(ids), self.page_size)
        if aligned == 0:
            return 0
        if len(gpu_locs) < aligned:
            raise ValueError("prefix tier put got fewer GPU locations than token ids")
        self.stats.put_requests += 1

        match = self.index.match_prefix(ids[:aligned])
        handle = match.cuda_handle
        matched = int(handle.cached_len)
        self.index.lock_handle(handle)
        new_page_ids: list[int] = []
        try:
            self._touch_path(handle)
            suffix = aligned - matched
            if suffix:
                pages = div_ceil(suffix, self.page_size)
                gpu_slice = gpu_locs[matched:aligned]
                # Copy and spill page-by-page so a host budget smaller than one prefix
                # still works; a large all-at-once copy would need a separate staging
                # buffer or would fail before it could evict itself.
                slabs = self.pool.gather_prefix_tier(gpu_slice, pin=False)
                for page_idx in range(pages):
                    self._ensure_host_capacity(1)
                    page_id = self._allocate_page_ids(1)[0]
                    lo = page_idx * self.page_size
                    hi = min(lo + self.page_size, len(slabs[0]))
                    self.host_pages[page_id] = [
                        slab[lo:hi].contiguous() for slab in slabs
                    ]
                    new_page_ids.append(page_id)

            matched_rows = (
                handle.get_matched_indices().clone() if matched else torch.empty(0, dtype=torch.int64)
            )
            new_rows = torch.tensor(
                [page_id * self.page_size + i for page_id in new_page_ids for i in range(self.page_size)],
                dtype=torch.int64,
            )
            row_ids = torch.cat((matched_rows, new_rows))[:aligned]
            result = self.index.insert_prefix(ids[:aligned], row_ids)
            self.stats.put_tokens += max(0, int(result.handle.cached_len) - matched)
            return int(result.handle.cached_len)
        finally:
            self.index.lock_handle(handle, unlock=True)

    def get(
        self,
        token_ids: torch.Tensor,
        gpu_locs: torch.Tensor,
        *,
        start: int = 0,
    ) -> int:
        """Restore a host/disk prefix into ``gpu_locs``. Returns the restored prefix length."""
        ids = self._cpu_ids(token_ids)
        match = self.index.match_prefix(ids)
        handle = match.cuda_handle
        aligned = min(int(handle.cached_len), align_down(len(ids), self.page_size))
        start = align_down(int(start), self.page_size)
        if start >= aligned:
            return aligned
        if len(gpu_locs) < aligned:
            raise ValueError("prefix tier get got fewer GPU locations than token ids")

        self.index.lock_handle(handle)
        rows = handle.get_matched_indices()
        try:
            for lo in range(start, aligned, self.page_size):
                hi = min(lo + self.page_size, aligned)
                row_slice = rows[lo:hi]
                page_id = int(row_slice[0].item()) // self.page_size
                expected = torch.arange(
                    page_id * self.page_size,
                    (page_id + 1) * self.page_size,
                    dtype=torch.int64,
                )
                if not torch.equal(row_slice.cpu(), expected[: hi - lo]):
                    raise RuntimeError("host prefix tier page mapping is not contiguous")
                slabs = self._restore_page(page_id)
                self.pool.scatter_prefix_tier(gpu_locs[lo:hi], slabs)
                if hi - lo == self.page_size:
                    self._touch_page(page_id)
                if page_id in self.host_pages:
                    self.stats.host_restore_tokens += hi - lo
                elif page_id in self.disk_pages:
                    self.stats.disk_restore_tokens += hi - lo
            return aligned
        finally:
            self.index.lock_handle(handle, unlock=True)

    def evict(self, tokens: int) -> int:
        target = align_down(int(tokens), self.page_size)
        evictable = self.index.size_info.evictable_size
        target = min(target, align_down(evictable, self.page_size))
        if target <= 0:
            return 0
        row_ids = self.index.evict(target)
        page_ids = sorted({int(row.item()) // self.page_size for row in row_ids})
        for page_id in page_ids:
            self._drop_page(page_id)
        self.stats.evicted_tokens += target
        return target

    def clear(self) -> None:
        self.index = RadixPrefixCache(torch.device("cpu"), page_size=self.page_size)
        for page_id in list(self.host_pages):
            self._drop_page(page_id)
        for page_id in list(self.disk_pages):
            self._drop_page(page_id)
        self.host_pages.clear()
        self.flush()
        self.disk_pages.clear()
        self.free_page_ids.clear()
        self.next_page_id = 0

    def _cpu_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.detach().cpu().to(torch.int64)
        return ids.contiguous()

    def _allocate_page_ids(self, pages: int) -> list[int]:
        page_ids: list[int] = []
        for _ in range(pages):
            if self.free_page_ids:
                page_ids.append(self.free_page_ids.pop())
            else:
                page_ids.append(self.next_page_id)
                self.next_page_id += 1
        return page_ids

    def _touch_path(self, handle) -> None:
        rows = handle.get_matched_indices()
        for lo in range(0, len(rows), self.page_size):
            page_id = int(rows[lo].item()) // self.page_size
            self._touch_page(page_id)

    def _touch_page(self, page_id: int) -> None:
        if page_id in self.host_pages:
            self.host_pages.move_to_end(page_id)

    def _ensure_host_capacity(self, extra_pages: int) -> None:
        while len(self.host_pages) + extra_pages > self.capacity_pages:
            if not self.host_pages:
                raise RuntimeError("host prefix tier has no evictable page")
            if self.disk_dir is None:
                raise RuntimeError(
                    "host prefix tier is full; enable --host-prefix-cache-dir to spill to disk"
                )
            page_id, slabs = self.host_pages.popitem(last=False)
            path = self.disk_dir / f"page-{page_id:016x}.pt"
            self.disk_pages[page_id] = path
            self._disk_writes[page_id] = self._disk_pool.submit(self._write_page, page_id, path, slabs)

    def _write_page(self, page_id: int, path: Path, slabs: list[torch.Tensor]) -> Path:
        if self.write_raw:
            named = {f"slab.{i:04d}": slab for i, slab in enumerate(slabs)}
            if self._gds_writer is not None:
                try:
                    self._gds_writer.write(path, named)
                except Exception as exc:
                    # A failed cuFile write falls back to the CPU path so the
                    # page still lands on disk; the reader consumes either.
                    logger.warning("GDS page write failed for %s, using CPU path: %r",
                                   path, exc)
                    gds_io.write_page(path, named)
            else:
                gds_io.write_page(path, named)
            self.stats.disk_writes += 1
            return path
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.{page_id}")
        torch.save(
            {"version": 1, "page_size": self.page_size, "slabs": slabs},
            tmp,
        )
        os.replace(tmp, path)
        self.stats.disk_writes += 1
        return path

    def _restore_page(self, page_id: int) -> list[torch.Tensor]:
        """Slabs for a restore, preferring the GPUDirect read when possible.

        An I/O error falls back to the CPU read, which verifies the same
        checksums - corrupted bytes raise on both paths and never reach the GPU.
        """
        if self._gds is not None and page_id not in self.host_pages:
            path = self.disk_pages.get(page_id)
            if path is not None:
                try:
                    slabs = list(self._gds.read_page(path).values())
                    self.stats.gds_reads += 1
                    return slabs
                except Exception as exc:
                    logger.warning(
                        "GDS restore of prefix page %d failed, falling back to CPU: %r",
                        page_id, exc,
                    )
        return self._load_page(page_id)

    def _load_page(self, page_id: int) -> list[torch.Tensor]:
        if page_id in self.host_pages:
            return self.host_pages[page_id]
        pending = self._disk_writes.get(page_id)
        if pending is not None:
            pending.result()
            self._disk_writes.pop(page_id, None)
        path = self.disk_pages.get(page_id)
        if path is None or not path.is_file():
            raise RuntimeError(f"host prefix tier page {page_id} is missing")
        if gds_io.is_raw_page(path):
            flat = gds_io.read_page_cpu(path)
            self.stats.disk_reads += 1
            return [flat[f"slab.{i:04d}"] for i in range(len(flat))]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        slabs = list(payload["slabs"])
        self.stats.disk_reads += 1
        return slabs

    def _drop_page(self, page_id: int) -> None:
        self.host_pages.pop(page_id, None)
        pending = self._disk_writes.pop(page_id, None)
        if pending is not None:
            # A page can be evicted from the radix index while its spill write is queued.
            # Finish it before deleting the file.
            pending.result()
        path = self.disk_pages.pop(page_id, None)
        if path is not None:
            path.unlink(missing_ok=True)
        self.free_page_ids.append(page_id)

    def flush(self) -> None:
        """Wait for queued spill writes. Safe to call repeatedly."""
        for future in list(self._disk_writes.values()):
            future.result()
        self._disk_writes.clear()

    def close(self) -> None:
        self.flush()
        if self._gds is not None:
            self._gds.close()
            self._gds = None
        if self._gds_writer is not None:
            self._gds_writer.close()
            self._gds_writer = None
        self._disk_pool.shutdown(wait=True)


def create_host_prefix_tier(
    pool,
    *,
    page_size: int,
    capacity_gib: float,
    disk_dir: str | None = None,
    use_gds: bool = False,
) -> HostPrefixTier | None:
    if capacity_gib <= 0:
        return None
    if not getattr(pool, "supports_prefix_tier", False):
        logger.warning("Host prefix cache requested but %s does not support it", type(pool).__name__)
        return None
    bytes_per_token = pool.prefix_tier_bytes_per_token()
    capacity_tokens = max(page_size, int(capacity_gib * (1 << 30) // max(1, bytes_per_token)))
    return HostPrefixTier(
        pool,
        page_size=page_size,
        capacity_tokens=capacity_tokens,
        disk_dir=disk_dir,
        use_gds=use_gds,
    )


__all__ = ["HostPrefixTier", "PrefixTierStats", "create_host_prefix_tier"]
