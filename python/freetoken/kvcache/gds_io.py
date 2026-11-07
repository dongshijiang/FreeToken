"""GPUDirect Storage reader and raw page format for the prefix tiers.

torch.save pages cannot feed GDS: cuFile moves raw bytes, while a pickle stream
needs CPU deserialization. In "raw" mode a page file stores each tensor's bytes
at a fixed offset in one contiguous block, so a restore is one cuFile pread into
a registered GPU staging buffer plus a device-side checksum check - no host
memory and no CPU in the path. Availability is probed at runtime; callers fall
back to the host path when the nvidia_fs kernel module is missing.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import torch

MAGIC = b"FTPG"
FORMAT_VERSION = 3
BLOCK_ALIGN = 4096

_probe_result: bool | None = None
_probe_lock = threading.Lock()


def gds_available() -> bool:
    """True when pages can be read straight into GPU memory on this machine.

    Probes for torch.cuda.gds and a loaded nvidia_fs module, then does one real
    write/read roundtrip: a cufile.json policy or a filesystem without GDS
    support only fails at I/O time, not at import time.
    """
    global _probe_result
    if _probe_result is not None:
        return _probe_result
    with _probe_lock:
        if _probe_result is None:
            _probe_result = _probe()
    return _probe_result


def _probe() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        try:
            from torch.cuda import gds
        except ImportError:
            return False
        try:
            with open("/proc/modules", "rb") as modules:
                if not any(line.split(b" ", 1)[0] == b"nvidia_fs" for line in modules):
                    return False
        except OSError:
            return False
        src = torch.arange(256, dtype=torch.uint8, device="cuda")
        dst = torch.zeros_like(src)
        gds.gds_register_buffer(src.untyped_storage())
        gds.gds_register_buffer(dst.untyped_storage())
        # tmpfs is commonly unsupported by the allow-listed mounts; probe on a
        # disk-backed filesystem first so a tmpfs default temp dir cannot mask GDS.
        try:
            for probe_dir in ("/var/tmp", None):
                try:
                    with tempfile.TemporaryDirectory(prefix="ft-gds-probe-", dir=probe_dir) as tmp:
                        path = Path(tmp) / "probe.bin"
                        path.write_bytes(b"\0" * (BLOCK_ALIGN + src.numel()))
                        handle = gds.GdsFile(str(path), os.O_CREAT | os.O_RDWR)
                        handle.save_storage(src.untyped_storage(), offset=BLOCK_ALIGN)
                        handle.load_storage(dst.untyped_storage(), offset=BLOCK_ALIGN)
                    if torch.equal(src, dst):
                        return True
                except Exception:
                    continue
            return False
        finally:
            gds.gds_deregister_buffer(src.untyped_storage())
            gds.gds_deregister_buffer(dst.untyped_storage())
    except Exception:
        return False


def byte_checksum(block: torch.Tensor) -> tuple[int, int]:
    """(int64 word sum, xor-fold of the int64 words) over a uint8 block.

    Runs on CPU or CUDA so the GPU restore path can verify without a D2H.
    Deliberately not cryptographic - it guards against torn or corrupted page
    bytes at near-zero cost, matching what the torch.save path protects against.
    """
    flat = block.detach().flatten().view(torch.uint8)
    n = flat.numel()
    if n == 0:
        return 0, 0
    padded = torch.zeros(((n + 63) // 64) * 64, dtype=torch.uint8, device=flat.device)
    padded[:n] = flat
    words = padded.view(torch.int64)
    folded = words
    while folded.numel() > 1:
        if folded.numel() % 2:
            folded = torch.cat((folded, torch.zeros(1, dtype=torch.int64, device=flat.device)))
        folded = torch.bitwise_xor(folded[0::2], folded[1::2])
    return int(words.sum()), int(folded)


def _verify_block(header: dict, block: torch.Tensor, path: Path) -> None:
    block_sum, block_xor = byte_checksum(block[: header["block_nbytes"]])
    if block_sum != header["sum"] or block_xor != header["xor"]:
        raise RuntimeError(
            f"raw page {path} checksum mismatch "
            f"(sum {block_sum}/{header['sum']}, xor {block_xor}/{header['xor']})"
        )


def _tensor_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().flatten().view(torch.uint8)


def _layout(tensors: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor, dict]]:
    """Sorted (name, uint8 byte view, layout entry) triples for a page."""
    layout: list[tuple[str, torch.Tensor, dict]] = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name]
        raw = _tensor_bytes(t)
        layout.append((name, raw, {
            "offset": offset,
            "nbytes": raw.numel(),
            "shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
        }))
        offset += raw.numel()
    return layout


def _build_header(entries: list[tuple[str, torch.Tensor, dict]], block: torch.Tensor,
                  meta: dict | None) -> tuple[dict, bytes]:
    """Serialize the v3 header, converging its self-referential block_offset."""
    block_sum, block_xor = byte_checksum(block)
    header = {
        "magic": MAGIC.decode(),
        "version": FORMAT_VERSION,
        "block_offset": BLOCK_ALIGN,
        "block_nbytes": int(block.numel()),
        "sum": block_sum,
        "xor": block_xor,
        "tensors": {name: spec for name, _, spec in entries},
    }
    if meta:
        header["meta"] = meta
    header_bytes = json.dumps(header).encode()
    # Real DSV4 pools carry ~170 tensors per page, so the header can span several
    # alignment blocks; the tensor block starts on the next boundary after it.
    # block_offset lives inside the JSON, so iterate until the serialized length
    # is stable (one pass in practice).
    while True:
        block_offset = -(-len(header_bytes) // BLOCK_ALIGN) * BLOCK_ALIGN
        if header["block_offset"] == block_offset:
            break
        header["block_offset"] = block_offset
        header_bytes = json.dumps(header).encode()
    return header, header_bytes


def write_page(path: Path, tensors: dict[str, torch.Tensor],
               meta: dict | None = None) -> dict:
    """Write a v3 raw page file atomically. Returns the JSON header dict.

    ``tensors`` maps a stable name to a tensor; names must be zero-padded when
    they carry an index (the reader sorts them lexically). The tensor block is
    padded to BLOCK_ALIGN so the cuFile read starts and ends on an aligned span.
    """
    entries = _layout(tensors)
    blobs = [raw.cpu() for _, raw, _ in entries]
    block = torch.cat(blobs) if blobs else torch.empty(0, dtype=torch.uint8)
    header, header_bytes = _build_header(entries, block, meta)
    block_offset = header["block_offset"]
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\0" * (block_offset - 8 - len(header_bytes)))
            f.write(block.numpy().tobytes())
            padded = (-block.numel()) % BLOCK_ALIGN
            f.write(b"\0" * padded)
            # The GDS reader reads through cuFile in direct mode, which bypasses
            # the page cache; flush so a restore right after a spill cannot see
            # stale disk blocks.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return header


def is_raw_page(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a raw page file")
        (hlen,) = struct.unpack("<I", f.read(4))
        return json.loads(f.read(hlen))


def _dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def _slice_views(block: torch.Tensor, header: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, spec in header["tensors"].items():
        raw = block[spec["offset"]:spec["offset"] + spec["nbytes"]]
        out[name] = raw.view(_dtype(spec["dtype"])).view(spec["shape"])
    return out


def read_page_cpu(path: Path) -> dict[str, torch.Tensor]:
    """CPU read with checksum verification (fallback and prefetch-warm path)."""
    header = read_header(path)
    if header["block_nbytes"] == 0:
        return {}
    with open(path, "rb") as f:
        f.seek(header["block_offset"])
        payload = f.read(header["block_nbytes"])
        if len(payload) != header["block_nbytes"]:
            raise RuntimeError(f"raw page {path} is truncated")
    block = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    _verify_block(header, block, path)
    # Views, not clones: a page carries ~170 tensors and per-tensor clones cost
    # far more in thread-pool dispatch than in bytes. The block stays alive as
    # long as any view does, which is exactly the page's payload size.
    return _slice_views(block, header)


class GdsPageWriter:
    """Writes raw page tensor blocks straight from GPU memory via cuFile.

    The CPU write path stages every page through host RAM (one D2H copy per
    tensor plus a host-side concatenation); this writer concatenates on the GPU
    into a registered staging buffer and issues one cuFile write. Like the
    reader, staging is exactly-sized per distinct block size and registered once.
    """

    def __init__(self, device: torch.device, max_staged: int = 8) -> None:
        from torch.cuda import gds

        self._gds = gds
        self._device = torch.device(device)
        self._staging: dict[int, torch.Tensor] = {}
        self._max_staged = max_staged

    def write(self, path: Path, tensors: dict[str, torch.Tensor],
              meta: dict | None = None) -> dict:
        entries = _layout(tensors)
        total = entries[-1][2]["offset"] + entries[-1][2]["nbytes"] if entries else 0
        if total == 0:
            return write_page(path, tensors, meta)
        staging = self._staging_for(total)
        for _, raw, spec in entries:
            if raw.device != staging.device:
                raw = raw.to(staging.device, non_blocking=False)
            staging[spec["offset"]:spec["offset"] + spec["nbytes"]] = raw
        header, header_bytes = _build_header(entries, staging[:total], meta)
        block_offset = header["block_offset"]
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            # The header goes through the page cache (it is tiny); the block is
            # written by cuFile afterwards. Direct-mode block writes bypass the
            # page cache, so the two regions never alias.
            with open(tmp, "wb") as f:
                f.write(MAGIC)
                f.write(struct.pack("<I", len(header_bytes)))
                f.write(header_bytes)
                f.write(b"\0" * (block_offset - 8 - len(header_bytes)))
            handle = self._gds.GdsFile(str(tmp), os.O_WRONLY)
            handle.save_storage(staging[:total].untyped_storage(), offset=block_offset)
            del handle
            padded = (-total) % BLOCK_ALIGN
            fd = os.open(tmp, os.O_WRONLY)
            try:
                if padded:
                    os.ftruncate(fd, block_offset + total + padded)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return header

    def _staging_for(self, nbytes: int) -> torch.Tensor:
        buf = self._staging.get(nbytes)
        if buf is None:
            if len(self._staging) >= self._max_staged:
                raise RuntimeError(
                    f"more than {self._max_staged} distinct raw page sizes"
                )
            # cuFileBufRegister rejects pages the GPU has never touched with
            # "invalid device context": fresh cudaMalloc pages are not committed
            # until the first kernel access, and a lazy staging allocation on a
            # worker thread hits exactly that case. Zero the buffer once so the
            # pages are backed before registration (we overwrite them anyway).
            buf = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            buf.zero_()
            self._gds.gds_register_buffer(buf.untyped_storage())
            self._staging[nbytes] = buf
        return buf

    def close(self) -> None:
        for buf in self._staging.values():
            try:
                self._gds.gds_deregister_buffer(buf.untyped_storage())
            except Exception:
                pass
        self._staging.clear()


class GdsPageReader:
    """Reads raw page tensor blocks straight into GPU memory via cuFile.

    torch's ``load_storage`` reads ``storage.nbytes`` bytes from a file offset,
    so each distinct block size gets its own exactly-sized staging tensor,
    registered once. Restore pages come in a handful of distinct sizes (window
    bound or not), which keeps the registrations bounded.
    """

    def __init__(self, device: torch.device, max_staged: int = 8,
                 max_handles: int = 64) -> None:
        from torch.cuda import gds

        self._gds = gds
        self._device = torch.device(device)
        self._staging: dict[int, torch.Tensor] = {}
        self._max_staged = max_staged
        # cuFileHandleRegister costs ~2ms per page (measured); pages are read
        # repeatedly, so keep LRU handles keyed by the file's identity. The stat
        # triple invalidates an entry when a page file is rewritten or replaced.
        self._handles: OrderedDict[tuple, object] = OrderedDict()
        self._max_handles = max_handles

    def _handle_for(self, path: Path):
        st = os.stat(path)
        key = (str(path), st.st_ino, st.st_mtime_ns, st.st_size)
        handle = self._handles.get(key)
        if handle is not None:
            self._handles.move_to_end(key)
            return handle
        for stale in [k for k in self._handles if k[0] == key[0]]:
            self._drop_handle(stale)
        handle = self._gds.GdsFile(str(path), os.O_RDONLY)
        self._handles[key] = handle
        while len(self._handles) > self._max_handles:
            self._drop_handle(next(iter(self._handles)))
        return handle

    def _drop_handle(self, key: tuple) -> None:
        handle = self._handles.pop(key, None)
        if handle is None:
            return
        try:
            handle.deregister_handle()
        except Exception:
            pass

    def read_page(self, path: Path) -> dict[str, torch.Tensor]:
        header = read_header(path)
        block_nbytes = header["block_nbytes"]
        if block_nbytes == 0:
            return {}
        staging = self._staging_for(block_nbytes)
        handle = self._handle_for(path)
        handle.load_storage(staging.untyped_storage(), offset=header["block_offset"])
        _verify_block(header, staging, path)
        return _slice_views(staging, header)

    def _staging_for(self, nbytes: int) -> torch.Tensor:
        buf = self._staging.get(nbytes)
        if buf is None:
            if len(self._staging) >= self._max_staged:
                raise RuntimeError(
                    f"more than {self._max_staged} distinct raw page sizes"
                )
            # cuFileBufRegister rejects pages the GPU has never touched with
            # "invalid device context": fresh cudaMalloc pages are not committed
            # until the first kernel access, and a lazy staging allocation on a
            # worker thread hits exactly that case. Zero the buffer once so the
            # pages are backed before registration (we overwrite them anyway).
            buf = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
            buf.zero_()
            self._gds.gds_register_buffer(buf.untyped_storage())
            self._staging[nbytes] = buf
        return buf

    def close(self) -> None:
        for key in list(self._handles):
            self._drop_handle(key)
        for buf in self._staging.values():
            try:
                self._gds.gds_deregister_buffer(buf.untyped_storage())
            except Exception:
                pass
        self._staging.clear()


__all__ = [
    "BLOCK_ALIGN",
    "FORMAT_VERSION",
    "GdsPageReader",
    "GdsPageWriter",
    "byte_checksum",
    "gds_available",
    "is_raw_page",
    "read_page_cpu",
    "write_page",
]
