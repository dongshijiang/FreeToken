import os
from pathlib import Path

import pytest
import torch

from freetoken.kvcache import gds_io


def _sample_tensors():
    return {
        "slab.0000": torch.randn(4, 6, dtype=torch.bfloat16),
        "slab.0001": torch.randn(4, 6, dtype=torch.float32),
        "meta.0000": torch.arange(7, dtype=torch.int64),
    }


def test_raw_page_cpu_roundtrip(tmp_path):
    path = tmp_path / "page-0.pt"
    tensors = _sample_tensors()
    gds_io.write_page(path, tensors, meta={"token_ids": [1, 2, 3]})
    assert gds_io.is_raw_page(path)

    header = gds_io.read_header(path)
    assert header["version"] == gds_io.FORMAT_VERSION
    assert header["meta"] == {"token_ids": [1, 2, 3]}
    assert header["block_offset"] % gds_io.BLOCK_ALIGN == 0
    assert sum(s["nbytes"] for s in header["tensors"].values()) == header["block_nbytes"]

    flat = gds_io.read_page_cpu(path)
    assert set(flat) == set(tensors)
    for name, t in tensors.items():
        assert flat[name].dtype == t.dtype
        assert flat[name].shape == t.shape
        assert torch.equal(flat[name], t)


def test_raw_page_empty_roundtrip(tmp_path):
    path = tmp_path / "page-0.pt"
    gds_io.write_page(path, {})
    assert gds_io.read_page_cpu(path) == {}


def test_byte_checksum_is_deterministic():
    block = torch.randint(0, 255, (1000,), dtype=torch.uint8)
    assert gds_io.byte_checksum(block) == gds_io.byte_checksum(block.clone())
    flipped = block.clone()
    flipped[500] ^= 0xFF
    assert gds_io.byte_checksum(block) != gds_io.byte_checksum(flipped)


def test_raw_page_detects_corruption(tmp_path):
    path = tmp_path / "page-0.pt"
    gds_io.write_page(path, _sample_tensors())
    header = gds_io.read_header(path)
    raw = bytearray(path.read_bytes())
    raw[header["block_offset"] + 5] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        gds_io.read_page_cpu(path)


def test_raw_page_detects_truncation(tmp_path):
    path = tmp_path / "page-0.pt"
    gds_io.write_page(path, _sample_tensors())
    header = gds_io.read_header(path)
    raw = path.read_bytes()
    path.write_bytes(raw[: header["block_offset"] + header["block_nbytes"] - 8])
    with pytest.raises(RuntimeError, match="truncated"):
        gds_io.read_page_cpu(path)


def test_is_raw_page_rejects_torch_save(tmp_path):
    path = tmp_path / "page-0.pt"
    torch.save({"slabs": []}, path)
    assert not gds_io.is_raw_page(path)
    with pytest.raises(ValueError, match="not a raw page"):
        gds_io.read_header(path)


def test_gds_available_returns_bool():
    # No CUDA device assumptions: the probe must answer without raising either way.
    assert isinstance(gds_io.gds_available(), bool)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gds_io.gds_available(),
    reason="GPUDirect Storage needs CUDA and a loaded nvidia_fs module",
)
def test_gds_page_reader_roundtrip():
    import shutil
    import tempfile

    # pytest's tmp_path is usually on tmpfs, which cuFile does not serve; use a
    # disk-backed dir like the availability probe does.
    workdir = tempfile.mkdtemp(prefix="ft-gds-test-", dir="/var/tmp")
    try:
        path = Path(os.path.join(workdir, "page-0.pt"))
        tensors = _sample_tensors()
        gds_io.write_page(Path(path), tensors)

        reader = gds_io.GdsPageReader(torch.device("cuda"))
        try:
            flat = reader.read_page(Path(path))
            assert flat["slab.0000"].device.type == "cuda"
            for name, t in tensors.items():
                assert torch.equal(flat[name].cpu(), t)
        finally:
            reader.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gds_io.gds_available(),
    reason="GPUDirect Storage needs CUDA and a loaded nvidia_fs module",
)
def test_gds_page_writer_roundtrip():
    import shutil
    import tempfile

    workdir = tempfile.mkdtemp(prefix="ft-gds-test-", dir="/var/tmp")
    try:
        path = Path(os.path.join(workdir, "page-0.pt"))
        tensors = {k: t.cuda() for k, t in _sample_tensors().items()}
        writer = gds_io.GdsPageWriter(torch.device("cuda"))
        try:
            header = writer.write(Path(path), tensors)
            assert gds_io.is_raw_page(path)
            assert header["block_offset"] % gds_io.BLOCK_ALIGN == 0

            # Both readers must consume the GDS-written file unchanged.
            flat_cpu = gds_io.read_page_cpu(Path(path))
            reader = gds_io.GdsPageReader(torch.device("cuda"))
            try:
                flat_gpu = reader.read_page(Path(path))
            finally:
                reader.close()
            for name, t in tensors.items():
                assert torch.equal(flat_cpu[name], t.cpu())
                assert torch.equal(flat_gpu[name], t)
        finally:
            writer.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gds_io.gds_available(),
    reason="GPUDirect Storage needs CUDA and a loaded nvidia_fs module",
)
def test_gds_page_writer_handles_empty_page():
    import shutil
    import tempfile

    workdir = tempfile.mkdtemp(prefix="ft-gds-test-", dir="/var/tmp")
    try:
        path = Path(os.path.join(workdir, "page-0.pt"))
        writer = gds_io.GdsPageWriter(torch.device("cuda"))
        try:
            writer.write(Path(path), {})
            assert gds_io.read_page_cpu(Path(path)) == {}
        finally:
            writer.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gds_io.gds_available(),
    reason="GPUDirect Storage needs CUDA and a loaded nvidia_fs module",
)
def test_gds_reader_and_writer_stage_from_worker_threads():
    # Freshly cudaMalloc'd pages cannot be handed to cuFileBufRegister before the
    # GPU has touched them once ("invalid device context"). Server spill writes
    # and restores run on worker threads whose first staging allocation is lazy,
    # which used to fall back to the CPU path on every page. Run writer and
    # reader from threads in a fresh child process to hit that exact case.
    import subprocess
    import sys
    import textwrap

    freetoken_root = str(Path(gds_io.__file__).resolve().parents[2])
    child = textwrap.dedent(
        """
        import shutil, sys, tempfile, threading, time
        from pathlib import Path
        import torch
        from freetoken.kvcache import gds_io

        torch.zeros(1, device="cuda")
        assert gds_io.gds_available()
        work = Path(tempfile.mkdtemp(prefix="ft-gds-race-", dir="/var/tmp"))
        errs = []

        def writer():
            w = gds_io.GdsPageWriter(torch.device("cuda"))
            try:
                w.write(work / "p0.pt",
                        {"slab.0000": torch.randn(1024, device="cuda", dtype=torch.bfloat16)})
            except Exception as e:
                errs.append(("writer", repr(e)))
            finally:
                w.close()

        def reader():
            r = gds_io.GdsPageReader(torch.device("cuda"))
            try:
                path = work / "p0.pt"
                for _ in range(200):
                    if path.is_file():
                        break
                    time.sleep(0.005)
                if path.is_file():
                    r.read_page(path)
            except Exception as e:
                errs.append(("reader", repr(e)))
            finally:
                r.close()

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        shutil.rmtree(work, ignore_errors=True)
        if errs:
            print(errs)
            sys.exit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={**os.environ, "PYTHONPATH": freetoken_root},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"concurrent first registration failed: {result.stdout}{result.stderr}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not gds_io.gds_available(),
    reason="GPUDirect Storage needs CUDA and a loaded nvidia_fs module",
)
def test_gds_reader_handle_cache_invalidates_rewritten_page():
    import shutil
    import tempfile

    workdir = tempfile.mkdtemp(prefix="ft-gds-test-", dir="/var/tmp")
    try:
        path = Path(os.path.join(workdir, "page-0.pt"))
        first = {"slab.0000": torch.arange(8, dtype=torch.int64)}
        second = {"slab.0000": torch.arange(8, dtype=torch.int64) + 100}
        gds_io.write_page(Path(path), first)
        reader = gds_io.GdsPageReader(torch.device("cuda"))
        try:
            assert torch.equal(reader.read_page(Path(path))["slab.0000"].cpu(),
                               first["slab.0000"])
            assert len(reader._handles) == 1
            # Same path, new content: the cached handle must not serve stale bytes.
            gds_io.write_page(Path(path), second)
            assert torch.equal(reader.read_page(Path(path))["slab.0000"].cpu(),
                               second["slab.0000"])
            assert len(reader._handles) == 1
        finally:
            reader.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
