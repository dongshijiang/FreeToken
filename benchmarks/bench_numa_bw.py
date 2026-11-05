"""Host bandwidth probe: PCIe H2D/D2H, same-NUMA and cross-NUMA DRAM read.

The three ceilings a TP2 offload / hybrid box is actually bounded by:

* H2D -- pinned host -> device linear copy, what the offload gather rides on.
* same-NUMA -- workers and their buffer on one socket.
* cross-NUMA -- workers on one socket streaming a buffer first-touched on the other.

Same measurement shape as ``ft bench bw``'s ceilings (STREAM read, CUDA-event
timed linear copy), just split by NUMA node. Linux only: the topology comes from
/sys/devices/system/node and threads are pinned with os.sched_setaffinity.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_numa_bw.py
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from freetoken.gpu_select import _gpu_numa_node, assign_gpu, bind_assigned_gpu, single_gpu_arg
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.benchbw import _available_ram_bytes
from freetoken.moe.cpu_executor import physical_core_cpus


MIB = 1 << 20


@dataclass(frozen=True)
class Node:
    id: int
    cpus: list[int]  # physical-core reps this process may use on the node
    mem_gib: float


def parse_cpulist(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _node_mem_gib(path: Path) -> float:
    try:
        for line in path.read_text().splitlines():
            if "MemTotal" in line:
                return int(line.split()[-2]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def read_nodes() -> list[Node]:
    """NUMA nodes visible to this process, each with the physical cores it owns."""
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return []
    allowed = set(physical_core_cpus())
    nodes: list[Node] = []
    for entry in sorted(base.glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
        try:
            cpus = parse_cpulist((entry / "cpulist").read_text())
        except OSError:
            continue
        reps = [c for c in cpus if c in allowed]
        if reps:
            nodes.append(Node(int(entry.name[4:]), reps, _node_mem_gib(entry / "meminfo")))
    return nodes


def numa_distance(nodes: list[Node]) -> str:
    """``10 32`` -- local vs remote memory latency ratio the topology expects."""
    try:
        return Path(f"/sys/devices/system/node/node{nodes[0].id}/distance").read_text().strip()
    except OSError:
        return "unknown"


def spread_note(samples: list[float]) -> str:
    return f"   [{samples[0]:.0f}..{samples[-1]:.0f}]" if len(samples) > 1 else ""


def _split(numel: int, workers: int) -> tuple[int, int]:
    n = max(1, min(workers, numel))
    return n, numel // n


def _median(samples: list[float]) -> float:
    s = sorted(samples)
    return s[len(s) // 2]


def _run_pinned(cores: list[int], fn, workers: int) -> list:
    """Run ``fn(i, core)`` once per worker thread, each pinned to its own core.

    Thread-level affinity is the point: sched_setaffinity(0, ...) sets the calling
    thread, so first-touch and stream workers each have to do it themselves.
    """
    results: list = [None] * workers
    errors: list[BaseException] = []

    def wrap(i: int) -> None:
        try:
            os.sched_setaffinity(0, {cores[i]})
            results[i] = fn(i, cores[i])
        except BaseException as exc:  # surface worker failures on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=wrap, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def first_touch(buf: torch.Tensor, cores: list[int]) -> None:
    """Fault every page of ``buf`` from workers pinned across ``cores``, which sets its node."""
    n, per = _split(buf.numel(), len(cores))
    prev = torch.get_num_threads()
    torch.set_num_threads(1)

    def fill(i: int, _core: int) -> None:
        buf.narrow(0, i * per, per).fill_(1)

    try:
        _run_pinned(cores, fill, n)
    finally:
        torch.set_num_threads(prev)


def stream_read(buf: torch.Tensor, cores: list[int], iters: int) -> tuple[float, int]:
    """Aggregate read bandwidth (GB/s) of ``buf`` from workers pinned across ``cores``."""
    n, per = _split(buf.numel(), len(cores))
    prev = torch.get_num_threads()
    torch.set_num_threads(1)  # one single-threaded reduction per worker
    release = threading.Barrier(n + 1)
    done = threading.Barrier(n + 1)

    def reduce(i: int, core: int) -> float:
        os.sched_setaffinity(0, {core})  # without this the worker drifts across sockets
        s = buf.narrow(0, i * per, per)
        acc = float(s.sum())  # warm the slice, and keep the loop below observable
        release.wait()
        for _ in range(iters):
            acc += float(s.sum())
        done.wait()
        return acc

    threads = [threading.Thread(target=reduce, args=(i, cores[i])) for i in range(n)]
    for t in threads:
        t.start()
    try:
        release.wait()
        t0 = time.perf_counter()
        done.wait()
        wall = time.perf_counter() - t0
    finally:
        for t in threads:
            t.join()
        torch.set_num_threads(prev)

    read_bytes = n * per * buf.element_size()
    return read_bytes * iters / wall / 1e9, n


def measure_dram(half_bytes: int, nodes: list[Node], workers: int,
                 iters: int, repeats: int) -> list[dict]:
    """Same- and cross-NUMA read bandwidth over one allocation.

    One buffer, one half first-touched on each node, then each half read from both
    nodes. A fresh allocation per cell does not work: a freed buffer comes back at the
    same virtual address with its pages still on the old node, and fill_ does not
    refault them, so later cells would remeasure the previous node's placement.
    """
    per_half = max(1, half_bytes // 4)
    buf = torch.empty(per_half * 2, dtype=torch.float32)
    halves = [buf.narrow(0, 0, per_half), buf.narrow(0, per_half, per_half)]
    try:
        for node, half in zip(nodes, halves):
            first_touch(half, node.cpus)
        rows = []
        for i, owner in enumerate(nodes):
            for j, reader in enumerate(nodes):
                samples = []
                for _ in range(repeats):
                    gbs, threads = stream_read(halves[i], reader.cpus[:workers], iters)
                    samples.append(gbs)
                rows.append({
                    "read_node": reader.id,
                    "page_node": owner.id,
                    "same": i == j,
                    "read_gbs": _median(samples),
                    "samples": sorted(samples),
                    "threads": threads,
                    "bytes": per_half * 4,
                })
        return rows
    finally:
        del buf


def alloc_pinned_on(cores: list[int], numel: int) -> torch.Tensor:
    """Pinned bf16 buffer whose pages the driver commits on ``cores``' NUMA node.

    cudaHostAlloc faults its pages in the calling thread's context, so the buffer has
    to be allocated from a thread pinned to the node whose memory we want.
    """
    box: list[torch.Tensor] = []
    _run_pinned(cores, lambda _i, _c: box.append(alloc_pinned_tensor(numel, dtype=torch.bfloat16)), 1)
    return box[0]


def measure_pcie(device: torch.device, host_cores: list[int], nbytes: int,
                 iters: int, repeats: int) -> dict:
    """Linear pinned<->device copy bandwidth (GB/s) out of a buffer placed on ``host_cores``."""
    n = max(1, nbytes // 2)  # bf16
    moved = n * 2
    src = alloc_pinned_on(host_cores, n)
    with torch.cuda.device(device):
        dst = torch.empty(n, dtype=torch.bfloat16, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def timed(copy) -> float:
            samples = []
            for _ in range(repeats):
                for _ in range(3):
                    copy()
                torch.cuda.synchronize(device)
                start.record()
                for _ in range(iters):
                    copy()
                end.record()
                end.synchronize()
                samples.append(moved * iters / (start.elapsed_time(end) / 1e3) / 1e9)
            return _median(samples)

        h2d = timed(lambda: dst.copy_(src, non_blocking=True))
        d2h = timed(lambda: src.copy_(dst, non_blocking=True))
    return {"h2d_gbs": h2d, "d2h_gbs": d2h, "bytes": moved, "iters": iters}


def numa_balancing_on() -> bool:
    try:
        return Path("/proc/sys/kernel/numa_balancing").read_text().strip() != "0"
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="GPU UUID or nvidia-smi index (default: the first visible GPU)")
    p.add_argument("--threads", type=int, default=0,
                   help="DRAM workers (0 = one per physical core of the read node)")
    p.add_argument("--bytes", type=int, default=0,
                   help="DRAM buffer size in bytes (0 = max(4 GiB, threads x --per-core-mib));"
                        " split in half, one half first-touched per node")
    p.add_argument("--per-core-mib", type=int, default=64, help="auto buffer size per worker")
    p.add_argument("--iters", type=int, default=16, help="DRAM read passes to time")
    p.add_argument("--repeats", type=int, default=3, help="DRAM samples per cell, median reported")
    p.add_argument("--pcie-mib", type=int, default=256, help="linear PCIe copy size in MiB")
    p.add_argument("--pcie-iters", type=int, default=30, help="linear PCIe copies to time")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("error: Linux only (sysfs NUMA topology + sched_setaffinity)")

    try:
        assign_gpu(args.gpu)
        device = bind_assigned_gpu()
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"error: {e}") from e

    nodes = read_nodes()
    if not nodes:
        raise SystemExit("error: no NUMA topology under /sys/devices/system/node")

    gpu_node = _gpu_numa_node(device.index)
    local = next((n for n in nodes if n.id == gpu_node), nodes[0])
    nodes = [local] + [n for n in nodes if n.id != local.id]
    remote = nodes[1] if len(nodes) > 1 else None

    workers = min(args.threads or len(nodes[0].cpus), *(len(n.cpus) for n in nodes))
    workers = max(1, workers)

    want = args.bytes or max(4 << 30, workers * args.per_core_mib * MIB)
    buf_bytes = min(want, max(1 << 30, _available_ram_bytes() // 8))

    print(f"host {socket.gethostname()}   cuda:{device.index} {torch.cuda.get_device_name(device)}"
          f"   pcie-node {gpu_node}")
    for n in nodes:
        tag = " <- gpu" if n.id == local.id else ""
        print(f"  node{n.id}  {len(n.cpus):4d} cores  {n.mem_gib:7.1f} GiB{tag}")
    print(f"  node distance {numa_distance(nodes)}")
    if numa_balancing_on():
        print("  warn: kernel.numa_balancing is on; it migrates pages between passes and"
              " distorts the cross-NUMA number. `sysctl -w kernel.numa_balancing=0` first.")
    print()

    print(f"DRAM read (STREAM, {workers} workers, {buf_bytes / 2**30:.1f} GiB buffer,"
          f" {args.iters} passes x {args.repeats})")
    rows = measure_dram(buf_bytes // 2, nodes, workers, args.iters, args.repeats)
    for r in rows:
        kind = "same-NUMA " if r["same"] else "cross-NUMA"
        print(f"  {kind}   read node{r['read_node']} <- pages node{r['page_node']}"
              f"  {r['read_gbs']:8.1f} GB/s{spread_note(r['samples'])}")
    if remote:
        same_local = next(r for r in rows if r["read_node"] == local.id and r["same"])
        cross = next(r for r in rows if r["read_node"] == local.id and not r["same"])
        ratio = cross["read_gbs"] / same_local["read_gbs"]
        print(f"  cross / same, read from node{local.id}: {ratio:8.1%}")
        if ratio > 0.9:
            print("  warn: cross-NUMA is within 10% of same-NUMA -- the far half probably"
                  " landed locally (zone shortage, cgroup cpuset, or a first-touch miss).")
    print()

    print(f"PCIe linear ({args.pcie_mib} MiB x {args.pcie_iters}, bf16)")
    try:
        here = measure_pcie(device, local.cpus, args.pcie_mib * MIB, args.pcie_iters, 3)
    except ImportError as e:
        print(f"  n/a ({e})")
        return 0
    print(f"  H2D  pinned@node{local.id} -> cuda{device.index}  {here['h2d_gbs']:8.1f} GB/s")
    print(f"  D2H  cuda{device.index} -> pinned@node{local.id}  {here['d2h_gbs']:8.1f} GB/s")
    if remote:
        far = measure_pcie(device, remote.cpus, args.pcie_mib * MIB, args.pcie_iters, 3)
        print(f"  H2D  pinned@node{remote.id} -> cuda{device.index}  {far['h2d_gbs']:8.1f} GB/s"
              f"   ({far['h2d_gbs'] / here['h2d_gbs']:.1%} of local)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
