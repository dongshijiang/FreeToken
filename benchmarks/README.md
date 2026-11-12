# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_numa_bw.py`** — host bandwidth split by NUMA node: PCIe H2D/D2H, same-NUMA
and cross-NUMA DRAM read. Linux-only (sysfs topology + `sched_setaffinity`); turn off
`kernel.numa_balancing` first or page migration will flatter the cross-NUMA number.

```bash
python benchmarks/bench_numa_bw.py --gpu 0
```

**`bench_gds_page.py`** — GDS raw-page write/read vs the host-staged CPU path vs the
legacy torch.save path, at DSV4 page scale (~17MB, ~170 tensors). Needs CUDA, a
loaded `nvidia_fs` module and a non-LVM disk-backed `--dir` (raw NVMe, e.g.
`/mnt/kvcache`); cuFile does not serve tmpfs or device-mapper mounts. Verify the
active cuFile mode in `/tmp/cufile.log` and `/etc/cufile.json` before comparing
machines — `fs: posix` routes reads through the page cache and the warm numbers
stop being disk numbers.

```bash
sudo env PYTHONPATH=python:. CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/bench_gds_page.py --dir /mnt/kvcache/ft-gds-bench --drop-caches
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
