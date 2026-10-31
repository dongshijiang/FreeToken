"""Benchmark DSV4 HiCache TTFT against full prefill / GPU radix hits.

The driver talks to an already-running server. A "hicache" round writes one target
prefix, sends unique fillers to force it out of the GPU radix cache, then sends the
target again. `/v1/stats` deltas classify whether the second request used HiCache.

Run against a serving model:
    python benchmarks/bench_hicache.py --rounds 3
Force a strict recompute baseline by starting the server without
`--enable-host-prefix-cache`; in that case `hicache_hit` rounds will report no
restore delta and should be interpreted as full-prefill baselines.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _stream_chat(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            content = (choices[0].get("delta") or {}).get("content") if choices else None
            if content and first_token is None:
                first_token = time.perf_counter()
    wall = time.perf_counter() - started
    return {
        "wall_s": wall,
        "ttft_s": (first_token - started) if first_token is not None else wall,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def _hicache_stats(base_url: str) -> dict:
    return _http_json(f"{base_url.rstrip('/')}/v1/stats").get("hicache") or {}


class Benchmark:
    def __init__(self, args) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.words = [f"alpha{i:03d} bravo{i:03d} charlie{i:03d}" for i in range(5000)]

    def _target(self, round_index: int) -> str:
        body = " ".join(self.rng.choice(self.words) for _ in range(self.args.prompt_chars // 18))
        return f"HICACHE-TARGET-{round_index:04d}\n{body}\nEnd target {round_index}."

    def _filler(self, round_index: int, filler_index: int) -> str:
        body = " ".join(self.rng.choice(self.words) for _ in range(self.args.prompt_chars // 18))
        return f"HICACHE-FILLER-{round_index:04d}-{filler_index:04d}\n{body}\nEnd filler."

    def run_round(self, round_index: int) -> dict:
        target = self._target(round_index)
        before = _hicache_stats(self.args.base_url)
        cold = _stream_chat(
            self.args.base_url, self.args.model, target, self.args.max_tokens
        )
        gpu_hit = _stream_chat(
            self.args.base_url, self.args.model, target, self.args.max_tokens
        )
        fillers_sent = 0
        for filler_index in range(self.args.max_fillers):
            filler = self._filler(round_index, filler_index)
            _stream_chat(
                self.args.base_url, self.args.model, filler, self.args.max_tokens
            )
            fillers_sent += 1
            after_fill = _hicache_stats(self.args.base_url)
            if after_fill.get("restore_pages", 0) > before.get("restore_pages", 0):
                # The target left GPU cache; the next target request exercises HiCache.
                break
            if filler_index + 1 >= self.args.fillers and (
                filler_index + 1 >= self.args.max_fillers
            ):
                break
        warm = _stream_chat(
            self.args.base_url, self.args.model, target, self.args.max_tokens
        )
        after = _hicache_stats(self.args.base_url)
        restore_pages = after.get("restore_pages", 0) - before.get("restore_pages", 0)
        disk_reads = after.get("disk_reads", 0) - before.get("disk_reads", 0)
        scheduled = after.get("prefetch_scheduled", 0) - before.get("prefetch_scheduled", 0)
        return {
            "round": round_index,
            "cold": cold,
            "gpu_hit": gpu_hit,
            "hicache": warm,
            "restore_pages": restore_pages,
            "disk_reads": disk_reads,
            "prefetch_scheduled": scheduled,
            "fillers_sent": fillers_sent,
            "hicache_used": restore_pages > 0,
        }

    def run(self) -> dict:
        rounds = [self.run_round(index) for index in range(self.args.rounds)]
        used = [row for row in rounds if row["hicache_used"]]
        summary = {
            "rounds": rounds,
            "hicache_used_rounds": len(used),
            "server": self.args.base_url,
            "prompt_chars": self.args.prompt_chars,
            "fillers_per_round": self.args.fillers,
        }
        if used:
            summary.update({
                "cold_ttft_median_s": statistics.median(row["cold"]["ttft_s"] for row in used),
                "hicache_ttft_median_s": statistics.median(row["hicache"]["ttft_s"] for row in used),
                "gpu_hit_ttft_median_s": statistics.median(row["gpu_hit"]["ttft_s"] for row in used),
                "ttft_reduction_median_s": statistics.median(
                    row["cold"]["ttft_s"] - row["hicache"]["ttft_s"] for row in used
                ),
                "speedup_median": statistics.median(
                    row["cold"]["ttft_s"] / max(row["hicache"]["ttft_s"], 1e-9) for row in used
                ),
            })
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt-chars", type=int, default=30000)
    parser.add_argument("--fillers", type=int, default=5, help="minimum filler requests per round")
    parser.add_argument("--max-fillers", type=int, default=20, help="stop early once HiCache restore is observed")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4747)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = Benchmark(args).run()
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
