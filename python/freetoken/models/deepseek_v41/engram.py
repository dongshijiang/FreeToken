"""DeepSeek-V4.1 Engram: compressed-token n-gram hashes + CPU-resident memory tables.

Faithful port of the reference engram (tokenizer normalization contract, per-layer
hash seeds, signed-square-root gate); V4.1 omits the short convolution of earlier
Engram architectures. Engine differences, both forced by the table's size:

  - the ~384M-row ``embed`` tables (fp8 e4m3 + e8m0 rowwise scales, ~203 GB for the
    two layers) stay CPU-resident: :meth:`EngramLayer.lookup` gathers rows on the
    host and dequantizes on the GPU. The host round-trip cannot run inside a CUDA
    graph, so the engine runs decode EAGER when engram layers are present (the
    model flags ``needs_eager_decode``; the engine drops its capture list).
  - the compressed token map is rebuilt from the checkpoint tokenizer at load (the
    hash multipliers depend on the compressed vocabulary size matching 99092).
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
import torch

from freetoken.layers import BaseOP, LinearReplicated

from .args import DeepseekV41Args


def _mmap_safetensor(path: str, key: str) -> tuple["np.memmap", tuple[int, ...]]:
    """Byte-exact mmap view of one safetensors entry. The fp8 engram tables are
    single-byte dtypes, so the raw payload maps 1:1 onto uint8."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header[key]
    assert meta["dtype"] in ("F8_E4M3", "F8_E8M0")
    begin, end = meta["data_offsets"]
    view = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n + begin, shape=(end - begin,))
    # reshape to the stored rank: the gather in lookup must keep [rows, dim] per row
    return view.reshape(tuple(meta["shape"])), tuple(meta["shape"])


def _next_prime(start: int, seen: set[int]) -> int:
    """Find the first unused prime above ``start`` (mirrors the reference's
    dependency-free search; the multipliers/primes contract pins the exact values)."""
    candidate = start + 1
    while True:
        is_prime = candidate >= 2 and (
            candidate == 2
            or (candidate % 2 != 0 and all(candidate % d for d in range(3, math.isqrt(candidate) + 1, 2)))
        )
        if is_prime and candidate not in seen:
            return candidate
        candidate += 1


def _compressed_token_map(tokenizer) -> tuple[tuple[int, ...], int]:
    """Normalize decoded vocabulary entries using the released token-ID contract.

    NFKC -> NFD -> StripAccents -> Lowercase -> whitespace fold, with the space
    token carried through the sentinel so Strip cannot eat it. Distinct normalized
    strings get consecutive compressed ids; every raw id maps to one of them.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    token_map = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        token_map.append(key_to_id[key])
    return tuple(token_map), len(key_to_id)


class NgramHash:
    """Build all Engram layers' hashes from complete, uncached input sequences.

    Pure integer math (gathers / xor / mod), device-agnostic: buffers live where
    the caller's ``input_ids`` live (moved once on the first forward). Returns
    ``[batch, seq, engram_layers, (max_ngram_size-1) * n_heads]`` logical rows.
    """

    def __init__(self, args: DeepseekV41Args):
        self.max_ngram_size = args.engram_max_ngram_size
        self.n_heads = args.engram_n_heads
        self.layer_ids = tuple(args.engram_layer_ids)
        if not self.layer_ids or self.max_ngram_size < 2 or self.n_heads < 1 or args.engram_vocab_size < 2:
            raise ValueError(
                "Engram hashing requires enabled layers, n-gram order >= 2, heads >= 1, "
                "and bucket size >= 2"
            )
        if len(self.layer_ids) != len(args.engram_num_embeddings) or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Engram requires one table row count for each distinct layer ID")
        if not args.model_path:
            raise ValueError("Engram hashing needs args.model_path (the checkpoint tokenizer)")

        from freetoken.utils.hf import load_tokenizer

        self._token_map_values, vocab_size = _compressed_token_map(load_tokenizer(args.model_path))
        if vocab_size != args.engram_compressed_vocab_size:
            raise ValueError(
                "Engram compressed tokenizer vocabulary mismatch: "
                f"got {vocab_size}, expected {args.engram_compressed_vocab_size}; "
                "hash multipliers depend on this size"
            )
        if not 0 <= args.engram_pad_token_id < len(self._token_map_values):
            raise ValueError(
                f"Engram pad token ID {args.engram_pad_token_id} lies outside the tokenizer vocabulary"
            )
        self.pad_id = self._token_map_values[args.engram_pad_token_id]
        primes, offsets, multipliers = [], [], []
        seen_primes: set[int] = set()
        multiplier_bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        for layer_id in self.layer_ids:
            layer_primes = []
            for _ in range(self.max_ngram_size - 1):
                current = args.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, seen_primes)
                    seen_primes.add(current)
                    layer_primes.append(current)
            primes.append(tuple(layer_primes))
            offsets.append(tuple(int(v) for v in np.cumsum([0, *layer_primes[:-1]])))
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(0, multiplier_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers.append(tuple(int(v) for v in values * 2 + 1))
        # [n_layers, 24] / [n_layers, 4]; heads order n-gram sizes 2..max, n_heads buckets each.
        # CPU device is explicit: model init runs under torch.device("meta") and this
        # module is not a BaseOP, so no weight-loader pass materializes these constants.
        self.token_map = torch.tensor(self._token_map_values, dtype=torch.long, device="cpu")
        self.primes = torch.tensor(primes, dtype=torch.long, device="cpu")
        self.offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
        self.multipliers = torch.tensor(multipliers, dtype=torch.long, device="cpu")
        self._device: torch.device | None = None

    def _to(self, device: torch.device) -> None:
        if self._device != device:
            self.token_map = self.token_map.to(device)
            self.primes = self.primes.to(device)
            self.offsets = self.offsets.to(device)
            self.multipliers = self.multipliers.to(device)
            self._device = device

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, S]`` raw tokenizer ids -> ``[B, S, n_layers, hash_heads]`` hash rows.
        Lookback never crosses the sequence start (padded with the pad id, whose
        contribution is deterministic, mirroring the reference). ``token_mask`` marks
        the text rows: image rows (content pad ids above the vocabulary) hash as the
        pad id and block every n-gram that reaches them."""
        if input_ids.ndim != 2:
            raise ValueError("Engram input_ids must be [batch, sequence]")
        self._to(input_ids.device)
        batch, sequence = input_ids.shape
        if token_mask is None:
            compressed = self.token_map[input_ids.long()]
        else:
            # image rows carry pad ids far above the vocabulary: clamp before the
            # table gather, the sentinel below overwrites them anyway
            compressed = torch.where(
                token_mask, self.token_map[input_ids.clamp_max(self.token_map.numel() - 1).long()], -1
            )
        positions = torch.arange(sequence, device=input_ids.device).expand(batch, sequence)
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        tokens = []
        for shift in range(self.max_ngram_size):
            source_positions = positions - shift
            source = compressed.gather(1, source_positions.clamp_min(0))
            # -1 is the image sentinel: an n-gram reaching one hashes as the pad id
            blocked = blocked | (source_positions < 0) | (source == -1)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            head_start = (shift - 1) * self.n_heads
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, head_start : head_start + self.n_heads])
        return torch.cat(hashes, dim=-1) + self.offsets

    __call__ = forward


class EngramLayer(BaseOP):
    """One layer's memory read: CPU fp8 table -> dequantized rows -> signed-sqrt-gated
    residual update over all HC streams."""

    def __init__(self, args: DeepseekV41Args, table_index: int, *, quant_config=None, prefix: str = ""):
        self.layer_id = args.engram_layer_ids[table_index]
        self.num_embeddings = args.engram_num_embeddings[table_index]
        self.hidden_size = args.dim
        self.hc_mult = args.hc_mult
        self.hash_heads = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.head_dim = args.engram_head_dim
        self.eps = args.norm_eps
        self.model_path = args.model_path
        # fused key/value projection over the flattened hash rows
        self.wkv = LinearReplicated(
            self.hash_heads * self.head_dim, self.hidden_size * (self.hc_mult + 1),
            has_bias=False, quant_config=quant_config, prefix=f"{prefix}.wkv",
        )
        self.q_weight = torch.empty(self.hc_mult, self.hidden_size, dtype=torch.bfloat16)
        self.k_weight = torch.empty(self.hc_mult, self.hidden_size, dtype=torch.bfloat16)
        # CPU-side table handles: (np.memmap uint8 view, shape) for the e4m3 payload
        # and the e8m0 exponent bytes, opened on first lookup. Kept off the heap --
        # page-cache-backed reads only (float8 CPU arithmetic is spotty across torch
        # versions, so dequantize on the GPU from the raw bytes).
        self._table: tuple["np.memmap", tuple[int, ...]] | None = None
        self._scales: tuple["np.memmap", tuple[int, ...]] | None = None
        self._device: torch.device | None = None

    def _ensure_table(self, device: torch.device) -> None:
        if self._table is not None:
            return
        with open(os.path.join(self.model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        base = f"layers.{self.layer_id}.engram.embed"
        self._table = _mmap_safetensor(os.path.join(self.model_path, weight_map[f"{base}.weight"]), f"{base}.weight")
        self._scales = _mmap_safetensor(os.path.join(self.model_path, weight_map[f"{base}.scale"]), f"{base}.scale")
        rows, dim = self._table[1]
        assert dim == self.head_dim and self._scales[1] == (rows, dim // 32)
        self._device = device

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather this layer's rows for ``[B, S, hash_heads]`` logical ids (GPU tensor)
        and return dequantized bf16 embeddings ``[B, S, hash_heads * head_dim]`` on the
        GPU. Syncs to the host for the table gather -- eager decode only (the engine
        disables decode graphs when engram is active)."""
        device = hash_ids.device
        self._ensure_table(device)
        self._device = device
        ids = hash_ids.reshape(-1).to("cpu", dtype=torch.int64)
        if bool((ids >= self.num_embeddings).any()):
            raise ValueError(f"Engram hash row out of range for layer {self.layer_id}")
        # Host gather straight off the mmap (OS page cache holds the hot set; the
        # 200GB table never has to be resident).
        raw = torch.from_numpy(self._table[0][ids.numpy()])
        scl = torch.from_numpy(self._scales[0][ids.numpy()]).to(device, non_blocking=False)
        raw = raw.to(device, non_blocking=False)
        # e4m3 payload x per-32-channel e8m0 scale (2^(byte - 127)), fp32 product -> bf16.
        emb = raw.view(torch.float8_e4m3fn).to(torch.float32)
        scale = scl.to(torch.float32).sub_(127.0).exp2_().to(torch.bfloat16)
        groups = self.head_dim // scale.size(-1)
        emb = emb * scale.repeat_interleave(groups, dim=-1)
        return emb.to(torch.bfloat16).view(*hash_ids.shape[:2], self.hash_heads * self.head_dim)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Inject the normalized, signed-square-root-gated memory residual.

        ``x`` is ``[B, S, hc_mult, dim]``; ``embeddings`` is ``[B, S, hash_heads *
        head_dim]`` from :meth:`lookup`. ``token_mask`` marks the text rows: image
        rows keep their streams. Returns the updated streams (no aliasing).
        """
        bsz, seq = x.shape[:2]
        key, value = self.wkv.forward(embeddings).split(
            (self.hc_mult * self.hidden_size, self.hidden_size), dim=-1
        )
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden = x.float()
        weights = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (hidden * weights * key).sum(-1) * rstd * self.hidden_size ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            # image rows keep their streams: the memory read only writes text rows
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        out = hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return out.to(x.dtype).view(bsz, seq, self.hc_mult, self.hidden_size)


__all__ = ["EngramLayer", "NgramHash"]
