"""DeepSeek-V4-Flash TP2: local attention shapes and loader slice helpers.

No multi-GPU in CI: ``set_tp_info`` is one-shot, so the tests monkeypatch the
``freetoken.distributed.info._TP_INFO`` global that every ``get_tp_info()``
caller reads. The weight helpers are exercised on the CPU with a round-trip
(cat of the rank slices == the full tensor).
"""

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.models.deepseek_v4.attention import Attention
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.weight import shard_cols, shard_rows, vocab_rows


@pytest.fixture
def tp2(monkeypatch):
    from freetoken.distributed import info
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 2))


def _tiny_args(**kw):
    kw.setdefault("dim", 64)
    kw.setdefault("n_heads", 8)
    kw.setdefault("head_dim", 64)
    kw.setdefault("rope_head_dim", 32)
    kw.setdefault("q_lora_rank", 16)
    kw.setdefault("o_lora_rank", 16)
    kw.setdefault("o_groups", 4)
    kw.setdefault("moe_inter_dim", 64)
    kw.setdefault("n_layers", 4)
    kw.setdefault("n_routed_experts", 8)
    kw.setdefault("n_hash_layers", 1)
    kw.setdefault("compress_ratios", (0, 0, 4, 128))
    kw.setdefault("max_seq_len", 64)
    kw.setdefault("window_size", 8)
    return DeepseekV4Args(**kw)


def test_attention_local_shapes(tp2):
    attn = Attention(0, _tiny_args())
    assert attn.n_heads_local == 4
    assert attn.n_groups_local == 2
    assert attn.attn_sink.shape == (4,)
    # wo_a: [group-local rows, group-internal K]; wo_a_k is unchanged by TP.
    assert attn.wo_a.shape == (2 * 16, 8 * 64 // 4)
    # wo_b input stays the FULL group count (LinearRowParallel div_evens it).
    assert attn.wo_b.full_input_size == 4 * 16
    assert attn.wo_b.local_input_size == 32


def test_attention_rejects_group_splitting_head_shard(tp2):
    # 10 heads: rank shard is 5, but heads per group is 2 -> a group is split.
    with pytest.raises(ValueError, match="heads per output group"):
        Attention(0, _tiny_args(n_heads=10))


# ---------------------------------------------------------------------------
# Loader slice helpers: cat(rank slices) == full.
# ---------------------------------------------------------------------------

def _roundtrip(shard, value, *, size, dim, **kw):
    parts = [shard(value, rank=r, size=size, **kw) for r in range(size)]
    return torch.cat(parts, dim=dim)


def test_shard_rows_roundtrip():
    v = torch.arange(2 * 48).view(8, 12).to(torch.float32)
    got = _roundtrip(shard_rows, v, size=2, rows=4, dim=0)
    assert torch.equal(got, v)


def test_shard_cols_roundtrip():
    v = torch.arange(2 * 24).view(6, 8).to(torch.float32)
    got = _roundtrip(shard_cols, v, size=2, cols=4, dim=1)
    assert torch.equal(got, v)


def test_shard_noop_at_tp1():
    v = torch.arange(12).view(3, 4).to(torch.float32)
    assert torch.equal(shard_rows(v, rank=0, size=1, rows=3), v)
    assert torch.equal(shard_cols(v, rank=0, size=1, cols=4), v)


def test_vocab_rows_roundtrip_even():
    v = torch.arange(10 * 3).view(10, 3).to(torch.float32)
    got = _roundtrip(vocab_rows, v, size=2, vocab_size=10, dim=0)
    assert torch.equal(got, v)


def test_vocab_rows_ceil_partition():
    v = torch.arange(9 * 3).view(9, 3).to(torch.float32)
    n = 5  # div_ceil(9, 2)
    assert vocab_rows(v, rank=0, size=2, vocab_size=9).shape == (n, 3)
    r1 = vocab_rows(v, rank=1, size=2, vocab_size=9)
    assert r1.shape == (4, 3)
    assert torch.equal(torch.cat([vocab_rows(v, rank=0, size=2, vocab_size=9), r1]), v)
