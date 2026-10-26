"""qwen4_exp QSA attention TP geometry: the layer's dims vs its sharded GEMMs and pool.

CPU-only, no multi-GPU: same monkeypatch-the-``_TP_INFO``-global trick as
test_qwen4_exp_gdn.py. Regression for the TP2 CUDA-graph capture crash: the
layer split qkv with checkpoint-wide segment widths ([12288, 512, 512] on
Qwen3.8-Flash-Next) while the col-parallel GEMM emits TP-local ones (6656
total), and reshaped q/k with checkpoint-wide head counts.
"""

import pytest
import torch

from freetoken.distributed import info as dist_info
from freetoken.distributed.info import DistributedInfo
from freetoken.layers import rotary
from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
from freetoken.utils import div_even
from tests.models.qwen4_exp.common import parsed_config

NUM_Q, NUM_KV, HEAD_DIM, HIDDEN = 4, 2, 256, 256
INDEX_PROJ = 640  # 4 index q heads x 128 + 1 index k head x 128, replicated on every rank


@pytest.fixture
def tp_size(monkeypatch):
    def _set(size: int):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, size))
    return _set


def _make_attn(set_tp, size: int) -> Qwen4ExpAttention:
    set_tp(size)
    config = parsed_config(num_q=NUM_Q, num_kv=NUM_KV, head_dim=HEAD_DIM, hidden=HIDDEN)
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"):
            return Qwen4ExpAttention(config, layer_id=3)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()  # the cpu rope must not leak into other tests' cache


def test_tp1_keeps_checkpoint_shapes(tp_size):
    attn = _make_attn(tp_size, 1)
    assert (attn.num_q, attn.num_kv) == (NUM_Q, NUM_KV)
    assert (attn.qo_attn_dim, attn.kv_attn_dim) == (1024, 512)
    assert attn._qkv_split == [2048, 512, 512]
    assert attn.qkv_proj.output_sizes == (2048, 512, 512)
    assert attn.qkv_proj.local_output_size == 3072
    assert attn.o_proj.local_input_size == 1024
    assert attn.o_proj.local_output_size == HIDDEN
    assert attn.indexer.index_qk_proj.local_output_size == INDEX_PROJ


def test_tp2_matches_kv_pool_shard(tp_size):
    # the pool receives spec.num_kv_heads at full width and cuts it per rank itself
    # (kvcache/mha_pool.py); the layer must land on the same local count.
    config = parsed_config(num_q=NUM_Q, num_kv=NUM_KV, head_dim=HEAD_DIM, hidden=HIDDEN)
    (spec,) = config.kv_cache_group_specs()
    assert spec.num_kv_heads == NUM_KV
    pool_kv_heads = div_even(spec.num_kv_heads, 2, allow_replicate=True)

    attn = _make_attn(tp_size, 2)
    assert (attn.num_q, attn.num_kv) == (2, pool_kv_heads)
    assert attn.kv_attn_dim == pool_kv_heads * HEAD_DIM
    assert attn._qkv_split == [1024, 256, 256]
    # every rank's slice of each segment is whole heads, so the fused per-rank
    # q|k|v shards line up with the loader's head-aware checkpoint slicing
    assert attn.qkv_proj.output_sizes == (1024, 256, 256)
    assert attn.qkv_proj.local_output_size == 1536
    assert attn.o_proj.local_input_size == 512
    assert attn.o_proj.local_output_size == HIDDEN
    assert attn.indexer.index_qk_proj.local_output_size == INDEX_PROJ
