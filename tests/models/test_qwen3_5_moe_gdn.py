"""qwen3_5_moe GatedDeltaNet TP geometry: the op's declared dims vs the state pool shard.

CPU-only, no multi-GPU: ``set_tp_info`` is one-shot, so the tests monkeypatch the
``freetoken.distributed.info._TP_INFO`` global that every ``get_tp_info()`` caller
reads. Shapes follow Qwen3.6 (16 k-heads / 32 v-heads, head_dim 128). Regression for
the TP2 CUDA-graph capture crash: the op declared checkpoint-wide split tables and
gate vectors while the state pool allocated TP-local ones.
"""

import pytest
import torch

from freetoken.distributed import info as dist_info
from freetoken.distributed.info import DistributedInfo
from freetoken.kvcache.linear_state_pool import _linear_local_dims
from freetoken.layers.quantization import NoQuantConfig, QuantKind, QuantScheme, WeightDesc
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, CONV_K, HIDDEN = 16, 32, 128, 4, 2048


class _QkvzSchemeOnly(NoQuantConfig):
    """Weights stay plain (kind NONE); a truthy scheme on in_proj_qkvz is all the
    fused/split branch looks at, but get_quant_method still reads ``scheme.kind``."""

    def scheme_for(self, prefix: str):
        if prefix.endswith("in_proj_qkvz"):
            return QuantScheme(QuantKind.NONE, WeightDesc("bf16", None, None), ("weight",))
        return None


@pytest.fixture
def tp_size(monkeypatch):
    def _set(size: int):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, size))
    return _set


def _make_op(set_tp, size: int, quant_config=None) -> Qwen3_5GatedDeltaNet:
    set_tp(size)
    with torch.device("meta"):
        return Qwen3_5GatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=NUM_K_HEADS, num_v_heads=NUM_V_HEADS,
            head_k_dim=HEAD_DIM, head_v_dim=HEAD_DIM, conv_kernel_size=CONV_K,
            rms_norm_eps=1e-6, layer_id=0, quant_config=quant_config,
        )


def test_tp1_keeps_checkpoint_shapes(tp_size):
    op = _make_op(tp_size, 1)
    assert (op.num_k_heads, op.num_v_heads) == (16, 32)
    assert (op.key_dim, op.value_dim, op.conv_dim) == (2048, 4096, 8192)
    assert op._in_proj_split == [8192, 4096, 32, 32]
    assert op.in_proj.local_output_size == 12352
    assert op.conv1d.weight.shape == (8192, 1, CONV_K)
    assert op.dt_bias.shape == (32,) and op.A_log.shape == (32,)
    assert op.out_proj.local_input_size == 4096
    assert op.out_proj.local_output_size == HIDDEN


def test_tp2_matches_state_pool_shard(tp_size):
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=NUM_K_HEADS,
        num_value_heads=NUM_V_HEADS, key_head_dim=HEAD_DIM, value_head_dim=HEAD_DIM,
        conv_kernel_dim=CONV_K, output_gate="sigmoid",
    )
    _, pool_conv_dim, pool_v_heads = _linear_local_dims(group, 2)
    assert (pool_conv_dim, pool_v_heads) == (4096, 16)

    op = _make_op(tp_size, 2)
    assert (op.num_k_heads, op.num_v_heads) == (8, pool_v_heads)
    assert op.conv_dim == pool_conv_dim
    assert op._in_proj_split == [4096, 2048, 16, 16]
    assert op.in_proj.local_output_size == 6176
    assert op.conv1d.weight.shape == (pool_conv_dim, 1, CONV_K)
    assert op.dt_bias.shape == (pool_v_heads,) and op.A_log.shape == (pool_v_heads,)
    assert op.out_proj.local_input_size == 2048
    assert op.out_proj.local_output_size == HIDDEN


def test_tp2_in_proj_segments_are_head_aligned(tp_size):
    """Each rank's contiguous slice of a segment must be whole heads: with the old conv|z
    segmentation rank 0 got every q/k column and zero v columns for its 16 local v heads."""
    op = _make_op(tp_size, 2)
    assert op.in_proj.output_sizes == (
        8 * HEAD_DIM, 8 * HEAD_DIM, 16 * HEAD_DIM, 16 * HEAD_DIM, 16, 16,
    )


def test_tp2_split_in_proj_branch(tp_size):
    op = _make_op(tp_size, 2, quant_config=_QkvzSchemeOnly())
    assert op._split_in_proj
    assert op.in_proj_qkvz.output_sizes == (1024, 1024, 2048, 2048)
    assert op.in_proj_qkvz.local_output_size == 6144
    assert op.in_proj_ba.local_output_size == 32
