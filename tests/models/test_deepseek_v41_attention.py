"""DeepSeek-V4.1 ragged prefill attention: the uneven-topk assembly contract.

The batched path stitches per-request segments into one flat launch; picks are
min(index_topk, that request's block count) wide, so a batch mixing a short and a
long prompt must pad the short part with -1 (the kernel masks -1) before the cat.
The real projection + indexer machinery runs on tiny dims; only the paged backend
is stubbed at its kernel boundary.
"""

import pytest
import torch

from freetoken.core import Context, get_global_ctx, set_global_ctx
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.attention import Attention
from freetoken.models.deepseek_v4.ops import get_freqs_cis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (rmsnorm kernel)")


@pytest.fixture(autouse=True)
def _single_rank():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _args():
    return DeepseekV41Args(
        dim=64, n_layers=1, n_heads=4, q_lora_rank=16, o_lora_rank=8, o_groups=2,
        head_dim=32, rope_head_dim=8, window_size=8, compress_ratios=(2,),
        kv_source_layer_ids=(0,), index_source_layer_ids=(0,),
        index_n_heads=2, index_head_dim=16, index_topk=4,
        norm_eps=1e-6, max_seq_len=64,
    )


class _Backend:
    """The paged kernel boundary: slot addressing passes through, the indexer keys
    and the sparse attend are shape-faithful stubs keyed on per-request block count."""

    def __init__(self, block_widths):
        self._widths = block_widths
        self.captured = None

    def window_slots_of(self, ti, lo, hi):
        return torch.arange(lo, hi)

    def win_cols_to_global(self, win_cols, slot_lut):
        g = slot_lut[win_cols.clamp_min(0)]
        return torch.where(win_cols < 0, torch.full_like(g, -1), g)

    def store_window(self, kv, layer_id, slots):
        pass

    def blocks_to_global(self, blocks, ratio, ti=None, rows=None):
        return blocks.clamp_min(0) * ratio

    def indexer_keys(self, ti, end_blocks, ratio, keys_layer, bsz):
        return torch.zeros(1, self._widths[ti], 8)

    def indexer_prefill_logits(self, q, keys, weights):
        return torch.zeros(1, q.shape[1], keys.shape[1])

    def attend(self, q, layer_id, topk_idxs, n_window, attn_sink, softmax_scale, **kw):
        self.captured = topk_idxs
        return torch.zeros(1, q.shape[1], 4 * 32, device=q.device, dtype=q.dtype)


def _layer():
    torch.manual_seed(7)
    layer = Attention(0, _args())
    for t in layer.state_dict().values():
        t.normal_(0, 0.05)
        t.data = t.data.to(torch.bfloat16).to(DEVICE)  # the loader's serving dtype
    # the group's kv_source duties are covered by the pools; this test drives only
    # the index-source assembly, so drop the compressor after construction
    layer.is_kv_source = False
    layer.compressor = None
    args = _args()
    layer._freqs_cis = get_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0, args.compress_rope_theta,
        args.rope_factor, args.beta_fast, args.beta_slow, DEVICE,
    )
    layer.indexer.bind_paged(None, 0, layer._freqs_cis, DEVICE)
    backend = _Backend({0: 1, 1: 5})  # ti 0 holds 1 live block, ti 1 holds 5
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=1)
        set_global_ctx(ctx)
    ctx.attn_backend = backend
    return layer, backend


@_cuda
def test_ragged_prefill_pads_uneven_topk_picks():
    layer, backend = _layer()
    from freetoken.models.deepseek_v41.compress import CSA2State

    x = torch.randn(1, 10, 64, dtype=torch.bfloat16, device=DEVICE)
    segments = [(0, 4, 0, 0), (4, 6, 1, 0)]
    state = CSA2State()
    out = layer.forward_ragged(x, segments, torch.arange(10, device=DEVICE), state)

    # the 1-block request's picks were padded to the 5-block request's width
    assert state.topk_indices.shape == (1, 10, 4)
    assert state.topk_indices[0, :4, 1:].eq(-1).all()
    assert out.shape == (1, 10, 64)
    # the flat launch saw window (8) + compressed (max picks width 4) columns
    assert backend.captured.shape == (1, 10, 12)
    assert backend.captured[0, :4, 9:].eq(-1).all()
