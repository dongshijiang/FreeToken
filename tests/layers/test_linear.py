"""Row-parallel linears count the full bias exactly once under TP > 1.

CPU-only, same monkeypatch-the-``_TP_INFO``-global trick as
tests/models/test_qwen4_exp_attention.py. Regression: the kernel application
folds the bias into every rank's partial sum, so the all-reduce counted it
world_size times -- silent on the bias-free LLM projections, but every vision
tower projection carries a bias, which poisoned the tower's features from the
first block onward and made TP2 multimodal serving describe garbage colors.
"""

import pytest
import torch
from torch.nn import functional as F

from freetoken.distributed import info as dist_info
from freetoken.distributed.info import DistributedInfo
from freetoken.layers.linear import LinearOProj, LinearRowParallel

M, N, K = 5, 6, 8  # rows, full output, full input


@pytest.fixture
def tp_size(monkeypatch):
    def _set(size: int):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, size))
    return _set


class _FakeComm:
    """Stands in for the TP collective: adds the other ranks' precomputed partials."""

    def __init__(self, others: list[torch.Tensor]):
        self._others = others

    def all_reduce(self, y: torch.Tensor) -> torch.Tensor:
        return y + sum(self._others)


def _row_shard(w_full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    return w_full[:, rank * (w_full.shape[1] // size) : (rank + 1) * (w_full.shape[1] // size)].clone()


def _other_partial(x_full: torch.Tensor, w_full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    other = 1 - rank
    k_loc = x_full.shape[1] // size
    return F.linear(
        x_full[:, other * k_loc : (other + 1) * k_loc], _row_shard(w_full, other, size)
    )


def test_row_parallel_tp2_counts_bias_once(tp_size):
    tp_size(2)
    torch.manual_seed(0)
    w_full, b_full, x_full = torch.randn(N, K), torch.randn(N), torch.randn(M, K)
    for rank in (0, 1):
        layer = LinearRowParallel(K, N, has_bias=True)
        layer.weight.copy_(_row_shard(w_full, rank, 2))
        layer.bias.copy_(b_full)
        layer._comm = _FakeComm([_other_partial(x_full, w_full, rank, 2)])
        k_loc = K // 2
        out = layer.forward(x_full[:, rank * k_loc : (rank + 1) * k_loc])
        assert torch.allclose(out, F.linear(x_full, w_full, b_full), atol=1e-5)


def test_o_proj_tp2_counts_bias_once(tp_size):
    tp_size(2)
    torch.manual_seed(1)
    w_full, b_full, x_full = torch.randn(N, K), torch.randn(N), torch.randn(M, K)
    for rank in (0, 1):
        layer = LinearOProj(K, N, has_bias=True)
        layer.weight.copy_(_row_shard(w_full, rank, 2))
        layer.bias.copy_(b_full)
        layer._comm = _FakeComm([_other_partial(x_full, w_full, rank, 2)])
        k_loc = K // 2
        out = layer.forward(x_full[:, rank * k_loc : (rank + 1) * k_loc])
        assert torch.allclose(out, F.linear(x_full, w_full, b_full), atol=1e-5)


def test_row_parallel_tp2_bias_free_unchanged(tp_size):
    tp_size(2)
    torch.manual_seed(2)
    w_full, x_full = torch.randn(N, K), torch.randn(M, K)
    for rank in (0, 1):
        layer = LinearRowParallel(K, N, has_bias=False)
        layer.weight.copy_(_row_shard(w_full, rank, 2))
        layer._comm = _FakeComm([_other_partial(x_full, w_full, rank, 2)])
        k_loc = K // 2
        out = layer.forward(x_full[:, rank * k_loc : (rank + 1) * k_loc])
        assert torch.allclose(out, F.linear(x_full, w_full), atol=1e-5)


def test_row_parallel_tp1_keeps_fused_bias(tp_size):
    tp_size(1)
    torch.manual_seed(3)
    w_full, b_full, x_full = torch.randn(N, K), torch.randn(N), torch.randn(M, K)
    layer = LinearRowParallel(K, N, has_bias=True)
    layer.weight.copy_(w_full)
    layer.bias.copy_(b_full)
    assert torch.allclose(layer.forward(x_full), F.linear(x_full, w_full, b_full), atol=1e-5)
