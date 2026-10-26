"""DistributedCommunicator.broadcast routing.

Stochastic sampling under TP draws from per-rank RNG state, so the engine publishes
rank 0's token through ``broadcast``; a wrong routing would let the ranks append
different tokens and silently diverge their KV caches. The NCCL kernel itself needs
a multi-rank GPU job, so these tests pin the Python contract with recording fakes.
"""

import torch
from freetoken.distributed.impl import (
    DistributedCommunicator,
    DistributedImpl,
    PyNCCLDistributedImpl,
)


class _Recording(DistributedImpl):
    def __init__(self):
        self.seen = []

    def all_reduce(self, x):
        return x

    def all_gather(self, x):
        return x

    def broadcast(self, x):
        self.seen.append(x.clone())
        return x


def test_broadcast_routes_to_the_last_plugin(monkeypatch):
    first, second = _Recording(), _Recording()
    monkeypatch.setattr(DistributedCommunicator, "plugins", [first, second])
    x = torch.arange(4, dtype=torch.int32)
    out = DistributedCommunicator().broadcast(x)
    assert out is x
    assert first.seen == []
    assert len(second.seen) == 1 and torch.equal(second.seen[0], x)


def test_pynccl_broadcast_delegates_to_the_device_comm():
    seen = []

    class _FakeComm:
        def broadcast(self, t):
            seen.append(t)

        def all_reduce(self, t, op):
            pass

        def all_gather(self, out, t):
            pass

    impl = PyNCCLDistributedImpl(comm=_FakeComm())
    x = torch.arange(4, dtype=torch.int32)
    assert impl.broadcast(x) is x
    assert seen == [x]
