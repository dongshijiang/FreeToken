from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_items: list | None = None
    mrope_positions_full: torch.Tensor | None = None
    mrope_delta: int = 0
    # Admission-deferral bookkeeping: how many passes this request was soft-skipped
    # because its DSV4 host pages were still warming. Incremented only on the
    # MIN-voted not-ready verdict, so it is identical on every TP rank.
    defer_count: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
