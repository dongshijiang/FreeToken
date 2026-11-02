"""DeepSeek-V4.1-Flash support for FreeToken.

Ports the CSA2 backbone (sliding-window + ratio-1/2 compressed tiers + Lightning
Indexer + manifold-constrained Hyper-Connections + Engram n-gram retrieval) onto
the DSV4 paged-KV engine. The checkpoint is a multimodal wrapper (``text_config``);
FreeToken serves the text tower only. The 3 nextn (DSpark) draft layers are not
loaded.
"""

from .args import DeepseekV41Args, load_args
from .config import parse_config
from .model import DeepseekV41ForCausalLM
from .weight import iter_expert_pieces, iter_weights

__all__ = [
    "DeepseekV41Args",
    "load_args",
    "parse_config",
    "DeepseekV41ForCausalLM",
    "iter_weights",
    "iter_expert_pieces",
]
