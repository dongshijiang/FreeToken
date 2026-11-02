"""DeepSeek-V4.1 image preprocessing: the released aspect-preserving resize, CPU PIL only.

The checkpoint ships no HF image processor; this ports the reference
``_plan_image_grid`` / ``_preprocess_image`` verbatim (fixed resize, fixed
normalization) so radix keys and the embedding cache hash the same pixels every run.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash


@dataclass(frozen=True)
class _Budget:
    """The vision_config sizing knobs the plan reads (released defaults when the section omits them)."""

    patch_size: int = 14
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None


@dataclass(frozen=True)
class _ImageGrid:
    height: int
    width: int
    vit_height: int
    vit_width: int
    llm_height: int
    llm_width: int


def _plan_image_grid(width: int, height: int, budget: _Budget) -> _ImageGrid:
    """The released aspect-preserving resize and exact row-major token budget."""
    if width <= 0 or height <= 0 or budget.max_image_tokens < 4:
        raise ValueError("Images require positive dimensions and max_image_tokens >= 4")
    logical_width: int | float = width
    logical_height: int | float = height
    if budget.max_wh_ratio is not None and width > height * budget.max_wh_ratio:
        logical_width = height * budget.max_wh_ratio
    if logical_width * logical_height < budget.min_pixels:
        ratio = (budget.min_pixels / (logical_width * logical_height)) ** 0.5
        logical_width, logical_height = int(logical_width * ratio), int(logical_height * ratio)
    patch_size, downsample = budget.patch_size, budget.downsample_ratio
    best_width = math.ceil(logical_width / patch_size) * patch_size
    best_height = math.ceil(logical_height / patch_size) * patch_size
    llm_height = math.ceil(best_height // patch_size / downsample)
    llm_width = math.ceil(best_width // patch_size / downsample)
    if llm_height * (llm_width + 1) + 2 > budget.max_image_tokens:
        aspect = logical_height / logical_width
        max_width = math.sqrt((budget.max_image_tokens - 2) / aspect + 0.25) - 0.5
        max_height = max_width * aspect
        cell = patch_size * downsample
        if max_width < 1:
            best_height, best_width = (budget.max_image_tokens - 2) // 2 * cell, cell
        elif max_height < 1:
            best_height, best_width = cell, (budget.max_image_tokens - 3) * cell
        else:
            beta = min(math.floor(max_width) * cell / logical_width, math.floor(max_height) * cell / logical_height)
            best_height = math.floor(logical_height * beta / patch_size) * patch_size
            best_width = math.floor(logical_width * beta / patch_size) * patch_size
        llm_height = math.ceil(best_height // patch_size / downsample)
        llm_width = math.ceil(best_width // patch_size / downsample)
    if min(best_height, best_width) <= 0 or llm_height * (llm_width + 1) + 2 > budget.max_image_tokens:
        raise ValueError("The image cannot fit in the configured visual-token budget")
    return _ImageGrid(
        best_height, best_width, best_height // patch_size, best_width // patch_size, llm_height, llm_width
    )


def _preprocess_image(image: Image.Image, budget: _Budget) -> tuple[torch.Tensor, _ImageGrid]:
    """Resize, normalize and flatten one RGB image into patches.

    Returns bf16 patches ``[vit_h * vit_w, 3, patch, patch]`` in row-major order
    and the grid plan.
    """
    grid = _plan_image_grid(image.width, image.height, budget)
    if budget.max_wh_ratio is not None and image.width >= budget.max_wh_ratio * image.height:
        image = image.resize((grid.width, grid.height))
    else:
        image = ImageOps.pad(image, (grid.width, grid.height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    ps = budget.patch_size
    patches = (
        pixels.reshape(3, grid.vit_height, ps, grid.vit_width, ps)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid.vit_height * grid.vit_width, 3, ps, ps)
    )
    return patches, grid


class DeepseekV41MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = hf_config.image_token_id
        self.placeholder = [self.image_token_id]
        budget = _Budget(
            patch_size=int(getattr(vc, "patch_size", 14)),
            downsample_ratio=int(getattr(vc, "downsample_ratio", 3)),
            max_image_tokens=int(getattr(vc, "max_image_tokens", 1024)),
            min_pixels=int(getattr(vc, "min_pixels", 295936)),
            max_wh_ratio=getattr(vc, "max_wh_ratio", None),
        )
        if mm.image_max_tokens is not None:
            if mm.image_max_tokens < 4:
                raise ValueError(f"--image-max-tokens {mm.image_max_tokens} is below the smallest image span (4 tokens)")
            budget = replace(budget, max_image_tokens=mm.image_max_tokens)
        if mm.image_min_tokens is not None:
            # the per-image floor converts to the resize floor in pixels
            budget = replace(
                budget, min_pixels=mm.image_min_tokens * (budget.patch_size * budget.downsample_ratio) ** 2
            )
        self._budget = budget

    def process(self, images: list[Any]) -> list[MMItem]:
        items: list[MMItem] = []
        for image in images:
            patches, grid = _preprocess_image(image, self._budget)
            # the grid rides the hash so transposed images with equal pixels stay distinct
            item_hash = content_hash(patches, struct.pack("<ii", grid.vit_height, grid.vit_width))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=patches,
                    model_specific_data={
                        "vit_grid": (grid.vit_height, grid.vit_width),
                        "llm_grid": (grid.llm_height, grid.llm_width),
                    },
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        llm_h, llm_w = item.llm_grid
        # START + llm_h rows of (llm_w soft tokens + NEWLINE) + END; every span token is the image id
        full = [self.image_token_id] * (2 + llm_h * (llm_w + 1))
        return PromptReplacement(full)

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        # the smallest encodable image: one ViT patch, a 1x1 llm grid
        ps = self._budget.patch_size
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 4]],
            feature=torch.zeros(1, 3, ps, ps, dtype=dtype, device=device),
            model_specific_data={"vit_grid": (1, 1), "llm_grid": (1, 1)},
        )]


__all__ = ["DeepseekV41MMProcessor"]
