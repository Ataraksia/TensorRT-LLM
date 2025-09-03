# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from tensorrt_llm.layers import Embedding, Linear
from tensorrt_llm.module import Module


class PartiallyFrozenEmbedding(Module):
    """Embedding with selective row freezing via a boolean mask.

    Frozen rows have their gradients zeroed during backward and can be kept
    fixed across training steps.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: Optional[int] = None,
        freeze_indices: Optional[Iterable[int]] = None,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        if padding_idx is not None:
            self.embedding.padding_idx = padding_idx
        mask = torch.zeros(num_embeddings, dtype=torch.bool)
        if freeze_indices is not None:
            for idx in freeze_indices:
                if 0 <= idx < num_embeddings:
                    mask[idx] = True
        # Store mask as simple attribute since TensorRT-LLM Module doesn't have register_buffer
        self.freeze_mask = mask

        # Register a hook to zero out gradients for frozen rows
        def _grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if self.freeze_mask.any():
                grad = grad.clone()
                grad[self.freeze_mask] = 0
            return grad

        self.embedding.weight.register_hook(_grad_mask_hook)

    def set_frozen(self, indices: Sequence[int]) -> None:
        for i in indices:
            if 0 <= i < self.freeze_mask.numel():
                self.freeze_mask[i] = True

    def set_trainable(self, indices: Sequence[int]) -> None:
        for i in indices:
            if 0 <= i < self.freeze_mask.numel():
                self.freeze_mask[i] = False

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(input_ids)


class PartiallyFrozenLinear(Module):
    """Linear layer with selective weight freezing.

    Supports per-output-row freezing. Frozen rows have their weight and bias
    gradients zeroed during backward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        freeze_output_rows: Optional[Iterable[int]] = None,
        dtype: str = "float16",
    ) -> None:
        super().__init__()
        self.linear = Linear(in_features=in_features, out_features=out_features, bias=bias)
        mask = torch.zeros(out_features, dtype=torch.bool)
        if freeze_output_rows is not None:
            for idx in freeze_output_rows:
                if 0 <= idx < out_features:
                    mask[idx] = True
        self.register_buffer("freeze_rows", mask, persistent=True)

        def _grad_mask_weight(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or not self.freeze_rows.any():
                return grad
            grad = grad.clone()
            grad[self.freeze_rows, :] = 0
            return grad

        def _grad_mask_bias(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or not self.freeze_rows.any():
                return grad
            grad = grad.clone()
            grad[self.freeze_rows] = 0
            return grad

        self.linear.weight.register_hook(_grad_mask_weight)
        if self.linear.bias is not None:
            self.linear.bias.register_hook(_grad_mask_bias)

    def set_frozen_rows(self, indices: Sequence[int]) -> None:
        for i in indices:
            if 0 <= i < self.freeze_rows.numel():
                self.freeze_rows[i] = True

    def set_trainable_rows(self, indices: Sequence[int]) -> None:
        for i in indices:
            if 0 <= i < self.freeze_rows.numel():
                self.freeze_rows[i] = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
