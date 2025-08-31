# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Custom Modules for Higgs Audio TensorRT-LLM Implementation

This module implements custom components required for the Higgs Audio model:
- PartiallyFrozenEmbedding: Allows freezing part of an embedding layer
- PartiallyFrozenLinear: Allows freezing part of a linear layer

These modules are essential for fine-tuning scenarios where we want to freeze
the original vocabulary/parameters while allowing training of new audio tokens.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm.functional import Tensor, gather, where, zeros_like
from tensorrt_llm.layers import Embedding, ColumnLinear
from tensorrt_llm.module import Module
from tensorrt_llm.parameter import Parameter


class PartiallyFrozenEmbedding(Module):
    """
    Split an existing embedding into frozen and trainable parts.
    
    This module splits the embedding into:
    - A frozen embedding for indices [0..freeze_until_idx]
    - A trainable embedding for indices [freeze_until_idx+1..vocab_size-1]
    
    This is useful for scenarios where we want to freeze the original vocabulary
    embeddings while allowing training of new token embeddings (e.g., audio tokens).
    """

    def __init__(
        self,
        original_vocab_size: int,
        embedding_dim: int,
        freeze_until_idx: int,
        dtype: str = 'float16',
        tp_size: int = 1,
        tp_group: Optional[list] = None,
        sharding_dim: int = 1,  # Embedding dimension sharding
        tp_rank: int = 0,
    ):
        """
        Initialize the partially frozen embedding.
        
        Args:
            original_vocab_size: Total vocabulary size
            embedding_dim: Embedding dimension
            freeze_until_idx: Index up to which embeddings are frozen (exclusive)
            dtype: Data type for parameters
            tp_size: Tensor parallelism size
            tp_group: Tensor parallelism group
            sharding_dim: Dimension for tensor parallelism sharding
            tp_rank: Tensor parallelism rank
        """
        super().__init__()
        
        self.freeze_until_idx = freeze_until_idx
        self.original_vocab_size = original_vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        
        # Frozen embedding for original vocabulary
        self.embedding_frozen = Embedding(
            num_embeddings=freeze_until_idx,
            embedding_dim=embedding_dim,
            dtype=dtype,
            tp_size=tp_size,
            tp_group=tp_group,
            sharding_dim=sharding_dim,
            tp_rank=tp_rank,
        )
        
        # Trainable embedding for new tokens  
        self.embedding_trainable = Embedding(
            num_embeddings=original_vocab_size - freeze_until_idx,
            embedding_dim=embedding_dim,
            dtype=dtype,
            tp_size=tp_size,
            tp_group=tp_group,
            sharding_dim=sharding_dim,
            tp_rank=tp_rank,
        )
        
        # Freeze the frozen embedding weights
        self.embedding_frozen.weight.requires_grad = False

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Memory-efficient forward pass for the split embedding.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        from tensorrt_llm.functional import zeros, gather_last_token_logits
        
        batch_size, seq_len = input_ids.shape
        
        # Initialize output tensor with zeros
        embeddings = zeros([batch_size, seq_len, self.embedding_dim], dtype=self.dtype)
        
        # Create masks for frozen vs trainable indices
        mask_frozen = input_ids < self.freeze_until_idx
        mask_trainable = input_ids >= self.freeze_until_idx
        
        # Only process frozen tokens if any exist
        if mask_frozen.any():
            # Clamp frozen indices to valid range [0, freeze_until_idx)
            frozen_ids = input_ids.clamp(max=self.freeze_until_idx - 1)
            frozen_embeddings = self.embedding_frozen(frozen_ids)
            
            # Apply mask to only keep frozen embeddings where appropriate
            embeddings = embeddings + frozen_embeddings * mask_frozen.unsqueeze(-1).cast(self.dtype)
        
        # Only process trainable tokens if any exist  
        if mask_trainable.any():
            # Adjust trainable IDs to local index space and clamp to valid range
            trainable_ids = (input_ids - self.freeze_until_idx).clamp(min=0, max=self.original_vocab_size - self.freeze_until_idx - 1)
            trainable_embeddings = self.embedding_trainable(trainable_ids)
            
            # Apply mask to only keep trainable embeddings where appropriate
            embeddings = embeddings + trainable_embeddings * mask_trainable.unsqueeze(-1).cast(self.dtype)
        
        return embeddings

    def load_frozen_weights(self, frozen_weights: Tensor) -> None:
        """
        Load weights for the frozen embedding portion.
        
        Args:
            frozen_weights: Weights for frozen embeddings [freeze_until_idx, embedding_dim]
        """
        if frozen_weights.shape[0] != self.freeze_until_idx:
            raise ValueError(
                f"Expected frozen weights shape [{self.freeze_until_idx}, {self.embedding_dim}], "
                f"but got {frozen_weights.shape}"
            )
        
        # Copy weights to frozen embedding
        self.embedding_frozen.weight.value = frozen_weights
        self.embedding_frozen.weight.requires_grad = False

    def load_trainable_weights(self, trainable_weights: Tensor) -> None:
        """
        Load weights for the trainable embedding portion.
        
        Args:
            trainable_weights: Weights for trainable embeddings 
                             [vocab_size - freeze_until_idx, embedding_dim]
        """
        expected_size = self.original_vocab_size - self.freeze_until_idx
        if trainable_weights.shape[0] != expected_size:
            raise ValueError(
                f"Expected trainable weights shape [{expected_size}, {self.embedding_dim}], "
                f"but got {trainable_weights.shape}"
            )
        
        # Copy weights to trainable embedding
        self.embedding_trainable.weight.value = trainable_weights


class PartiallyFrozenLinear(Module):
    """
    A wrapper around linear layer to partially freeze part of the weight matrix.
    
    This module splits a linear layer into frozen and trainable parts:
    - Frozen part corresponds to original vocabulary/features
    - Trainable part corresponds to new tokens/features (e.g., audio tokens)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        freeze_until_idx: int,
        bias: bool = True,
        dtype: str = 'float16',
        tp_size: int = 1,
        tp_group: Optional[list] = None,
        gather_output: bool = True,
        tp_rank: int = 0,
    ):
        """
        Initialize the partially frozen linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            freeze_until_idx: Index up to which output features are frozen (exclusive)
            bias: Whether to use bias
            dtype: Data type for parameters
            tp_size: Tensor parallelism size
            tp_group: Tensor parallelism group
            gather_output: Whether to gather output in tensor parallelism
            tp_rank: Tensor parallelism rank
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.freeze_until_idx = freeze_until_idx
        self.use_bias = bias
        self.dtype = dtype
        
        # Frozen linear layer for original features
        self.linear_frozen = ColumnLinear(
            in_features=in_features,
            out_features=freeze_until_idx,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=gather_output,
        )
        
        # Trainable linear layer for new features
        self.linear_trainable = ColumnLinear(
            in_features=in_features,
            out_features=out_features - freeze_until_idx,
            bias=bias,
            dtype=dtype,
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=gather_output,
        )
        
        # Freeze the frozen linear layer weights
        self.linear_frozen.weight.requires_grad = False
        if bias:
            self.linear_frozen.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Memory-efficient forward pass for the split linear layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            
        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        # Compute outputs from both parts
        frozen_output = self.linear_frozen(x)
        trainable_output = self.linear_trainable(x)
        
        # Concatenate along the output feature dimension
        # Use memory-efficient concatenation
        output = frozen_output.concat(trainable_output, dim=-1)
        
        return output

    def load_frozen_weights(self, frozen_weight: Tensor, frozen_bias: Optional[Tensor] = None) -> None:
        """
        Load weights for the frozen linear layer portion.
        
        Args:
            frozen_weight: Weights for frozen linear layer [freeze_until_idx, in_features]
            frozen_bias: Optional bias for frozen linear layer [freeze_until_idx]
        """
        if frozen_weight.shape != (self.freeze_until_idx, self.in_features):
            raise ValueError(
                f"Expected frozen weight shape [{self.freeze_until_idx}, {self.in_features}], "
                f"but got {frozen_weight.shape}"
            )
        
        # Copy weights to frozen linear layer
        self.linear_frozen.weight.value = frozen_weight
        self.linear_frozen.weight.requires_grad = False
        
        if self.use_bias and frozen_bias is not None:
            if frozen_bias.shape != (self.freeze_until_idx,):
                raise ValueError(
                    f"Expected frozen bias shape [{self.freeze_until_idx}], "
                    f"but got {frozen_bias.shape}"
                )
            self.linear_frozen.bias.value = frozen_bias
            self.linear_frozen.bias.requires_grad = False

    def load_trainable_weights(self, trainable_weight: Tensor, trainable_bias: Optional[Tensor] = None) -> None:
        """
        Load weights for the trainable linear layer portion.
        
        Args:
            trainable_weight: Weights for trainable linear layer 
                            [out_features - freeze_until_idx, in_features]
            trainable_bias: Optional bias for trainable linear layer 
                          [out_features - freeze_until_idx]
        """
        expected_out_features = self.out_features - self.freeze_until_idx
        if trainable_weight.shape != (expected_out_features, self.in_features):
            raise ValueError(
                f"Expected trainable weight shape [{expected_out_features}, {self.in_features}], "
                f"but got {trainable_weight.shape}"
            )
        
        # Copy weights to trainable linear layer
        self.linear_trainable.weight.value = trainable_weight
        
        if self.use_bias and trainable_bias is not None:
            if trainable_bias.shape != (expected_out_features,):
                raise ValueError(
                    f"Expected trainable bias shape [{expected_out_features}], "
                    f"but got {trainable_bias.shape}"
                )
            self.linear_trainable.bias.value = trainable_bias
