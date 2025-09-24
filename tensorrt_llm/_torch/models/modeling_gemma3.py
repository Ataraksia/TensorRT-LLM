import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
<<<<<<< HEAD
from tqdm import tqdm
from transformers import Gemma3TextConfig
from transformers.activations import ACT2FN

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import Attention, QkNormType
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, register_auto_model)


class Gemma3Attention(Attention):
=======
from transformers import Gemma3TextConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata, FlashInferAttentionMetadata
from ..attention_backend.interface import (AttentionMask, CustomAttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class Gemma3TextScaledWordEmbedding(Embedding):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
    ):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        self.embed_scale = torch.sqrt(torch.tensor(hidden_size)).to(self.dtype)

    @torch.inference_mode()
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma3Attention(QKNormRoPEAttention):
>>>>>>> upstream/main

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        self.is_sliding = is_sliding
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        self.attention_window_size = None
        if is_sliding:
<<<<<<< HEAD
            rope_params.theta = 10000
=======
            rope_params.theta = config.rope_local_base_freq
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
>>>>>>> upstream/main
            self.attention_window_size = config.sliding_window
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(
            config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
<<<<<<< HEAD
=======
            fuse_qk_norm_rope=False,
>>>>>>> upstream/main
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
<<<<<<< HEAD
            qk_norm_type=QkNormType.pre_rope,
            q_scaling=q_scaling,
        )
        self.q_norm = RMSNorm(hidden_size=config.head_dim,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
        self.k_norm = RMSNorm(hidden_size=config.head_dim,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)
        self.aux_stream = torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        attention_window_size = self.attention_window_size or attn_metadata.max_seq_len
=======
            q_scaling=q_scaling,
        )

    @torch.inference_mode()
    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if attention_mask_data is not None:
            assert isinstance(
                attn_metadata, FlashInferAttentionMetadata
            ), "Only FlashInfer backend supports custom attention mask currently."
            assert attention_mask == CustomAttentionMask.CUSTOM
>>>>>>> upstream/main
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
<<<<<<< HEAD
                               mrope_config=mrope_config,
                               all_reduce_params=all_reduce_params,
                               lora_params=lora_params,
                               attention_window_size=attention_window_size,
                               **kwargs)

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.q_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.k_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k


class Gemma3MLP(nn.Module):

    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype
        self.gate_proj = Linear(self.hidden_size,
                                self.intermediate_size,
                                bias=False,
                                dtype=self.dtype)
        self.up_proj = Linear(self.hidden_size,
                              self.intermediate_size,
                              bias=False,
                              dtype=self.dtype)
        self.down_proj = Linear(self.intermediate_size,
                                self.hidden_size,
                                bias=False,
                                dtype=self.dtype)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
=======
                               attention_window_size=self.attention_window_size,
                               attention_mask_data=attention_mask_data,
                               **kwargs)


# This function is written to be compatible with TRTLLM's GatedMLP class.
def pytorch_gelu_tanh(gate_x: torch.Tensor) -> torch.Tensor:
    gate, x = gate_x.chunk(2, dim=-1)
    return nn.functional.gelu(gate, approximate="tanh") * x
>>>>>>> upstream/main


class Gemma3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
<<<<<<< HEAD
        is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
=======
        is_sliding = (config.layer_types[layer_idx] == "sliding_attention")
>>>>>>> upstream/main
        self.self_attn = Gemma3Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )

<<<<<<< HEAD
        self.mlp = Gemma3MLP(config)
=======
        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            bias=False,
                            activation=pytorch_gelu_tanh,
                            dtype=config.torch_dtype,
                            config=model_config,
                            layer_idx=layer_idx)
>>>>>>> upstream/main

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
                                                 dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

<<<<<<< HEAD
    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
=======
    @torch.inference_mode()
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
>>>>>>> upstream/main
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
<<<<<<< HEAD
=======
            attention_mask=CustomAttentionMask.CUSTOM if attention_mask_data
            is not None else PredefinedAttentionMask.CAUSAL,
            attention_mask_data=attention_mask_data,
>>>>>>> upstream/main
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
<<<<<<< HEAD
        hidden_states = self.mlp(hidden_states)
=======
        hidden_states = self.mlp(hidden_states,
                                 lora_params=kwargs.get("lora_params", None))
>>>>>>> upstream/main
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Gemma3TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size
<<<<<<< HEAD
        self.padding_idx = config.pretrained_config.pad_token_id

        self.embed_tokens = Embedding(
=======

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
>>>>>>> upstream/main
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

<<<<<<< HEAD
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
=======
    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        local_attention_mask_data: Optional[torch.Tensor] = None,
        global_attention_mask_data: Optional[torch.Tensor] = None,
>>>>>>> upstream/main
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
<<<<<<< HEAD
            inputs_embeds = inputs_embeds * math.sqrt(self.hidden_size)
=======
>>>>>>> upstream/main

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
<<<<<<< HEAD
            hidden_states = decoder_layer(position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          attn_metadata=attn_metadata)
=======
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask_data=local_attention_mask_data
                if decoder_layer.self_attn.is_sliding else
                global_attention_mask_data,
                **kwargs,
            )
>>>>>>> upstream/main

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Gemma3ForCausalLM")
class Gemma3ForCausalLM(DecoderModelForCausalLM[Gemma3TextModel,
                                                Gemma3TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

<<<<<<< HEAD
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:

=======
    def _get_token_type_mask(self, image_token_mask: torch.BoolTensor):
        device = image_token_mask.device
        sequence_length = len(image_token_mask)

        # Create a list of token type ids. 0 for text tokens, 1 for all image tokens (regardless of which image they belong to).
        token_type_ids = torch.zeros(sequence_length,
                                     dtype=torch.int32,
                                     device=device)
        token_type_ids[image_token_mask] = 1

        # There could be image tokens from multiple images where those corresponding to the
        # same image are contiguous. We assign a unique id to each contiguous blob of image tokens now.

        # Pad with zero at the start to detect changes.
        padded = torch.cat((torch.tensor([0], device=device), token_type_ids))

        # Identify where blobs start (0->1 transitions)
        starts = (padded[1:] > padded[:-1]).int()

        # Cumulative sum of starts gives a unique id for each blob. Note that
        # this assigns a unique id to the zeros separating the blobs.
        blob_ids = torch.cumsum(starts, dim=0)

        # Mask out zeros (positions where token_type_ids == 0).
        token_type_ids *= blob_ids

        # Create a mask where each blob is a unique id.
        token_type_mask = token_type_ids.unsqueeze(
            0) == token_type_ids.unsqueeze(1)

        # If text token, do not change anything.
        token_type_mask = torch.where(token_type_ids == 0, False,
                                      token_type_mask)

        return token_type_mask

    def get_context_mask(
        self,
        image_token_mask: torch.BoolTensor,
        effective_sliding_window: Optional[int] = None,
    ):
        """
        Returns an attention mask such that text tokens attend to each other in causal fashion while image
        tokens attend in causal fashion as well as to all other image tokens in a bidirectional manner.
        Args:
            image_token_mask: A boolean tensor of shape (sequence_length,) where True indicates an image token.
            effective_sliding_window: The effective sliding window size for the attention mask. Default is None, which means no sliding window.
            For Gemma3, this is the sliding window size from config (e.g. 512 for 1B model).
        Returns:
            A boolean attention mask of shape (sequence_length, sequence_length).
        """
        device = image_token_mask.device
        sequence_length = len(image_token_mask)
        if effective_sliding_window is None or effective_sliding_window >= sequence_length:
            causal_mask = torch.arange(
                sequence_length, device=device).unsqueeze(0) <= torch.arange(
                    sequence_length, device=device).unsqueeze(1)
        else:
            attention_mask_1 = (torch.arange(sequence_length,
                                             device=device).unsqueeze(0)
                                <= torch.arange(sequence_length,
                                                device=device).unsqueeze(1))
            attention_mask_2 = (
                torch.arange(sequence_length, device=device).unsqueeze(0)
                > torch.arange(sequence_length, device=device).unsqueeze(1) -
                effective_sliding_window)
            causal_mask = attention_mask_1 & attention_mask_2

        # Apply a bidirectional mask for image tokens.
        token_type_mask = self._get_token_type_mask(image_token_mask)
        causal_mask = causal_mask.masked_fill(token_type_mask, True)
        return causal_mask

    # ASSUMPTIONS:
    # 1) Chunked prefill is disabled to avoid chunking image tokens as they need bidirectional attention.
    # 2) KV cache reuse is disabled to avoid partially matched image tokens (entire image must be reused to get things correct).
    def get_flashinfer_attention_mask(
            self,
            image_token_mask: torch.BoolTensor,
            attn_metadata: AttentionMetadata,
            effective_sliding_window: Optional[int] = None) -> torch.Tensor:
        """
        This is specifically needed for context phase requests. Currently, we don't create custom mask for generation requests because FlashInfer backend
        doesn't use it anyway and there's nothing special we need to do for generation requests.
        - This function will only be called for a batch when there's at least one context request in the batch with image tokens.
        - In context phase, each sample's input_ids may have a mix of image tokens and text tokens where tokens corresponding to an image
        appear as a contiguous blob. Example: torch.IntTensor([2, 3, 4, 5, img_idx, img_idx, img_idx, ..., img_idx, 100])
        - While the text tokens attend to other tokens in a causal fashion, image tokens attend to others in a causal fashion and well as
        attend to other image tokens in a bidirectional manner. Hence, the need for custom masking.
        Args:
            image_token_mask: A boolean tensor of shape (len(input_ids),) where True indicates an image token. This corresponds to concatenated
            list of tokens for all samples in the batch.
            attn_metadata: The attention metadata for the batch.
            effective_sliding_window: The effective sliding window size for the attention mask. Default is None, which means no sliding window.
            For Gemma3, this is the sliding window size from config (e.g. 512 for 1B model).
        Returns:
            A flattened boolean mask of shape (sum(q_len[i] * k_len[i] for i in range(batch_size)).
        """

        assert isinstance(
            attn_metadata, FlashInferAttentionMetadata
        ), "Only FlashInfer backend supports custom mask currently."
        num_contexts = attn_metadata.num_contexts
        assert num_contexts > 0, "There should be at least one context request in the batch for custom mask."

        qo_indptr = attn_metadata.qo_indptr[:num_contexts + 1]
        cached_token_lens = attn_metadata.cached_token_lens[:num_contexts]
        assert (cached_token_lens == 0).all(
        ), "cached_token_lens should be 0 for context requests since chunked prefill and kv cache reuse must be disabled."

        # Create masks for context requests.
        context_mask_list = []
        for i in range(num_contexts):
            mask_i = self.get_context_mask(
                image_token_mask=image_token_mask[qo_indptr[i]:qo_indptr[i +
                                                                         1]],
                effective_sliding_window=effective_sliding_window,
            )
            context_mask_list.append(mask_i.flatten())
        return torch.cat(context_mask_list, dim=0).contiguous()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        image_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        local_attention_mask_data = None
        global_attention_mask_data = None
        if image_token_mask is not None:
            global_attention_mask_data = self.get_flashinfer_attention_mask(
                image_token_mask=image_token_mask,
                attn_metadata=attn_metadata,
                effective_sliding_window=None,
            )
            local_attention_mask_data = self.get_flashinfer_attention_mask(
                image_token_mask=image_token_mask,
                attn_metadata=attn_metadata,
                effective_sliding_window=self.config.sliding_window,
            )

>>>>>>> upstream/main
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
<<<<<<< HEAD
=======
            local_attention_mask_data=local_attention_mask_data,
            global_attention_mask_data=global_attention_mask_data,
            **kwargs,
>>>>>>> upstream/main
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

<<<<<<< HEAD
    # This is a modified version of the load_weights function in modeling_utils.py with the
    # minor change for Gemma3 RMSNorm.
    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue

                # Skip loading weights for embedding and lm_head if LoRA is enabled.
                if hasattr(
                        self.model_config, 'lora_config'
                ) and self.model_config.lora_config is not None and len(
                        self.model_config.lora_config.lora_dir) == 1 and (
                            name == "model.embed_tokens" or name == "lm_head"):
                    continue

                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }

                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                # Gemma3 RMSNorm uses +1 just like LayerNorm-1P.
                                if 'norm' in names[-1]:
                                    p.data.copy_(module_weights[n][:] + 1)
                                else:
                                    p.data.copy_(module_weights[n][:])
=======
    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        super().load_weights(weights, weight_mapper)
>>>>>>> upstream/main
