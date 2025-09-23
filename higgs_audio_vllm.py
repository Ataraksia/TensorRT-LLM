# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Inference-only Higgs Audio model compatible with HuggingFace weights."""

import copy
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    BatchFeature,
    ProcessorMixin,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.v1.multimodal.metadata import MultimodalMetadata
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "audio_decoder_proj.audio_lm_head": "audio_lm_head",
    "audio_decoder_proj.text_lm_head": "text_lm_head",
}

AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoFeatureExtractor.register(HiggsAudioConfig, AudioTokenizer)
# if transformers.__version__.startswith("4.46"):
transformers._modules.add("AudioTokenizer")
transformers.AudioTokenizer = AudioTokenizer


# # === Audio Inputs === #
class HiggsAudioInputs(TypedDict):
    # (num_audios, num_mel_bins, 3000)`
    audio_features: torch.Tensor

    # (num_audios, 3000)
    audio_feature_attention_mask: torch.Tensor

    # (num_audios, num_codebooks)
    audio_out_ids: torch.Tensor


def _validate_and_reshape_mm_tensor(
    mm_input: object,
    name: str,
    pad_with: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(mm_input, (torch.Tensor, list)):
        raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
    if isinstance(mm_input, torch.Tensor):
        return torch.concat(list(mm_input))
    else:
        if pad_with is not None:
            max_size = max(
                [tensor.size(-1) for tensor in mm_input]
            )  # Find max size along the last dimension
            # Step 2: Pad each tensor to the max size along the last
            # dimension
            padded_tensors = []
            for tensor in mm_input:
                pad_size = max_size - tensor.size(-1)  # Calculate how much padding is needed
                if pad_size > 0:
                    # Pad tensor along the last dimension (right side)
                    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            return torch.concat(padded_tensors)
        else:
            return torch.concat(mm_input)


def _build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=8. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  *,  *,  *, P, P, P]
    - [ B,  *,  *,  *,  *,  *,  *,  *, *, P, P]
    - [ B,  B,  *,  *,  *,  *,  *,  *, *, *, P]
    - [ B,  B,  B,  *,  *,  *,  *,  *, *, *, *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones(
        (bsz, num_codebooks, new_seq_len),
        dtype=torch.long,
        device=input_ids.device,
    )
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the
    LLM component to audio / text logits."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

    def forward(
        self,
        hidden_states,
        audio_out_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape
                           `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices
            position_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Position ids for the input tokens

        Returns:
            logits (`torch.Tensor` of shape
                   `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape
                `(num_audio_out_tokens, num_codebooks * codebook_size)`):
                Logits for audio tokens. We ensure
                `num_text_tokens + num_audio_tokens == batch_size * seq_len`.
                If we the model only outputs text logits,
                `audio_logits` will be `None`.

        """
        # TODO(sxjscience) Need to check if DeepSpeed Zero3 supports zero-shape input.

        return hidden_states


def get_processor(
    tokenzier,
    *args,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoFeatureExtractor

    HIGGS_AUDIO_TOKENIZER = os.getenv("HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo")

    audio_stream_bos_id = kwargs.pop("audio_stream_bos_id", None)
    audio_stream_eos_id = kwargs.pop("audio_stream_eos_id", None)

    if HIGGS_AUDIO_TOKENIZER == "openai/whisper-large-v3-turbo":
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            HIGGS_AUDIO_TOKENIZER,  # TODO: Write into config file
            *args,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            **kwargs,
        )
    else:
        HIGGS_AUDIO_TOKENIZER_PATH = os.environ.get(
            "HIGGS_AUDIO_TOKENIZER_PATH",
            None,
        )
        feature_extractor = AudioTokenizer(
            model=HIGGS_AUDIO_TOKENIZER,
            device="cuda",
        )
    processor = HFHiggsAudioProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenzier,
        audio_stream_bos_id=audio_stream_bos_id,
        audio_stream_eos_id=audio_stream_eos_id,
    )
    logger.info("Loaded HFHiggsAudioProcessor")

    return processor


cached_get_processor = lru_cache(get_processor)


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers
    and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


class HFHiggsAudioProcessor(ProcessorMixin):
    """
    HF Processor class for Higgs audio model. Mostly borrow from
    processing_qwen2_audio.py.
    """

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_stream_bos_id=None,
        audio_stream_eos_id=None,
        is_audio_out_model=False,
    ):
        self.is_audio_out_model = is_audio_out_model
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = (
            tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        )
        self.audio_bos_token = (
            tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        )
        self.audio_eos_token = (
            tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        )

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        # HACK: Workaround the class check in the base class
        if feature_extractor is not None:
            self.feature_extractor_class = feature_extractor.__class__.__name__
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[
            TextInput,
            PreTokenizedInput,
            List[TextInput],
            List[PreTokenizedInput],
        ] = None,
        audio: Union[np.ndarray, List[np.ndarray]] = None,
        audios=None,  # kept for BC
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and
        audio(s). Borrowed the code from Qwen2 Audio.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence
                can be a string or a list of strings (pretokenized string). If
                the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity
                with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be
                a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*,
                    defaults to `False`):
                Select a strategy to pad the returned sequences (according to
                the model's padding side and padding index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the
                  batch (or no padding if only a single sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the
                  argument `max_length` or to the maximum acceptable input
                  length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can
                  output a batch with sequences of different lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be
                digitalized expressed in hertz (Hz).
        """

        # Handle BC when user passes deprecared keyword argument
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "You may have used the keyword argument for the `audio` inputs. "
                "It is strongly recommended to pass inputs with keyword arguments "
                "with keys `audio` and `text`. From transformers v4.55 `audio` "
                "will be the only acceptable keyword argument.",
                FutureWarning,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        if audio is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token"
                    f"{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio"
                    f"{'s' if num_audios > 1 else ''}"
                )
            # Some kwargs should not be changed so we can expand text with audio tokens below
            use_whisper = False
            if hasattr(self.feature_extractor, "encode"):
                if isinstance(audio, np.ndarray):
                    audio = [audio]
                audio = [a.astype(np.float32) for a in audio]
                audio_ids = [
                    self.feature_extractor.encode(
                        a, self.feature_extractor.sampling_rate
                    ).unsqueeze(0)
                    for a in audio
                ]

                # -2 is the number of codebooks
                num_codebook_dim = -2
                use_delay_pattern = audio_ids[0].shape[num_codebook_dim] > 1
                if use_delay_pattern:
                    for i, audio_id in enumerate(audio_ids):
                        audio_id = torch.cat(
                            [
                                torch.full(
                                    (1, audio_id.shape[num_codebook_dim], 1),
                                    self.audio_stream_bos_id,
                                    dtype=torch.long,
                                    device=audio_id.device,
                                ),
                                audio_id,
                                torch.full(
                                    (1, audio_id.shape[num_codebook_dim], 1),
                                    self.audio_stream_eos_id,
                                    dtype=torch.long,
                                    device=audio_id.device,
                                ),
                            ],
                            dim=-1,
                        )
                        audio_ids[i] = _build_delay_pattern_mask(
                            audio_id,
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )

                audio_lengths = [a.shape[-1] for a in audio_ids]
                audio_in_ids_length = torch.tensor(audio_lengths)
                audio_in_ids = _validate_and_reshape_mm_tensor(
                    audio_ids, "audio_in_ids", pad_with=0
                )
                audio_feature_attention_mask = torch.arange(audio_in_ids.shape[-1]).expand(
                    audio_in_ids.shape[0], audio_in_ids.shape[-1]
                ).to(audio_in_ids_length.device) < audio_in_ids_length.unsqueeze(-1)
                audio_inputs = {
                    "input_features": audio_in_ids,
                    "audio_feature_attention_mask": audio_feature_attention_mask,
                }

        inputs = self.tokenizer(text, padding=padding, **kwargs)

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    @property
    def default_chat_template(self):
        # fmt: off
        if self.is_audio_out_model:
            return (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + "
                    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}"
                        "{% set content = bos_token + content %}"
                    "{% endif %}"
                    "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
                        "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
                    "{% endif %}"
                    "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>' }}"
                "{% endif %}"
            )

        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + "
                "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        # fmt: on


HiggsAudioFeatureExtractor = Union[AudioTokenizer, WhisperFeatureExtractor]


class HiggsAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(HiggsAudioConfig)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> HFHiggsAudioProcessor:
        hf_config = self.get_hf_config()
        return cached_get_processor(
            self.ctx.tokenizer,
            audio_stream_bos_id=hf_config.audio_stream_bos_id,
            audio_stream_eos_id=hf_config.audio_stream_eos_id,
        )

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> HiggsAudioFeatureExtractor:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        self.audio_tokenizer_type = os.getenv(
            "HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"
        )
        if self.audio_tokenizer_type == "openai/whisper-large-v3-turbo":
            max_source_position = hf_config.audio_encoder_config.max_source_positions
            max_output_lengths = (max_source_position - 2) // 2 + 1
        else:
            max_output_lengths = (
                30 * self.get_feature_extractor().tps
                + self.get_feature_extractor().num_codebooks
                - 1
                + 2
            )  # bos and eos
        return {"audio": max_output_lengths}


class HiggsAudioMultiModalProcessor(BaseMultiModalProcessor[HiggsAudioProcessingInfo]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_tokenizer_type = os.getenv(
            "HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"
        )
        self.use_whisper_tokenizer = self.audio_tokenizer_type == "openai/whisper-large-v3-turbo"

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.info.get_feature_extractor().sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audios", []):
            # Set add_special_tokens=False to avoid
            # adding an extra begin of text token
            prompt_ids = self.info.get_tokenizer().encode(prompt, add_special_tokens=False)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            batch_data = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
            return batch_data

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        batch_data = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        batch_data["audio_features"] = batch_data.pop("input_features")
        return batch_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_token_id = vocab[audio_token]

        audio_feature_attention_mask = out_mm_kwargs.get("audio_feature_attention_mask")
        if audio_feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(audio_feature_attention_mask, torch.Tensor)

            if self.use_whisper_tokenizer:
                _, audio_output_lens = _get_feat_extract_output_lengths(
                    audio_feature_attention_mask.sum(-1)
                )
            else:
                audio_output_lens = audio_feature_attention_mask.sum(-1)
            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_higgs_audio(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(
                    f"The audio (len={audio_len}) is too short to be represented inside the model"
                )

            audio_tokens = [audio_token_id] * num_features

            # New API: PromptUpdateDetails only accepts 'full' and optional 'is_embed'.
            # All tokens are embeddings here, so use from_seq.
            return PromptUpdateDetails.from_seq(audio_tokens)

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_higgs_audio,
            )
        ]


class HiggsAudioDummyInputsBuilder(BaseDummyInputsBuilder[HiggsAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Use the processor's placeholder for audio inputs.
        # The processor recognizes '<|AUDIO|>' tokens in prompt text.
        return "<|AUDIO|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, object]:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = feature_extractor.chunk_length * sampling_rate
        else:
            # Default to 30 seconds audio
            audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = feature_extractor.chunk_length * sampling_rate
        else:
            # Default to 30 seconds audio
            audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        mm_data = {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}

        return ProcessorInputs(
            prompt="<|AUDIO|>" * num_audios,
            mm_data=mm_data,
        )


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and
    text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be
    encoded with separate feedforward layers. In addition, the audio tokens can
    be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (audio self-attention layer)
                        v
    t    t     t    h'_a h'_a  h'_a  t  t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for
        audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
    This should have the same effect as the mixture-of-expert layer and
       we may expect better performance due to the scaling law.
    3) We can replace the original FFN in LLMs with the dual-path FFN without
       changing the model architecture.


    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        rope_theta = getattr(text_config, "rope_theta", 10000)
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            text_config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                text_config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(text_config, "max_position_embeddings", 8192)
        attention_bias = getattr(text_config, "attention_bias", False) or getattr(
            text_config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=text_config,
            hidden_size=self.hidden_size,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=getattr(
                text_config,
                "num_key_value_heads",
                text_config.num_attention_heads,
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            quant_config=quant_config,
            bias=getattr(text_config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.fast_forward = self.layer_idx not in config.audio_dual_ffn_layers
        self.use_audio_attention = config.use_audio_out_self_attention

        if self.fast_forward or self.use_audio_attention:
            raise NotImplementedError(
                f"The fast-forward and audio-attention mode are not supported in "
                f"HiggsAudioDualFFNDecoderLayer, but got fast_forward={self.fast_forward}"
                f"and use_audio_attention={self.use_audio_attention}."
            )

        if not self.fast_forward:
            if self.use_audio_attention:
                self.audio_attn = LlamaAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=getattr(
                        config,
                        "num_key_value_heads",
                        config.num_attention_heads,
                    ),
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=quant_config,
                    bias=attention_bias,
                    cache_config=cache_config,
                    prefix=f"{prefix}.self_attn",
                )
                self.audio_post_audio_attn_layer_norm = RMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=quant_config,
                bias=getattr(text_config, "mlp_bias", False),
                prefix=f"{prefix}.audio_mlp",
            )
            self.audio_input_layernorm = RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
            self.audio_post_attention_layernorm = RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )

        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        assert residual is None, (
            "The residual output is not supported in HiggsAudioDualFFNDecoderLayer."
        )

        residual = hidden_states

        # if self.fast_forward and has_audio_out:
        #     original_hidden_states = hidden_states.clone()

        audio_out_mask = get_forward_context().multimodal_metadata.token_mm_map.unsqueeze(-1)
        if not self.fast_forward:
            hidden_states = torch.where(
                audio_out_mask,
                self.audio_input_layernorm(hidden_states),
                self.input_layernorm(hidden_states),
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # # Audio Attention
        # if self.use_audio_attention and has_audio_out:
        #     assert (
        #         kv_cache.shape[0] == 4
        #     ), "The KV cache should have shape (4, batch_size, seq_len, hidden_size)"
        #     audio_hidden_states = self.audio_attn(
        #         positions=positions,
        #         hidden_states=hidden_states,
        #         kv_cache=kv_cache[2:4],
        #         attn_metadata=attn_metadata,
        #     )
        #     audio_hidden_states = residual + audio_hidden_states
        #     residual = torch.where(audio_out_mask.unsqueeze(-1),
        #                            audio_hidden_states, residual)
        #     audio_hidden_states = self.audio_post_audio_attn_layer_norm(
        #         audio_hidden_states)
        #     hidden_states = torch.where(audio_out_mask.unsqueeze(-1),
        #                                 audio_hidden_states, hidden_states)

        # Text Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if not self.fast_forward:
            text_hidden_states = torch.masked_fill(hidden_states, audio_out_mask, 0)
            text_hidden_states = self.post_attention_layernorm(text_hidden_states)
            audio_hidden_states = torch.masked_fill(hidden_states, ~audio_out_mask, 0)
            audio_hidden_states = self.audio_post_attention_layernorm(audio_hidden_states)
            text_hidden_states = self.mlp(text_hidden_states)
            residual += text_hidden_states
            audio_hidden_states = self.audio_mlp(audio_hidden_states)
            residual += audio_hidden_states
            hidden_states = residual
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # if self.fast_forward:
        #     hidden_states = torch.where(audio_out_mask.unsqueeze(-1),
        #                                 original_hidden_states, hidden_states)

        # Add a None as the residual output for the compatibility
        outputs = (hidden_states, None)

        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    HiggsAudioMultiModalProcessor,
    info=HiggsAudioProcessingInfo,
    dummy_inputs=HiggsAudioDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,  # sequence dimension
        "inputs_embeds": 0,  # batch dimension
    }
)
class HiggsAudioForConditionalGeneration(nn.Module, SupportsMultiModal):
    # Explicitly declare SupportsMultiModal flag
    supports_multimodal: ClassVar[Literal[True]] = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        # _placeholder_str
        if modality.startswith("audio"):
            return "<|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config

        self.multimodal_config = multimodal_config

        # Force to set attention implementation
        config.audio_encoder_config._attn_implementation = "sdpa"
        self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)

        self.quant_config = quant_config

        self.embed_tokens = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            config.pad_token_id,
        )

        if config.audio_adapter_type == "dual_ffn_fast_forward":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: HiggsAudioDualFFNDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        elif config.audio_adapter_type == "stack":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: LlamaDecoderLayer(
                    config=config.text_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        else:
            raise NotImplementedError(
                f"Audio adapter type {config.audio_adapter_type} not implemented."
            )
        self.norm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False
        self.rotary_emb = get_rope(
            head_size=config.text_config.head_dim,
            rotary_dim=config.text_config.head_dim,
            max_position=config.text_config.max_position_embeddings,
            base=config.text_config.rope_theta,
            rope_scaling=config.text_config.rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.audio_encoder_proj = HiggsAudioFeatureProjector(vllm_config)
        # We add 1 for the audio_stream_bos token and 1
        # for theaudio_stream_eos token
        self.codebook_size = config.codebook_size + 2
        self.num_codebooks = config.num_codebooks

        # HACK: This is a hack to tell if it is a audio generation model
        # FIXME: This should be fixed. We should simply reply on the token
        # history to determine if we should generate audio out tokens.
        self.generate_audio_out_token = config.skip_audio_tower
        self.audio_tokenizer_type = os.getenv(
            "HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"
        )
        self.use_whisper_tokenizer = self.audio_tokenizer_type == "openai/whisper-large-v3-turbo"

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
                bias=False,
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.num_codebooks * self.codebook_size,
            config.text_config.hidden_size,
        )

        self.text_lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )

        self.audio_lm_head = ParallelLMHead(
            config.num_codebooks * self.codebook_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )

        if get_pp_group().is_last_rank:
            self.audio_decoder_proj = HiggsAudioDecoderProjector(vllm_config)
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.text_config.vocab_size,
                config.text_config.vocab_size,
                logit_scale,
            )
            self.audio_logits_processor = LogitsProcessor(
                self.audio_lm_head.num_embeddings_padded,
                self.audio_lm_head.org_vocab_size,
                logit_scale,
            )
            self.sampler = get_sampler()

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[HiggsAudioInputs]:
        audio_out_ids = kwargs.pop("audio_out_ids", None)
        if audio_out_ids is None:
            return None

        if audio_out_ids is not None:
            audio_out_ids = _validate_and_reshape_mm_tensor(audio_out_ids, "audio_out_ids")
            # audio_out_ids_length = _validate_and_reshape_mm_tensor(
            #     audio_out_ids_length, "audio_out_ids_length")
        return HiggsAudioInputs(
            audio_out_ids=audio_out_ids,
        )

    def _process_audio_input(self, audio_input: HiggsAudioInputs) -> torch.Tensor:
        audio_features = audio_input["audio_features"]
        audio_feature_attention_mask = audio_input["audio_feature_attention_mask"]

        audio_features_flattened = audio_features.transpose(1, 0).reshape(
            audio_features.shape[1], -1
        )
        audio_features_embeddings = self._embed_audio_ids(audio_features_flattened)
        audio_features_attention_mask_flattened = audio_feature_attention_mask.flatten()
        masked_audio_features_embeddings = audio_features_embeddings[
            audio_features_attention_mask_flattened
        ]
        audio_features_lens = audio_feature_attention_mask.sum(-1)
        masked_audio_features_embeddings = torch.split(
            masked_audio_features_embeddings, audio_features_lens.tolist()
        )
        return masked_audio_features_embeddings

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.num_codebooks, device=audio_ids.device) * self.codebook_size
        )
        codebook_shift = codebook_shift.unsqueeze(-1)
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift)
        audio_embed = torch.sum(audio_embed, dim=0)
        if self.config.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        if audio_input["audio_features"] is not None:
            masked_audio_features = self._process_audio_input(audio_input)
        else:
            masked_audio_features = None
        if kwargs.get("audio_out_ids", None) is not None:
            audio_out_ids = kwargs["audio_out_ids"]
            audio_out_flattened = audio_out_ids.transpose(1, 0)
            audio_out_embeddings = self._embed_audio_ids(audio_out_flattened)
            audio_out_embeddings = torch.chunk(audio_out_embeddings, audio_out_ids.shape[0], dim=0)
            if masked_audio_features is not None:
                masked_audio_features.extend(audio_out_embeddings)
            else:
                masked_audio_features = audio_out_embeddings

        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [
                    self.config.audio_in_idx,
                    self.config.audio_out_idx,
                ],
            )

        return inputs_embeds

    def get_language_model(self) -> torch.nn.Module:
        """
        Return the underlying language model used for text generation.
        For this architecture, the current module encapsulates the
        core text model (embedding, decoder layers, norm).
        """
        return self

    def get_input_mm_map(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.isin(
            input_ids,
            torch.tensor(
                [
                    self.config.audio_in_idx,
                    self.config.audio_out_idx,
                ],
                device=input_ids.device,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            # NOTE: In v1, inputs_embeds is always generated at model runner,
            # this condition is for v0 compatibility.
            if inputs_embeds is None:
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
                input_ids = None
            hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                hidden_states, _ = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=None,
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_logits = self.logits_processor(self.text_lm_head, hidden_states, sampling_metadata)
        if self.generate_audio_out_token:
            audio_logits = self.audio_logits_processor(self.audio_lm_head, hidden_states, None)
            audio_logits = audio_logits.view(-1, self.num_codebooks, self.codebook_size).float()
        else:
            audio_logits = None
        return text_logits, audio_logits

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        raise NotImplementedError("Not implemented")

    def sample_with_multimodal_metadata(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        multimodal_metadata: MultimodalMetadata,
    ) -> Optional[SamplerOutput]:
        if isinstance(logits, tuple):
            logits, audio_logits = logits
        else:
            audio_logits = None
        next_tokens = self.sampler(logits, sampling_metadata)
        next_mm_tokens = None
        n_reqs = logits.shape[0]

        # Check which stage we are in
        # 0: text generation mode
        # 1: audio generation mode initialization
        # 2: audio generation mode in progress
        audio_generation_mode = [0] * n_reqs
        if self.generate_audio_out_token:
            for i in range(n_reqs):
                last_prompt_token_id = multimodal_metadata.last_prompt_token_ids[i]
                output_token_ids = sampling_metadata.output_token_ids[i]
                if (
                    len(output_token_ids) > 0
                    and output_token_ids[-1] == self.config.audio_out_bos_id
                ) or (
                    len(output_token_ids) == 0
                    and last_prompt_token_id == self.config.audio_out_bos_id
                ):
                    # check if the previous token is audio_out_bos. If so, we should always generate <|AUDIO_OUT|>
                    # Start the audio generation mode
                    audio_generation_mode[i] = 1
                elif (
                    len(output_token_ids) > 0 and output_token_ids[-1] == self.config.audio_out_idx
                ):
                    # Still in the audio generation mode
                    audio_generation_mode[i] = 2

            assert audio_logits is not None
            audio_logits = audio_logits.reshape(-1, self.codebook_size)
            mm_sampling_metadata = self.prepare_mm_sampling_metadata(sampling_metadata)
            next_mm_tokens = self.sampler(audio_logits, mm_sampling_metadata)
            next_mm_tokens.sampled_token_ids = next_mm_tokens.sampled_token_ids.reshape(
                -1, self.num_codebooks
            )

            # Check if we are generating the audio tokens
            for i in range(n_reqs):
                if audio_generation_mode[i] == 1:
                    # Generate start of the audio stream
                    next_mm_tokens.sampled_token_ids[i] = self.config.audio_stream_bos_id
                elif audio_generation_mode[i] == 2:
                    # Update the next mm tokens based on the delay pattern
                    num_audio_delay = multimodal_metadata.num_audio_delays[i]
                    num_audio_eos = multimodal_metadata.num_audio_eos[i]

                    # Generate the delayed for the first few tokens
                    if num_audio_delay < self.num_codebooks:
                        next_mm_tokens.sampled_token_ids[i][num_audio_delay:] = (
                            self.config.audio_stream_bos_id
                        )

                    # Generate the eos token for the last few tokens
                    if num_audio_eos < self.num_codebooks:
                        all_eos_indices = torch.where(
                            next_mm_tokens.sampled_token_ids[i] == self.config.audio_stream_eos_id
                        )[0]
                        if all_eos_indices.shape[0] > 0:
                            last_eos_index = all_eos_indices[-1]
                            next_mm_tokens.sampled_token_ids[i][:last_eos_index] = (
                                self.config.audio_stream_eos_id
                            )
                    elif num_audio_eos >= self.num_codebooks:
                        # We already generated the last audio token,
                        # so we should just generate the eos token for the text
                        next_mm_tokens.sampled_token_ids[i] = -1

                else:
                    next_mm_tokens.sampled_token_ids[i] = -1

        return next_tokens, next_mm_tokens

    def prepare_mm_sampling_metadata(self, sampling_metadata: SamplingMetadata) -> SamplingMetadata:
        mm_sampling_metadata = copy.copy(sampling_metadata)
        if sampling_metadata.top_k is not None:
            mm_sampling_metadata.top_k = sampling_metadata.top_k.clip(
                max=self.codebook_size
            ).repeat_interleave(self.num_codebooks)
        if sampling_metadata.top_p is not None:
            mm_sampling_metadata.top_p = sampling_metadata.top_p.repeat_interleave(
                self.num_codebooks
            )
        if sampling_metadata.temperature is not None:
            mm_sampling_metadata.temperature = sampling_metadata.temperature.repeat_interleave(
                self.num_codebooks
            )
        return mm_sampling_metadata

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.audio_adapter_type == "stack":
                audio_param_names = [
                    "audio_attn",
                    "audio_input_layernorm",
                    "audio_mlp",
                    "audio_post_attention_layernorm",
                ]
                if any(p in name for p in audio_param_names):
                    continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):  # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            if "audio_tower" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


def convert_audio_to_base64(
    audio: np.ndarray, sampling_rate: int, target_format: str = "wav"
) -> str:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sampling_rate, format=target_format)
    return base64.b64encode(audio_buffer.getvalue()).decode("utf-8")


class HiggsAudioServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        chat_template_content_format: ChatTemplateContentFormatOption,
        chat_template: Optional[str] = None,
        return_tokens_as_token_ids: bool = False,
        enable_reasoning: bool = False,
        reasoning_parser: Optional[str] = None,
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
        audio_tokenizer: Optional[AudioTokenizer] = None,
    ):
        super().__init__(
            engine_client=engine_client,
            model_config=model_config,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.request_logger = request_logger
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_reasoning = enable_reasoning
        self.reasoning_parser = reasoning_parser

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                '"auto" tool choice has been enabled please note that while'
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored."
            )

        self.enable_reasoning: bool = enable_reasoning
        self.reasoning_parser: Optional[Callable[[AnyTokenizer], ReasoningParser]] = None
        if self.enable_reasoning:
            try:
                self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser
                )
            except Exception as e:
                raise TypeError(
                    "Error: --enable-reasoning requires "
                    f"reasoning_parser:'{reasoning_parser}' "
                    "which has not been registered"
                ) from e

        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if tool_parser == "pythonic" and model_config.model.startswith(
                    "meta-llama/Llama-3.2"
                ):
                    logger.warning("Llama3.2 models may struggle to emit valid pythonic tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(tool_parser)
            except Exception as e:
                raise TypeError(
                    "Error: --enable-auto-tool-choice requires "
                    f"tool_parser:'{tool_parser}' which has not "
                    "been registered"
                ) from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info(
                "Using default chat sampling params from %s: %s",
                source,
                self.default_sampling_params,
            )

        self.audio_tokenizer = audio_tokenizer
        self.audio_num_codebooks = self.audio_tokenizer.num_codebooks
        self.audio_stream_bos_id = model_config.hf_config.audio_stream_bos_id
        self.audio_stream_eos_id = model_config.hf_config.audio_stream_eos_id
        self.audio_codebook_size = self.audio_tokenizer.codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)

    # ruff: noqa: E501  # Disable specific lint rules
    def get_chat_template(self) -> str:
        if self.chat_template is not None:
            return self.chat_template

        # fmt: off
        return (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + "
                    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}"
                        "{% set content = bos_token + content %}"
                    "{% endif %}"
                    "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
                        "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
                    "{% endif %}"
                    "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )
        # fmt: on

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None

        # Hold the audio token
        audio_tokens_cache = [
            np.ndarray((0, self.audio_num_codebooks), dtype=np.int64) for _ in range(num_choices)
        ]
        is_first_audio_chunk = [True] * num_choices
        fade_out_audio = [None] * num_choices

        audio_chunk_size = self.audio_tokenizer_tps
        audio_chunk_overlap_size = self.audio_tokenizer_tps
        if request.audio is not None and hasattr(request.audio, "audio_chunk_size"):
            audio_chunk_size = request.audio.audio_chunk_size or audio_chunk_size
            audio_chunk_overlap_size = (
                request.audio.audio_chunk_overlap_size or audio_chunk_overlap_size
            )

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name and self._should_stream_with_auto_tool_parsing(request)
        )

        should_stream_with_reasoning_parsing = self._should_stream_with_reasoning_parsing(request)

        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned: Optional[list[bool]] = None

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto or should_stream_with_reasoning_parsing:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            previous_texts = [""] * num_choices
            function_name_returned = [False] * num_choices
            all_previous_token_ids = None
        else:
            previous_texts, all_previous_token_ids = None, None

        try:
            # There is no need to check if the reasoning_parser is None
            # because the should_stream_with_reasoning_parsing check
            # already ensures that the reasoning_parser is not None.
            # but the pre-commit hook requires it.
            if should_stream_with_reasoning_parsing and self.reasoning_parser is not None:
                reasoning_parser = self.reasoning_parser(tokenizer)
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and self.tool_parser:
                tool_parsers: list[Optional[ToolParser]] = [
                    self.tool_parser(tokenizer)
                ] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage
            include_continuous_usage = include_usage and stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    tool_parser = tool_parsers[i]

                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    # Process audio tokens
                    audio_chunk = None
                    if output.mm_token_ids is None:
                        if audio_tokens_cache[i].shape[0] > 0:
                            audio_chunk, fade_out_audio[i] = create_audio_chunk(
                                audio_tokens_cache[i],
                                audio_chunk_size,
                                fade_out_audio[i],
                                finalize=True,
                                audio_tokenizer=self.audio_tokenizer,
                                audio_codebook_size=self.audio_codebook_size,
                                samples_per_token=self.samples_per_token,
                                audio_num_codebooks=self.audio_num_codebooks,
                                audio_stream_bos_id=self.audio_stream_bos_id,
                                audio_stream_eos_id=self.audio_stream_eos_id,
                            )
                            audio_tokens_cache[i] = np.ndarray(
                                (0, self.audio_num_codebooks), dtype=np.int64
                            )
                            fade_out_audio[i] = None
                            # Reset the flag for the next audio sequences
                            is_first_audio_chunk[i] = True
                    else:
                        audio_tokens_cache[i] = np.concatenate(
                            [
                                audio_tokens_cache[i],
                                output.mm_token_ids,
                            ],
                            axis=0,
                        )
                        curr_audio_chunk_size = audio_tokens_cache[i].shape[0]

                        # The first audio chunk is generated with with less tokens than other chunks
                        # to reduce the first audio latency
                        if is_first_audio_chunk[i] and curr_audio_chunk_size >= (
                            10 + self.audio_num_codebooks - 1
                        ):
                            first_audio_chunk_size = int(10 - self.audio_num_codebooks + 1)
                            audio_chunk, fade_out_audio[i] = create_audio_chunk(
                                audio_tokens_cache[i],
                                first_audio_chunk_size,
                                fade_out_audio[i],
                                finalize=False,
                                audio_tokenizer=self.audio_tokenizer,
                                audio_codebook_size=self.audio_codebook_size,
                                samples_per_token=self.samples_per_token,
                                audio_num_codebooks=self.audio_num_codebooks,
                                audio_stream_bos_id=self.audio_stream_bos_id,
                                audio_stream_eos_id=self.audio_stream_eos_id,
                            )
                            audio_tokens_cache[i] = audio_tokens_cache[i][first_audio_chunk_size:]
                            is_first_audio_chunk[i] = False
                        elif not is_first_audio_chunk[i] and curr_audio_chunk_size >= (
                            audio_chunk_size + audio_chunk_overlap_size
                        ):
                            audio_chunk, fade_out_audio[i] = create_audio_chunk(
                                audio_tokens_cache[i],
                                audio_chunk_size,
                                fade_out_audio[i],
                                finalize=False,
                                audio_tokenizer=self.audio_tokenizer,
                                audio_codebook_size=self.audio_codebook_size,
                                samples_per_token=self.samples_per_token,
                                audio_num_codebooks=self.audio_num_codebooks,
                                audio_stream_bos_id=self.audio_stream_bos_id,
                                audio_stream_eos_id=self.audio_stream_eos_id,
                            )
                            audio_tokens_cache[i] = audio_tokens_cache[i][audio_chunk_size:]

                    delta_text = output.text

                    if not delta_text and not output.token_ids and not previous_num_tokens[i]:
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: Optional[DeltaMessage]

                    # just update previous_texts and previous_token_ids
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        current_token_ids = previous_token_ids + list(output.token_ids)

                    # handle streaming deltas for tools with named tool_choice
                    if tool_choice_function_name:
                        if self.enable_reasoning and not reasoning_parser.is_reasoning_end(
                            previous_token_ids
                        ):
                            assert reasoning_parser is not None
                            delta_message = reasoning_parser.extract_reasoning_content_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                output.token_ids,
                            )
                            # When encountering think end id in delta_token_ids,
                            # process the `content`. Only keep 'content',
                            # remove 'reasoning_content'
                            if reasoning_parser.is_reasoning_end(list(output.token_ids)):
                                if delta_message and delta_message.content:
                                    # This need to be added to next `delta_text`
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""
                        else:
                            # Just to add remaining `content`
                            if self.enable_reasoning:
                                delta_text = previous_text + delta_text
                                current_text = ""

                            delta_message = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        function=DeltaFunctionCall(
                                            name=tool_choice_function_name,
                                            arguments=delta_text,
                                        ),
                                        index=i,
                                    )
                                ]
                            )

                    elif request.tool_choice == "required":
                        assert previous_texts is not None
                        assert function_name_returned is not None
                        previous_text = previous_texts[i]
                        current_text = previous_text + delta_text
                        fn_name_returned = function_name_returned[i]

                        delta_message, function_name_returned[i] = (
                            self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned,
                            )
                        )

                        # update the previous values for the next iteration
                        previous_texts[i] = current_text

                    # handle streaming deltas for tools with "auto" tool choice
                    # and reasoning parser
                    elif tool_choice_auto and self.enable_reasoning:
                        assert tool_parser is not None
                        assert reasoning_parser is not None
                        assert added_content_delta_arr is not None
                        assert reasoning_end_arr is not None
                        if not reasoning_end_arr[i]:
                            delta_message = reasoning_parser.extract_reasoning_content_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                output.token_ids,
                            )

                            # When encountering think end id in delta_token_ids,
                            # set reasoning status to end.
                            # Remove the text and token ids related
                            # to 'reasoning_content'.
                            if reasoning_parser.is_reasoning_end(list(output.token_ids)):
                                reasoning_end_arr[i] = True
                                current_token_ids = reasoning_parser.extract_content_ids(
                                    list(output.token_ids)
                                )
                                if delta_message and delta_message.content:
                                    current_text = delta_message.content
                                    delta_message.content = None
                                else:
                                    current_text = ""

                        # handle tool calls only after reasoning is done,
                        else:
                            delta_token_ids = list(output.token_ids)
                            # First time to tool call,
                            # add the remaining text and token ids
                            # to delta from previous
                            if not added_content_delta_arr[i]:
                                added_content_delta_arr[i] = True
                                previous_text = ""
                                previous_token_ids = []
                                delta_text = current_text
                                delta_token_ids = current_token_ids

                            delta_message = tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=delta_token_ids,
                                request=request,
                            )
                    # when only tool calls
                    elif tool_choice_auto:
                        assert tool_parser is not None
                        delta_message = tool_parser.extract_tool_calls_streaming(
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            delta_token_ids=output.token_ids,
                            request=request,
                        )
                    # when only reasoning
                    elif self.enable_reasoning:
                        assert reasoning_parser is not None
                        delta_message = reasoning_parser.extract_reasoning_content_streaming(
                            previous_text,
                            current_text,
                            delta_text,
                            previous_token_ids,
                            current_token_ids,
                            output.token_ids,
                        )
                    # handle streaming just a content delta
                    else:
                        delta_message = DeltaMessage(content=delta_text, audio=audio_chunk)

                    # update the previous values for the next iteration
                    if tool_choice_auto or should_stream_with_reasoning_parsing:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                        )

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using guided decoding
                        auto_tools_called = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1 if auto_tools_called else 0
                            )
                        else:
                            index = 0

                        if (
                            self._should_check_for_unstreamed_tool_arg_tokens(delta_message, output)
                            and tool_parser
                        ):
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments,
                                str,
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON
                            expected_call = json.dumps(
                                tool_parser.prev_tool_call_arr[index].get("arguments", {}),
                                ensure_ascii=False,
                            )

                            # get what we've streamed so far for arguments
                            # for the current tool
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                actual_call = actual_call[:-latest_delta_len]

                            # check to see if there's anything left to stream
                            remaining_call = expected_call.replace(actual_call, "", 1)
                            # set that as a delta message
                            delta_message = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=index,
                                        function=DeltaFunctionCall(
                                            arguments=remaining_call
                                        ).model_dump(exclude_none=True),
                                    )
                                ]
                            )

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason
                            if not auto_tools_called
                            else "tool_calls",
                            stop_reason=output.stop_reason,
                        )

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Process any remaining audio tokens if any
            for i in range(num_choices):
                if audio_tokens_cache[i].shape[0] > 0:
                    audio_chunk, fade_out_audio[i] = create_audio_chunk(
                        audio_tokens_cache[i],
                        audio_chunk_size,
                        fade_out_audio[i],
                        audio_tokenizer=self.audio_tokenizer,
                        audio_codebook_size=self.audio_codebook_size,
                        samples_per_token=self.samples_per_token,
                        audio_num_codebooks=self.audio_num_codebooks,
                        audio_stream_bos_id=self.audio_stream_bos_id,
                        audio_stream_eos_id=self.audio_stream_eos_id,
                        finalize=True,
                    )
                    if audio_chunk is not None:
                        delta_message = DeltaMessage(audio=audio_chunk)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i, delta=delta_message, finish_reason=None
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            model_name = self._get_model_name(request.model, lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            if request.tool_choice == "auto" and not (
                self.enable_auto_tools and tool_parser is not None
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            tool_dicts = (
                None if request.tools is None else [tool.model_dump() for tool in request.tools]
            )

            chat_template = request.chat_template or self.get_chat_template()
            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]
                default_max_tokens = self.max_model_len - len(engine_prompt["prompt_token_ids"])
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, self.default_sampling_params
                    )
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )

                self._log_inputs(
                    request_id,
                    request_prompts[i],
                    params=sampling_params,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                )

                trace_headers = (
                    None
                    if raw_request is None
                    else await self._get_trace_headers(raw_request.headers)
                )

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            should_stream_with_reasoning_parsing = self._should_stream_with_reasoning_parsing(
                request
            )

            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = False

            if should_stream_with_reasoning_parsing and self.reasoning_parser is not None:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = reasoning_parser.extract_reasoning_content(
                    output.text, request=request
                )
            else:
                reasoning_content = None
                content = output.text

            # Post-process the audio tokens to audio waveform
            if output.mm_token_ids is not None:
                audio_datas = split_interleaved_delayed_audios(
                    output.mm_token_ids,
                    self.audio_tokenizer,
                    self.audio_stream_eos_id,
                )

                wv_list = []
                for audio_data in audio_datas:
                    audio_data = np.array(audio_data, dtype=np.int64)[1:-1, :]
                    reverted_audio_data = revert_delay_pattern(audio_data.transpose(1, 0))
                    reverted_audio_data = reverted_audio_data.clip(0, self.audio_codebook_size - 1)
                    wv, sampling_rate = self.audio_tokenizer.decode(vq_code=reverted_audio_data)
                    wv_list.append(wv)
                wv_numpy = np.concatenate(wv_list)

                # Convert audio to base64
                response_audio_format = "wav" if request.audio is None else request.audio.format
                audio_base64 = convert_audio_to_base64(
                    wv_numpy, sampling_rate, response_audio_format
                )
            else:
                audio_base64 = None

            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content=content,
                    audio=ChatCompletionAudio(
                        id=f"audio-{random_uuid()}",
                        data=audio_base64,
                        expires_at=0,
                        transcript="",
                    ),
                )

            # if the request uses tools and specified a tool choice
            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                tool_call_class = ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=content,
                            )
                        )
                    ],
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(output.text)
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        tool_call_class(
                            function=FunctionCall(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.parameters),
                            )
                        )
                        for tool_call in tool_calls
                    ],
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content=content,
                    audio=ChatCompletionAudio(
                        id=f"audio-{random_uuid()}",
                        data=audio_base64,
                        expires_at=0,
                        transcript="",
                    ),
                )

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    content if content is not None else "", request=request
                )
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=tool_call_info.content,
                        tool_calls=tool_call_info.tool_calls,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content=content,
                    audio=ChatCompletionAudio(
                        id=f"audio-{random_uuid()}",
                        data=audio_base64,
                        expires_at=0,
                        transcript="",
                    ),
                )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if auto_tools_called
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
        )

        return response

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_stream_with_reasoning_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the
        reasoning parser that was configured.

        We only want to do this IF reasoning is enabled and a reasoning
        parser is configured.
        """
        return self.enable_reasoning and self.reasoning_parser is not None

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: Optional[DeltaMessage],
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        # yapf: disable
        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools and self.tool_parser and delta_message
            and delta_message.tool_calls and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )


def token2wav(
    token: np.ndarray,
    audio_chunk_size: int,
    audio_tokenizer: AudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    fade_out_audio: Optional[np.ndarray] = None,
    finalize: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if token.shape[0] <= audio_num_codebooks + 2:
        logger.warning(
            "The audio token length %s is too short. Skipping this chunk.", token.shape[0]
        )
        return None, None

    audio_datas = split_interleaved_delayed_audios(token, audio_tokenizer, audio_stream_eos_id)

    audio_codes_list = []
    for audio_data in audio_datas:
        # Prune the first and last stream bos/eos tokens
        if np.all(audio_data[0] == audio_stream_bos_id):
            audio_data = audio_data[1:]
            audio_chunk_size -= 1
        if np.all(audio_data[-1] == audio_stream_eos_id):
            audio_data = audio_data[:-1]
            audio_chunk_size -= 1

        audio_data = audio_data.transpose(1, 0)
        audio_codes = revert_delay_pattern(audio_data).clip(0, audio_codebook_size - 1)
        audio_codes_list.append(audio_codes)

    audio_codes = np.concatenate(audio_codes_list, axis=1)
    tts_speech, _ = audio_tokenizer.decode(vq_code=audio_codes)
    if fade_out_audio is not None:
        hamming_window_len = min(2 * len(fade_out_audio), samples_per_token)
        hamming_window = _get_hamming_window(hamming_window_len)
        fade_overlap = hamming_window_len // 2
        tts_speech[:fade_overlap] = (
            tts_speech[:fade_overlap] * hamming_window[:fade_overlap]
            + fade_out_audio[:fade_overlap] * hamming_window[fade_overlap:]
        )

    fade_out_audio = tts_speech[audio_chunk_size * samples_per_token :]
    if not finalize:
        tts_speech = tts_speech[: audio_chunk_size * samples_per_token]
    else:
        fade_out_audio = None
    return tts_speech, fade_out_audio


def create_audio_chunk(
    audio_tokens_cache: np.ndarray,
    audio_chunk_size: int,
    fade_out_audio: Optional[np.ndarray],
    audio_tokenizer: AudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    audio_stream_bos_id: int,
    audio_stream_eos_id: int,
    finalize: bool = False,
    return_as_numpy_audio: bool = False,
) -> tuple[Optional[ChatCompletionAudio], np.ndarray]:
    new_audio, new_fade_out_audio = token2wav(
        audio_tokens_cache,
        audio_chunk_size,
        fade_out_audio=fade_out_audio,
        finalize=finalize,
        audio_tokenizer=audio_tokenizer,
        audio_codebook_size=audio_codebook_size,
        samples_per_token=samples_per_token,
        audio_num_codebooks=audio_num_codebooks,
        audio_stream_bos_id=audio_stream_bos_id,
        audio_stream_eos_id=audio_stream_eos_id,
    )

    if return_as_numpy_audio:
        return new_audio, new_fade_out_audio

    audio_pcm16 = (new_audio * np.iinfo(np.int16).max).astype(np.int16)

    return ChatCompletionAudio(
        id=f"audio-{random_uuid()}",
        data=base64.b64encode(audio_pcm16).decode("utf-8"),
        expires_at=0,
        transcript="",
    ), new_fade_out_audio


@lru_cache
def _get_hamming_window(len):
    return np.hamming(len)


def split_interleaved_delayed_audios(
    audio_data: Union[list[list[int]], np.ndarray],
    audio_tokenizer: AudioTokenizer,
    audio_stream_eos_id: int,
) -> list[tuple[list[list[int]], np.ndarray]]:
    separator = [audio_stream_eos_id] * audio_tokenizer.num_codebooks

    # Convert separator to numpy array if audio_data is numpy array
    if isinstance(audio_data, np.ndarray):
        separator = np.array(separator)
        # Find the indices where the rows equal the separator
        split_indices = np.where(np.all(audio_data == separator, axis=1))[0]
        start = 0
        groups = []
        for idx in split_indices:
            groups.append(audio_data[start:idx])
            start = idx + 1
        if start < len(audio_data):
            groups.append(audio_data[start:])
    else:
        groups = []
        current = []
        for row in audio_data:
            current.append(row)

            # Handle comparison for both list and numpy array types
            if isinstance(audio_data, np.ndarray):
                if np.array_equal(row, separator):
                    groups.append(current)
                    current = []
            else:
                if row == separator:
                    groups.append(current)
                    current = []

        # Don't forget the last group if there's no trailing separator
        if current:
            groups.append(current)

    return groups
