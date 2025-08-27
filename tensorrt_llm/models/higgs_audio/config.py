# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Union

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class HiggsAudioConfig(PretrainedConfig):
    """TensorRT-LLM config for Higgs-Audio composition model.

    This mirrors key fields from the HF reference config at
    `higgs_audio/model/higgs_audio/configuration_higgs_audio.py` and flattens
    what we need for engine build/runtime. The underlying text backbone is Llama-like.
    """

    def __init__(
        self,
        *,
        # Text (LLM) backbone
        architecture: str = "LlamaForCausalLM",
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        head_size: Optional[int] = None,
        vocab_size: int = 128256,
        max_position_embeddings: int = 8192,
        position_embedding_type: str = "rope_gpt_neox",
        rotary_embedding_dim: Optional[int] = None,
        rotary_base: float = 100000.0,
        rotary_scaling: Optional[dict] = None,
        hidden_act: str = "silu",
        norm_epsilon: float = 1e-5,
        attn_bias: bool = False,
        seq_length: int = 8192,
        # Audio encoder (Whisper-like) essentials
        audio_num_mel_bins: int = 128,
        audio_encoder_layers: int = 32,
        audio_encoder_heads: int = 20,
        audio_encoder_ffn_dim: int = 5120,
        audio_d_model: int = 1280,
        audio_max_source_positions: int = 1500,
        # Audio-text fusion and tokens
        audio_adapter_type: str = "stack",
        audio_embed_avg: bool = False,
        audio_dual_ffn_layers: Optional[list[int]] = None,
        audio_decoder_proj_num_layers: int = 0,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        use_delay_pattern: bool = False,
        skip_audio_tower: bool = False,
        use_audio_out_embed_projector: bool = False,
        use_audio_out_self_attention: bool = False,
        audio_num_codebooks: int = 12,
        audio_codebook_size: int = 1024,
        audio_in_token_idx: int = 128015,
        audio_out_token_idx: int = 128016,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        audio_out_bos_token_id: int = 128013,
        audio_eos_token_id: int = 128012,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: int = 128001,
        # TRT-LLM common
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quantization: Optional[QuantConfig] = None,
        **kwargs,
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_size = head_size or (hidden_size // num_attention_heads)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.rotary_embedding_dim = rotary_embedding_dim
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.hidden_act = hidden_act
        self.norm_epsilon = norm_epsilon
        self.attn_bias = attn_bias
        self.seq_length = seq_length
        self.architecture = architecture

        # Audio encoder essentials
        self.audio_num_mel_bins = audio_num_mel_bins
        self.audio_encoder_layers = audio_encoder_layers
        self.audio_encoder_heads = audio_encoder_heads
        self.audio_encoder_ffn_dim = audio_encoder_ffn_dim
        self.audio_d_model = audio_d_model
        self.audio_max_source_positions = audio_max_source_positions

        # Audio/text fusion + tokens
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_dual_ffn_layers = audio_dual_ffn_layers or []
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.encode_whisper_embed = encode_whisper_embed
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(
            architecture=architecture,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            head_size=head_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            rotary_embedding_dim=rotary_embedding_dim,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            hidden_act=hidden_act,
            norm_epsilon=norm_epsilon,
            attn_bias=attn_bias,
            seq_length=seq_length,
            dtype=dtype,
            mapping=mapping,
            quantization=quantization,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def to_dict(self):
        d = super().to_dict()
        # Nothing special beyond defaults; ensure custom fields are serialized
        for k, v in self.__dict__.items():
            if k not in d:
                d[k] = v
        return d

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: Union[str, "transformers.PretrainedConfig"],
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "HiggsAudioConfig":
        import transformers
        trust_remote_code = kwargs.pop("trust_remote_code", True)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                str(hf_config_or_dir), trust_remote_code=trust_remote_code
            )

        # The HF HiggsAudio config is a composition; pull out text and audio encoder parts
        text_cfg = getattr(hf_config, "text_config", None)
        audio_enc_cfg = getattr(hf_config, "audio_encoder_config", None)
        assert text_cfg is not None and audio_enc_cfg is not None, (
            "Expected HiggsAudio HF config to contain text_config and audio_encoder_config"
        )

        inferred_dtype = infer_dtype(dtype, getattr(text_cfg, "torch_dtype", None))

        num_key_value_heads = getattr(
            text_cfg, "num_key_value_heads", text_cfg.num_attention_heads
        )
        rotary_scaling = getattr(text_cfg, "rope_scaling", None)
        rotary_base = getattr(text_cfg, "rope_theta", 100000.0)
        seq_length = getattr(text_cfg, "seq_length", getattr(text_cfg, "max_position_embeddings", 8192))
        attn_bias = getattr(text_cfg, "attn_bias", False)
        hidden_act = getattr(text_cfg, "hidden_act", "silu")

        return cls(
            architecture=getattr(text_cfg, "architectures", ["LlamaForCausalLM"])[0],
            dtype=inferred_dtype,
            num_hidden_layers=text_cfg.num_hidden_layers,
            num_attention_heads=text_cfg.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            head_size=getattr(text_cfg, "head_dim", None),
            vocab_size=text_cfg.vocab_size,
            position_embedding_type="rope_gpt_neox",
            rotary_embedding_dim=getattr(text_cfg, "rotary_dim", None),
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            hidden_act=hidden_act,
            norm_epsilon=getattr(text_cfg, "rms_norm_eps", 1e-5),
            attn_bias=attn_bias,
            seq_length=seq_length,
            # Audio encoder
            audio_num_mel_bins=audio_enc_cfg.num_mel_bins,
            audio_encoder_layers=audio_enc_cfg.encoder_layers,
            audio_encoder_heads=audio_enc_cfg.encoder_attention_heads,
            audio_encoder_ffn_dim=audio_enc_cfg.encoder_ffn_dim,
            audio_d_model=audio_enc_cfg.d_model,
            audio_max_source_positions=audio_enc_cfg.max_source_positions,
            # Keep TRT-LLM bookkeeping
            mapping=mapping,
            quantization=quant_config,
            **kwargs,
        )
