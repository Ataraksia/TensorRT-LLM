# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Union, Dict, Any, List
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig


class HiggsAudioConfig(PretrainedConfig):
    """TensorRT-LLM configuration for Higgs Audio multimodal TTS model.

    This configuration class manages parameters for the Higgs Audio text-to-speech model,
    which combines a Llama 3.2 3B text backbone with Whisper-like audio processing
    capabilities. The configuration flattens multimodal parameters into a single
    unified config for simplified engine building and runtime management.

    The underlying architecture consists of:
    - Text backbone: Llama-like transformer for language modeling
    - Audio encoder: Whisper-like architecture for audio feature extraction  
    - Fusion mechanism: Prompt table integration for audio-text generation
    - TTS optimizations: Real-time performance and streaming support

    Attributes:
        Text Backbone Parameters:
            architecture (str): Model architecture type (default: "LlamaForCausalLM")
            num_hidden_layers (int): Number of transformer layers (default: 32)
            num_attention_heads (int): Number of attention heads (default: 32)
            num_key_value_heads (Optional[int]): Number of KV heads for GQA
            hidden_size (int): Hidden dimension size (default: 4096)
            intermediate_size (int): FFN intermediate size (default: 11008)
            head_size (Optional[int]): Attention head dimension
            vocab_size (int): Vocabulary size (default: 128256)
            max_position_embeddings (int): Maximum sequence length (default: 8192)

        Audio Encoder Parameters:
            audio_num_mel_bins (int): Mel spectrogram bins (default: 128)
            audio_encoder_layers (int): Audio encoder layers (default: 32)
            audio_encoder_heads (int): Audio attention heads (default: 20)
            audio_encoder_ffn_dim (int): Audio FFN dimension (default: 5120)
            audio_d_model (int): Audio model dimension (default: 1280)
            audio_max_source_positions (int): Max audio sequence length (default: 1500)

        Audio-Text Fusion Parameters:
            audio_adapter_type (str): Adapter architecture type (default: "stack")
            audio_dual_ffn_layers (Optional[List[int]]): DualFFN layer indices
            encode_whisper_embed (bool): Use Whisper embeddings (default: True)
            use_delay_pattern (bool): Enable delay pattern for streaming (default: False)
            audio_num_codebooks (int): Number of RVQ codebooks (default: 12)
            audio_codebook_size (int): Codebook vocabulary size (default: 1024)

        TTS-Specific Parameters:
            audio_delay_pattern_strategy (str): Delay pattern strategy ('linear', 'exponential', 'custom', 'none') (default: 'linear')
            audio_delay_pattern_stride (int): Delay pattern stride (default: 1)
            audio_delay_pattern_custom_delays (Optional[List[int]]): Custom delays for 'custom' strategy (default: None)
            audio_delay_pattern_max_delay (Optional[int]): Maximum allowed delay (default: None)

    Example:
        >>> config = HiggsAudioConfig.from_hugging_face("path/to/higgs/model")
        >>> model = HiggsAudioForCausalLM(config)
    """

    def __init__(
        self,
        *,
        # Text (LLM) backbone parameters
        architecture: str = "LlamaForCausalLM",
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: Optional[int] = 8,
        hidden_size: int = 4096,
        intermediate_size: int = 8192,
        head_size: Optional[int] = 128,
        vocab_size: int = 128256,
        max_position_embeddings: int = 8192,
        position_embedding_type: str = "rope_gpt_neox",
        rotary_embedding_dim: Optional[int] = None,
        rotary_base: float = 100000.0,
        rotary_scaling: Optional[Dict[str, Any]] = None,
        hidden_act: str = "silu",
        norm_epsilon: float = 1e-5,
        attn_bias: bool = False,
        seq_length: int = 1024,
        # Audio encoder (Whisper-like) parameters
        audio_num_mel_bins: int = 128,
        audio_encoder_layers: int = 32,
        audio_encoder_heads: int = 24,
        audio_encoder_ffn_dim: int = 5120,
        audio_d_model: int = 1280,
        audio_max_source_positions: int = 1500,
        # Audio-text fusion and token parameters
        audio_adapter_type: str = "dual_ffn_fast_forward",
        audio_embed_avg: bool = False,
        audio_dual_ffn_layers: Optional[List[int]] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],   
        audio_decoder_proj_num_layers: int = 0,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = True,
        use_delay_pattern: bool = True,
        skip_audio_tower: bool = False,
        use_audio_out_embed_projector: bool = False,
        use_audio_out_self_attention: bool = False,
        audio_num_codebooks: int = 8,
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
        # CUDA Graph optimization parameters for TTS workloads
        # Enable/disable CUDA graphs for different TTS components
        cuda_graph_enable: bool = True,
        cuda_graph_enable_streaming: bool = True,
        cuda_graph_enable_delay_patterns: bool = True,
        
        # TTS-specific batch sizes and sequence lengths for graph optimization
        cuda_graph_tts_batch_sizes: List[int] = [1],
        cuda_graph_tts_sequence_lengths: List[int] = [1024],
        cuda_graph_streaming_chunk_sizes: List[int] = [32],
        cuda_graph_max_cache_size: int = 32,
        cuda_graph_memory_pool_size_gb: int = 2,
        # Performance optimization settings
        cuda_graph_enable_performance_monitoring: bool = True,
        cuda_graph_fallback_enabled: bool = True,
        
        # DualFFN-specific graph optimization
        cuda_graph_dualffn_separate_graphs: bool = True,
        cuda_graph_dualffn_audio_text_ratio_threshold: float = 0.3,
        
        # Delay pattern optimization for RVQ multi-codebook coordination
        cuda_graph_delay_pattern_max_codebooks: int = 16,
        cuda_graph_delay_pattern_optimization_enabled: bool = True,
        
        cuda_graph_validation_enabled: bool = True,
        # TensorRT-LLM common parameters
        memory_efficient_build: bool = True,
        dtype: str = "bfloat16",
        mapping: Optional[Mapping] = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        quantization: Optional[QuantConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HiggsAudioConfig with comprehensive parameter validation.

        Args:
            architecture: Model architecture identifier
            num_hidden_layers: Number of transformer decoder layers
            num_attention_heads: Number of attention heads in each layer
            num_key_value_heads: Number of key-value heads (for GQA), defaults to num_attention_heads
            hidden_size: Hidden state dimension
            intermediate_size: FFN intermediate dimension
            head_size: Individual attention head dimension, computed if not provided
            vocab_size: Text vocabulary size
            max_position_embeddings: Maximum supported sequence length
            position_embedding_type: Type of positional encoding
            rotary_embedding_dim: RoPE embedding dimension
            rotary_base: RoPE base frequency
            rotary_scaling: RoPE scaling configuration
            hidden_act: Activation function for FFN
            norm_epsilon: Layer normalization epsilon
            attn_bias: Whether to use bias in attention layers
            seq_length: Default sequence length
            audio_num_mel_bins: Number of mel-frequency bins for audio processing
            audio_encoder_layers: Number of audio encoder transformer layers
            audio_encoder_heads: Number of attention heads in audio encoder
            audio_encoder_ffn_dim: Audio encoder FFN dimension
            audio_d_model: Audio encoder model dimension
            audio_max_source_positions: Maximum audio sequence length
            audio_adapter_type: Type of audio adapter ("stack", "concat", etc.)
            audio_embed_avg: Whether to average audio embeddings
            audio_dual_ffn_layers: Layer indices for DualFFN audio adaptation
            audio_decoder_proj_num_layers: Number of decoder projection layers
            encode_whisper_embed: Whether to encode Whisper embeddings
            encode_audio_in_tokens: Whether to encode audio as input tokens
            use_delay_pattern: Enable delay pattern for streaming generation
            skip_audio_tower: Skip audio tower processing
            use_audio_out_embed_projector: Use output embedding projector
            use_audio_out_self_attention: Use output self-attention
            audio_num_codebooks: Number of RVQ codebooks for audio tokenization
            audio_codebook_size: Size of each audio codebook
            audio_in_token_idx: Special token index for audio input
            audio_out_token_idx: Special token index for audio output
            audio_stream_bos_id: Beginning-of-stream token for audio
            audio_stream_eos_id: End-of-stream token for audio
            audio_out_bos_token_id: Beginning-of-output token for audio
            audio_eos_token_id: End-of-sequence token for audio
            bos_token_id: Beginning-of-sequence token for text
            eos_token_id: End-of-sequence token for text
            pad_token_id: Padding token for text
            dtype: Data type for model weights
            mapping: TensorRT-LLM tensor/pipeline parallelism mapping
            quantization: Quantization configuration
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If configuration parameters are invalid or incompatible
        """
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

        # Set CUDA graph parameters as instance attributes
        self.cuda_graph_enable = cuda_graph_enable
        self.cuda_graph_enable_streaming = cuda_graph_enable_streaming
        self.cuda_graph_enable_delay_patterns = cuda_graph_enable_delay_patterns
        
        self.cuda_graph_tts_batch_sizes = cuda_graph_tts_batch_sizes
        self.cuda_graph_tts_sequence_lengths = cuda_graph_tts_sequence_lengths
        self.cuda_graph_streaming_chunk_sizes = cuda_graph_streaming_chunk_sizes
        
        self.cuda_graph_max_cache_size = cuda_graph_max_cache_size
        self.cuda_graph_memory_pool_size_gb = cuda_graph_memory_pool_size_gb
        
        self.cuda_graph_enable_performance_monitoring = cuda_graph_enable_performance_monitoring
        self.cuda_graph_fallback_enabled = cuda_graph_fallback_enabled
        
        self.cuda_graph_dualffn_separate_graphs = cuda_graph_dualffn_separate_graphs
        self.cuda_graph_dualffn_audio_text_ratio_threshold = cuda_graph_dualffn_audio_text_ratio_threshold
        
        self.cuda_graph_delay_pattern_max_codebooks = cuda_graph_delay_pattern_max_codebooks
        self.cuda_graph_delay_pattern_optimization_enabled = cuda_graph_delay_pattern_optimization_enabled
        
        self.cuda_graph_validation_enabled = cuda_graph_validation_enabled
                
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

