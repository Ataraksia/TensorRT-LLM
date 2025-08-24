# Higgs-Audio (TensorRT-LLM) — Example Runner

This example mirrors the `qwen2audio` runtime to exercise a Higgs-Audio-style multimodal flow (audio + text) in TensorRT-LLM.

Status: scaffolding prototype. The audio encoder engine and the LLM engine must already be built. The model forward and weight conversion are WIP.

## Prerequisites
- Built LLM engine under `--engine_dir` (e.g., a Llama-like causal LM).
- Built audio encoder TensorRT engine (`--audio_engine_path`).
- Access to the Hugging Face tokenizer and processor for the target model (`--tokenizer_dir`).

## Run
```bash
python examples/models/core/higgs_audio/run.py \
  --engine_dir engine_outputs \
  --tokenizer_dir /path/to/hf-model-dir \
  --audio_engine_path plan/audio_encoder/audio_encoder_fp16.plan \
  --audio_url ./audio/sample.wav \
  --input_text "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:" \
  --max_new_tokens 64 \
  --gpu_id 0
```

Common generation/runtime knobs are available; see `utils.py`.

## Notes
- The example uses a prompt-table trick to feed audio features where `<|AUDIO|>` tokens occur.
- Depending on your audio encoder, output binding name may be `output` or `encoder_output`. The script handles both.
- This is an initial scaffold; the final integration will implement:
  - Proper Higgs-Audio forward path in `tensorrt_llm/models/higgs_audio/model.py`.
  - HF → TRT-LLM weight mapping in `tensorrt_llm/models/higgs_audio/convert.py`.
  - Builder/export updates in `tensorrt_llm/tools/multimodal_builder.py`.
```
