# Higgs Audio – TensorRT-LLM Runner

This directory contains the Higgs Audio model and a TensorRT-LLM-based runner for high-performance text+audio inference.

## HiggsAudioTRTRunner (in `model.py`)

A compact wrapper that wires:
- Hugging Face config/tokenizer + a local HF processor (`HFHiggsAudioProcessor`) to expand `<|AUDIO|>` placeholders
- Optional audio encoder plan via `tensorrt_llm.runtime.Session`
- `ModelRunnerCpp` to generate text conditioned on audio features via prompt-tuning tables

### Instantiate

```python
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner

runner = HiggsAudioTRTRunner(
    engine_dir="/path/to/text/engine_dir",
    tokenizer_dir="/path/to/hf_tokenizer",
    audio_engine_path="/path/to/audio_encoder.plan",  # required for Whisper-style audio
    num_beams=1,
    max_new_tokens=128,
)
```

### Run inference

```python
import numpy as np

# input_text: str, audios: List[np.ndarray] (mono waveform, float32)
text = runner.infer(
    input_text="Transcribe and summarize the audio:",
    audios=[np_waveform],
    temperature=0.7,
    top_p=0.9,
)
print(text)
```

### Notes
- The audio encoder TRT plan is expected to take inputs named `"input"` (features) and `"mask"`, and return `"output"`. Adjust `_run_audio_encoder` in `model.py` if your plan uses different I/O names.
- Whisper-style audio features and attention masks are produced by the HF processor; the runner computes downsampled lengths consistent with the encoder.
- The runner builds a prompt table from valid audio frames and replaces the `<|AUDIO|>` token with fake prompt IDs, passing `prompt_table` and `input_token_extra_ids` into `ModelRunnerCpp.generate`.
- For environments without C++ bindings, you can set `use_py_session=True` to use the Python runner (`ModelRunner`).
