from dotenv import load_dotenv
import torch
from typing import Optional


from tensorrt_llm.runtime import ModelRunnerCpp

load_dotenv()


class HiggsAudioModelRunner(ModelRunnerCpp):
    def __init__(self, *args, config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.audio_in_start: int = 0
        self.audio_in_end: int = 0
        self.audio_out_start: int = 0
        self._reset_higgs_state()

    def _reset_higgs_state(self) -> None:
        self.num_bos: int = 0
        self.num_eos: Optional[int] = None

    @classmethod
    def from_dir(cls, config, **kwargs):
        instance = super().from_dir(**kwargs)
        instance.config = config
        instance._reset_higgs_state()
        return instance

    def _fill_output(self, **kwargs):
        outputs = super()._fill_output(**kwargs)
        # Additional processing for Higgs Audio outputs

        outputs = outputs.view(self.audio_num_codebooks, -1)
        if self.num_bos < self.config.audio_num_codebooks:
            outputs[self.num_bos :] = self.config.audio_stream_bos_id
            self.num_bos += 1
        if self.num_eos is not None:
            outputs[: self.config.audio_num_codebooks - self.num_eos] = (
                self.config.audio_stream_eos_id
            )
            self.num_eos -= 1
        else:
            all_eos_indices = (outputs == self.config.audio_stream_eos_id).nonzero()
            if torch.numel(all_eos_indices) > 0:
                all_eos_indices = all_eos_indices[0]
                last_eos_idx = all_eos_indices[-1]
                outputs[:last_eos_idx] = self.config.audio_stream_eos_id
                self.num_eos = self.config.audio_num_codebooks - last_eos_idx - 1
        if self.num_eos is not None and self.num_eos <= 0:
            self._reset_higgs_state()
            outputs[...] = 0

        return outputs.view(1, 1, -1)
