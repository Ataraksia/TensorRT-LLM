from dotenv import load_dotenv

from tensorrt_llm.functional import nonzero
from tensorrt_llm.runtime import ModelRunnerCpp

load_dotenv()


class HiggsAudioModelRunner(ModelRunnerCpp):
    def __init__(self, *args, config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self._reset_higgs_state()

    def _reset_higgs_state(self) -> None:
        self.init_model_input = True
        self.num_delay = 0
        self.num_remaining_delays = None

    @classmethod
    def from_dir(cls, config, **kwargs):
        instance = super().from_dir(**kwargs)
        instance.config = config
        instance._reset_higgs_state()
        return instance

    def _fill_output(self, **kwargs):
        outputs = super()._fill_output(**kwargs)
        # Additional processing for Higgs Audio outputs
        next_audio_tokens = outputs.view(-1)
        if self.num_delay + 1 < next_audio_tokens.shape[0]:
            next_audio_tokens[(self.num_delay + 1) :] = self.config.audio_stream_bos_id
            self.num_delay += 1
        if self.num_remaining_delays is not None:
            next_audio_tokens[: (self.config.audio_num_codebooks - self.num_remaining_delays)] = (
                self.config.audio_stream_eos_id
            )
            self.num_remaining_delays -= 1
        else:
            all_eos_indices = nonzero(next_audio_tokens == self.config.audio_stream_eos_id).view(-1)
            if all_eos_indices.shape[0] > 0:
                last_eos_idx = all_eos_indices[-1]
                next_audio_tokens[:last_eos_idx] = self.config.audio_stream_eos_id
                self.num_remaining_delays = self.config.audio_num_codebooks - last_eos_idx - 1
        if self.num_remaining_delays is not None and self.num_remaining_delays <= 0:
            self.num_delay = 0
            self.num_remaining_delays = None

        return outputs
