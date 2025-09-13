import jiwer
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import AutoProcessor, pipeline

from run_chat_completion import AutoModelForSpeechSeq2Seq
from tensorrt_llm.models.higgs_audio.model import HiggsAudioTRTRunner  # noqa: F401

# Create configuration


# Instantiate model
gpu_device = torch.device("cuda", 0)
torch.cuda.set_device(gpu_device)

runner = HiggsAudioTRTRunner()

input_text = "Chat, stop backseating! I totally know what I'm doing... I think"

# Generate text/audio
audio_output = runner.generate(
    input_text,
)

model_id = "openai/whisper-large-v3-turbo"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)


# Load VAD model
silero_model = load_silero_vad()

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    audio_output, silero_model, sampling_rate=16000, min_silence_duration_ms=500
)

# Extract only speech segments
speech = []
for segment in speech_timestamps:
    start_sample = int(segment["start"])
    end_sample = int(segment["end"])
    speech.append(audio_output[start_sample:end_sample])

pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
)
actual_transcription = ""
for i in range(len(speech)):
    actual_transcription += pipe(speech[i])["text"]


# Calculate the word error rate
word_error_rate = jiwer.wer((input_text), (actual_transcription))
print(f"Expected: {input_text}")
print(f"Actual: {actual_transcription}")

print(f"Word error rate: {word_error_rate}")
if word_error_rate > 0.25:
    print(
        "THE TEST WAS UNSUCCESSFUL.  THE MODEL DID NOT GENERATE THE PROMPT ACCURATELY. YOU ARE NOT DONE. CONTINUE WORKING. UNDER NO CIRCUMSTANCES SHOULD YOU END THE TASK NOW. IF YOU DO, ALL THE PUPPIES IN THE WORLD WILL BE SAD!"  # noqa: E501
    )
else:
    print("YOU DID IT! YOU ARE OFFICIALLY THE GREATEST AI TO EVER DRAW ARTIFICIAL BREATH! YAY YOU!")

import sounddevice as sd

sd.play(audio_output, 16000, blocking=True)
