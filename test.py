import base64
import re

import jiwer
import soundfile as sf
import torch
from openai import OpenAI
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


client = OpenAI()


sf.write("output.wav", audio_output, 16000)

with open("output.wav", "rb") as f:
    wav_data = f.read()

encoded_string = base64.b64encode(wav_data).decode("utf-8")

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please rate this on a scale of 1 to 10 on how human-like it sounds.  Rate a 5 if it sounds like a robot speaking gibberish.",  # noqa: E501
                },
                {"type": "input_audio", "input_audio": {"data": encoded_string, "format": "wav"}},
            ],
        },
    ],
)

transcript = completion.choices[0].message.audio.transcript
print(transcript)


match = re.search(r"([0-9]+(?:\.[0-9]+)?)", transcript)
if match:
    rating = float(match.group(1))
    print(f"Judge rating: {rating}")
