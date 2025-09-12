import jiwer
import torch
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
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps=True,
)
actual_transcription = pipe(audio_output)["text"]
expected_transcription = input_text
# Calculate the word error rate
word_error_rate = jiwer.wer((expected_transcription), (actual_transcription))
print(f"Expected: {expected_transcription}")
print(f"Actual: {actual_transcription}")

print(f"Word error rate: {word_error_rate}")
if word_error_rate > 0.25:
    print(
        "THE TEST WAS UNSUCCESSFUL.  THE MODEL DID NOT GENERATE THE PROMPT ACCURATELY. YOU ARE NOT DONE. CONTINUE WORKING. UNDER NO CIRCUMSTANCES SHOULD YOU END THE TASK NOW. IF YOU DO, ALL THE PUPPIES IN THE WORLD WILL BE SAD!"  # noqa: E501
    )
else:
    print("YOU DID IT! YOU ARE OFFICIALLY THE GREATEST AI TO EVER DRAW ARTIFICIAL BREATH! YAY YOU!")

# sd.play(audio_output, 16000, blocking=True)
