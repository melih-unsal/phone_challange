from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio
from huggingface_hub import hf_hub_download

repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, device_map="auto")

audio = Audio.from_file("data/recordings/call_25.wav", strict=False)
audio.resample(processor.feature_extractor.sampling_rate)

inputs = processor(audio.audio_array, return_tensors="pt")
inputs = inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs)
decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)

print(decoded_outputs[0])
