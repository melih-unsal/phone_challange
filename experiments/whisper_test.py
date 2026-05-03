import whisper

model = whisper.load_model("large")
result = model.transcribe("data/recordings/call_25.wav")
print(result["text"])