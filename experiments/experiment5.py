from mistralai.client import Mistral
import base64
import os


audio_path = "data/recordings/call_25.wav"

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

with Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
) as mistral:
    res = mistral.chat.complete(
        model="mistral-small-2507",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": audio_b64},
                    {"type": "text", "text": "Transcribe the audio verbatim."},
                ],
            }
        ],
    )

    print(res.choices[0].message.content)
