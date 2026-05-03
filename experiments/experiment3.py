from unittest import result

from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import os
import json
from tqdm import tqdm
from langchain_core.output_parsers import JsonOutputParser

def normalize_phone_number(phone_number):
    if len(phone_number) > 0:
        phone_number = phone_number.replace(" ", "")
        if phone_number.startswith("+00"):
            phone_number = "+" + phone_number[3:]
        elif phone_number.startswith("+0"):
            phone_number = "+" + phone_number[2:]
        elif phone_number.startswith("00"):
            phone_number = "+" + phone_number[2:]
        elif phone_number.startswith("0"):
            phone_number = "+49" + phone_number[1:]
        elif not phone_number.startswith("+"):
            phone_number = "+" + phone_number
        k = min(3, len(phone_number) - 11)
        if k > 0:
            phone_number = phone_number[:3] + " " + phone_number[3:3+k] + " " + phone_number[3+k:]
    return phone_number

parser = JsonOutputParser()

device = "cuda"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

all_results = {"recordings": []}

root = "data/recordings"
for filename in tqdm(os.listdir(root), desc="Processing Recordings"):
    if filename.endswith(".wav"):
        filepath = os.path.join(root, filename)
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": filepath,
                    },
                    {"type": "text", "text": "Extract first_name, last_name, email and phone_number from the above audio. Give them in a JSON format. The audio is in the german language. The names might not be in german so try to extract them as they are."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
        )
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        raw_result = decoded_outputs[0]
        result = parser.invoke(raw_result)
        result["phone_number"] = normalize_phone_number(result["phone_number"])
        recording = {
            "id": filename.split(".")[0],
            "file": filename,
            "expected": result
            
        }
        all_results["recordings"].append(recording)



with open("results3.json", "w") as f:
    json.dump(all_results, f, indent=4)