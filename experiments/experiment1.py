from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# import json parser from langchain
from langchain_core.output_parsers import JsonOutputParser
from collections import Counter
import os
import json
from tqdm import tqdm

MODEL_NAME = "gpt-5.4-mini"

SYSTEM_PROMPT = """You extract caller info from a phone-call transcription.

Output JSON with these keys, in order, with "thinking" FIRST:
<keys>
{keys}
</keys>
"thinking" is for your reasoning; it must be present. If a value is not in the transcription, use "". Do not invent values.

CONTEXT
- The transcript was produced by a German speech-to-text system. The transcript uses German orthography and German phonetics, regardless of the caller's actual origin. Trust the transcript's WORDS, distrust its SPELLING of non-German content.
- The caller may be of any nationality. Infer caller origin from, in order of signal strength:
    1. Email country-code TLD (.fr, .br, .se, .es, .it, .nl, .uk, .pt, etc.) and full domain (uol.com.br, free.fr, gmx.de, ...).
    2. Cultural pattern of the name itself.
    3. Phone country code (WEAKEST — German number does not mean German caller).

NAMES — origin-correct spelling and DIACRITICS
- Use the STANDARD spelling of the name in the inferred origin's language. Never default to German spelling for a non-German caller.
- Diacritics are mandatory whenever the inferred origin's standard spelling carries them. The transcript and the email are ASCII and cannot encode diacritics, so their absence is NOT evidence that the name lacks diacritics.
- Apply ALL of these patterns:
    • Spanish surnames ending in -ez → acute accent on the vowel before -ez (gives -áez / -éez / -íez / -óez / -úez patterns).
    • Spanish surnames whose stress falls on a non-default syllable (e.g., contain the letter sequence "-ía-" with the í stressed, or end in a stressed vowel) → mark that vowel with an acute accent (á, é, í, ó, ú). In particular, common short Spanish surnames whose final two letters are "ia" with stress on the i carry an accent on that í.
    • Spanish/Portuguese names with ñ, ã, õ keep those characters.
    • Portuguese/Brazilian: tildes (ã, õ), acutes (á, é, í, ó, ú), ç where standard.
    • French: é, è, ê, à, ç where standard.
    • Scandinavian: å, ä, ö, ø, æ where standard.
    • German: ä, ö, ü, ß where standard.
- If you can recall how a name is overwhelmingly written on a passport/ID/legal document in its native country, output THAT form.

NAME ↔ EMAIL CONSISTENCY (token-level — be CONSERVATIVE about modifying the email)

Split the email's local-part by ".", "-", "_" into tokens. Classify each token:
  - DIGITS-ONLY (e.g., "1983", "47") → keep unchanged.
  - MIXED (alphanumeric, contains at least one digit or non-letter, e.g. "h47", "user2") → keep unchanged. The user chose these; do not touch them.
  - SINGLE LETTER (e.g., "k", "m") → keep unchanged (it is an initial).
  - ALPHABETIC of length ≥ 2 → consider for name normalization (next paragraph).

For each ALPHABETIC token of length ≥ 2, compute its character-level edit distance to (a) the spoken first name (lowercased, ASCII) and (b) the spoken last name (lowercased, ASCII). Take the minimum.
  - If min_distance == 0 → already correct, leave it alone (but the NAME field should still apply origin diacritics; see DIACRITICS).
  - If min_distance ≤ 2 (or ≤ 1 for tokens of length ≤ 5) → this is a NAME-LIKE token, almost certainly an ASR mistake of the name. Replace it with the standard origin spelling of the matching name (lowercased, ASCII, no diacritics).
  - If min_distance > 2 → the token does NOT resemble the spoken name. LEAVE IT UNCHANGED. It may be a chosen handle, prefix, or unrelated identifier (do not strip it, do not replace it).

For the NAME fields (first_name, last_name): use the standard origin spelling WITH diacritics.

Direction of correction (when the spoken NAME and a NAME-LIKE token disagree):
  - If spoken NAME is a recognised name in the inferred origin and the email token is a near-miss → fix the EMAIL token to match the standard NAME spelling (ASCII).
  - If the email token is a recognised name in the inferred origin's language but the spoken NAME is in a different language's spelling → fix the NAME field to match the email token (then apply origin diacritics).
  - If both look like recognised names in different traditions → use the form standard in the inferred origin for BOTH.

The phrase "names look too different" maps onto min_distance > 2 above: in that case do NOT modify the email. Preserve all unrecognised tokens exactly as transcribed.

EMAIL — other rules
- Lowercase output.
- If the local-part contains a sequence of single letters joined by hyphens (caller spelled the name letter-by-letter), concatenate those letters and drop the hyphens between them. A hyphen between multi-letter tokens is a real hyphen — keep it.
- Separators between multi-letter name tokens (".", "-", "_") are preserved exactly as transcribed. In German speech, "Unterstrich" (underscore), "Bindestrich" (hyphen), and "Punkt" (dot) are distinct; do not normalize one to another.
- If the transcript has NO "@" but contains a name-shaped string ending in a recognised TLD (.de, .com, .fr, .es, .it, .nl, .uk, .br, .se, .no, .pt, .net, .org, .eu, .ch, .at), the "@" was dropped by the ASR. Reconstruct: the domain is the rightmost two dot-segments (or three for known multi-level TLDs like .co.uk, .com.br); insert "@" before that domain. Never output "" when an address was clearly spoken.

PHONE NUMBER

Normalize the dialing prefix BEFORE applying any other phone rule:
- If the transcribed number begins with "+" → already international, keep as-is.
- If it begins with "00" (and no "+") → replace the "00" with "+" (German/European international call prefix).
- If it begins with a single "0" (and no "+") AND the next 1–3 digits are a valid country code (e.g., "049…", "044…") → drop that leading "0" and prepend "+". This case occurs when the speaker said "00 49 …" but the ASR ate one zero, leaving "0 49 …".
- If it begins with a single "0" (and no "+") AND the next digits are a national-format number (e.g., German mobile prefixes 0151/0152/0157/0159/0160/0162/0163/0170/0171/0172/0173/0174/0175/0176/0177/0178/0179, German landline prefixes 030/040/069/089/…) → drop the leading "0" and prepend the country code with "+" (default "+49" for a German national-format number; use the country code that matches the inferred origin/area code).

After prefix normalization, the number is "+CC" + digits_after_cc.

Apply the stutter rule:
    if len(digits_after_cc) >= 12 and digits_after_cc[-1] == digits_after_cc[-2]:
        digits_after_cc = digits_after_cc[:-1]    # drop final stutter digit

Otherwise preserve every transcribed digit. Do NOT shorten just because the number looks long. The final phone field is "+" + country_code + digits_after_cc.

VERIFY before outputting (each item must be answered with the literal result, not "no correction needed")
    V1. Does email contain exactly one "@"? If 0, reconstruct via the no-@ rule above. Email is never "" if an address was spoken.
    V2. For each ALPHABETIC length-≥2 token in the email's local-part: is its edit distance to spoken first or last name ≤ 2 (≤ 1 for length-≤5 tokens)? If YES and it does not already match the standard NAME spelling, replace that token with the standard NAME spelling (ASCII). If a token's distance is > 2, LEAVE IT UNCHANGED (do not strip it). Tokens with digits or single letters are always left unchanged.
    V3. Does NAME use the diacritics standard in the inferred origin? If the origin's standard spelling carries an accent/tilde/cedilla and your NAME doesn't, ADD it now. The email stays ASCII.
    V4. digits_after_cc check: len >= 12 and last == second-to-last? If yes and you haven't dropped the final digit yet, drop it now and update the phone field.
"""

HUMAN_PROMPT = """The transcription of the phone call is:
<transcription>
{transcription}
</transcription>

Give the JSON dictionary with the extracted information.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, max_tokens=1200, base_url="https://us.api.openai.com/v1")

chain = prompt | llm | JsonOutputParser()

NUM_VOTES = 3

def majority_vote(results, keys):
    voted = {}
    for k in keys:
        values = [str(r.get(k, "")) for r in results if isinstance(r, dict)]
        if not values:
            voted[k] = ""
            continue
        voted[k] = Counter(values).most_common(1)[0][0]
    return voted

device = "cuda"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

KEYS = ["first_name", "last_name", "email", "phone_number"]

all_results = {"recordings":[]}

recordings = sorted(f for f in os.listdir("data/recordings") if f.lower().endswith(".wav"))
for filename in tqdm(recordings):
    path = os.path.join("data/recordings", filename)

    inputs = processor.apply_transcription_request(language="en", audio=path, model_id=repo_id)
    inputs = inputs.to(device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    decoded_outputs = [output.strip() for output in decoded_outputs]
    transcription = " ".join(decoded_outputs)
    chain_input = {"keys": KEYS, "transcription": transcription}
    try:
        batch_results = chain.batch([chain_input] * NUM_VOTES)
    except Exception:
        batch_results = []
        for _ in range(NUM_VOTES):
            try:
                batch_results.append(chain.invoke(chain_input))
            except Exception:
                pass
    if not batch_results:
        continue
    result = {"thinking": [r.get("thinking", "") for r in batch_results if isinstance(r, dict)]}
    result.update(majority_vote(batch_results, KEYS))
    phone_number = result.get("phone_number", "")
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
        result["phone_number"] = phone_number
            
    all_results["recordings"].append(
        {
            "id": filename.split(".")[0],
            "file": filename,
            "expected":result,
            "transcription": transcription
        }
    )
    
with open("results.json", "w") as f:
    json.dump(all_results, f, indent=4)