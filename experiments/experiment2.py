from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from collections import Counter
import os
import json
import gc
from tqdm import tqdm

MODEL_NAME = "gpt-5.4-mini"
NUM_VOTES = 5
KEYS = ["first_name", "last_name", "email", "phone_number"]


# ============================================================
# LAYER 2 — Country / origin detection
# ============================================================
COUNTRY_SYSTEM = """You determine the LINGUISTIC ORIGIN of a phone caller's name from a German speech-to-text transcript. The goal is to identify which spelling conventions / orthography to apply to the caller's NAME — NOT to guess where they currently live.

Use these signals, in order of strength:
1. Email country-code TLD (.fr, .br, .se, .es, .it, .nl, .uk, .pt, .jp, etc.) and full domain (uol.com.br, free.fr, gmx.de, gmail.com, ...).
2. Cultural pattern of the NAME itself (given name + family name).
3. Phone country code (WEAKEST — many callers use a German number even if not German).

HARD RULES (read before answering):
- The country you return determines what orthography to apply to the name. Pick the country whose standard orthography matches the name's heritage.
- NEVER return "United States" / "USA" / language_code "en" as a default for ambiguous-or-Hispanic-or-foreign names. The USA gives no orthographic guidance because its names span every tradition. If you would otherwise pick USA, pick the country of the name's actual heritage instead.
- Map each name's heritage to a single linguistic-origin country:
    • Hispanic / Spanish-language name pattern → country = "Spain", language_code = "es" (regardless of where the caller lives — Mexico, Argentina, etc. all use the same Spanish orthography).
    • French name pattern → "France", "fr".
    • Portuguese-language name pattern: with email TLD .br → "Brazil", "pt-br"; otherwise → "Portugal", "pt".
    • Scandinavian name pattern → "Sweden", "sv" (or Norway/Denmark/Iceland with corresponding code if the pattern is unmistakeably one of those).
    • Italian name pattern → "Italy", "it".
    • Japanese name pattern → "Japan", "ja".
    • Polish name pattern → "Poland", "pl".
    • Dutch / Flemish pattern → "Netherlands", "nl".
    • Clearly German name with no foreign signal → "Germany", "de".
    • Genuinely ambiguous English name (Smith, Brown, Johnson) with no foreign signal → "United Kingdom", "en" — NOT "United States".
- Trust signal 1 (email TLD/domain) most. Trust signal 2 (name heritage) when the email is a generic global domain like gmail.com / hotmail.com / outlook.com / yahoo.com.
- Country of residence is irrelevant. Linguistic origin of the NAME is the only thing that matters.

Output JSON exactly in this shape (no extra keys, no surrounding text):
{{"country": "<country name in English>", "language_code": "<lower-case code>", "reasoning": "<one short sentence>"}}"""

COUNTRY_HUMAN = """Transcription:
<transcription>
{transcription}
</transcription>

Output the JSON."""


# ============================================================
# LAYER 3 — Self-consistent extraction (country given as input)
# ============================================================
EXTRACT_SYSTEM = """You extract caller info from a phone-call transcription. The caller's country has already been determined and is provided to you as input. Use it as the authoritative signal for spelling and orthography.

Output JSON with these keys, "thinking" first:
<keys>
{keys}
</keys>
"thinking" must be present. If a value is not in the transcription, use "". Do not invent values.

CONTEXT
- The transcript was produced by a German speech-to-text system; it uses German orthography regardless of the caller's actual country.
- Use the STANDARD spelling of the caller's NAME in the GIVEN country's language. Never default to the German spelling for a non-German caller.

NAMES — origin-correct spelling and DIACRITICS
- Diacritics are mandatory whenever the country's standard spelling has them. The transcript and the email's local-part are ASCII and cannot encode diacritics, so their absence is NOT evidence the name lacks diacritics.
- Apply ALL of these patterns:
    • Spanish surnames ending in -ez → acute accent on the vowel before -ez (-áez / -éez / -íez / -óez / -úez).
    • Spanish surnames whose stress falls on a non-default syllable (e.g., the sequence "-ía-" with stressed í, or final stressed vowel) → mark with acute accent. Common short Spanish surnames whose final two letters are "ia" with stress on the i carry an accent on the í.
    • Spanish/Portuguese names with ñ, ã, õ keep those characters.
    • Portuguese/Brazilian: tildes (ã, õ), acutes (á, é, í, ó, ú), ç where standard.
    • French: é, è, ê, à, ç where standard.
    • Scandinavian: å, ä, ö, ø, æ where standard.
    • German: ä, ö, ü, ß where standard.
- Output the form a native of the given country would write on a passport/ID.

NAME <-> EMAIL CONSISTENCY (token-level — modify the email ONLY when its alphabetic substring is a near-miss of a spoken name)

Step A: split the email's local-part by ".", "-", "_" into tokens.

Step B: for each token, identify a "candidate string" — a single contiguous run of alphabetic letters within the token that the edit-distance test will apply to. The rules:
  - If the token is purely alphabetic of length ≥ 2, the candidate string is the whole token.
  - If the token is purely digits, OR is a single alphabetic letter (length 1), there is no candidate string — keep the token unchanged.
  - If the token is mixed (contains both letters and digits), find the LONGEST contiguous alphabetic run inside it. If that run has length ≥ 2, the candidate string is that run; the digits and any other letters are NOT part of the candidate string and will be preserved in place. If the longest alphabetic run has length < 2, there is no candidate string — keep the token unchanged.

Step C: for the candidate string (lowercased, ASCII), compute the minimum edit distance to (a) the spoken first name (lowercased, ASCII) and (b) the spoken last name (lowercased, ASCII).
  - If min_distance == 0 → already correct, leave the token unchanged.
  - If min_distance ≤ 2, OR ≤ 1 when the candidate string has length ≤ 5 → this is a NAME-LIKE substring. Replace the candidate string inside the token with the standard country spelling of the matching name (lowercase, ASCII, no diacritics). Every other character of the token (digits, other letters not in the candidate string) is preserved in its original position.
  - If min_distance > 2 → does NOT resemble the spoken name. Leave the entire token unchanged.

Step D: for the NAME fields (first_name, last_name), use the standard country spelling WITH diacritics. The email's local-part remains ASCII (no diacritics) regardless.

Example mental model (no specific dataset content): for a token that starts with letters and ends in digits, treat the alphabetic prefix as the candidate string and leave the digit suffix alone. For a token that starts with a single letter followed by digits, the alphabetic prefix is too short to be a name, so leave the whole token alone.

EMAIL — other rules
- Lowercase output.
- If the local-part contains a sequence of single letters joined by hyphens (caller spelled letter-by-letter), concatenate those letters and drop the hyphens between them. A hyphen between multi-letter tokens is real — keep it.
- Separators between multi-letter name tokens (".", "-", "_") are preserved exactly as transcribed. In German speech "Unterstrich" (underscore), "Bindestrich" (hyphen), "Punkt" (dot) are distinct words; do not normalize one to another.
- If the transcript has NO "@" but contains a name-shaped string ending in a recognised TLD (.de, .com, .fr, .es, .it, .nl, .uk, .br, .se, .no, .pt, .net, .org, .eu, .ch, .at), the "@" was dropped by the ASR. Reconstruct: the domain is the rightmost two dot-segments (or three for known multi-level TLDs like .co.uk, .com.br); insert "@" before that domain. Never output "" if an address was clearly spoken.

PHONE NUMBER

Normalize the dialing prefix BEFORE applying any other rule:
- "+CC..." → already international, keep.
- "00CC..." (no "+") → replace "00" with "+".
- "0CC..." (no "+", where CC is a valid country code such as 49) → drop the leading "0", prepend "+".
- "0NNN..." (no "+", where NNN is a national-format number such as German mobile prefixes 0151/0152/0157/0159/0160/0162/0163/0170/0171/0172/0173/0174/0175/0176/0177/0178/0179, or German landlines 030/040/069/089/...) → drop the leading "0" and prepend the country code that matches the caller's country (default "+49" for German national-format numbers).

After prefix normalization the number is "+CC" + digits_after_cc. Apply the stutter rule:
    if len(digits_after_cc) >= 12 and digits_after_cc[-1] == digits_after_cc[-2]:
        digits_after_cc = digits_after_cc[:-1]    # drop final stutter digit

Otherwise preserve every transcribed digit. Final phone field = "+" + country_code + digits_after_cc.

VERIFY before outputting (each item must be answered with the literal result, not "no correction needed")
    V1. Email contains exactly one "@"? If 0, reconstruct via the no-@ rule. Email is never "" if an address was spoken.
    V2. For each ALPHABETIC length-≥2 token in the email's local-part: edit distance to spoken first or last name ≤ 2 (≤ 1 for length-≤5 tokens)? If YES and not already standard, replace with the standard country spelling. If > 2, LEAVE UNCHANGED.
    V3. NAME uses all standard diacritics for the country? If the country's standard spelling carries an accent/tilde/cedilla and your NAME doesn't, ADD it now.
    V4. digits_after_cc length ≥ 12 AND last char == second-to-last char? If yes, drop the final digit.
"""

EXTRACT_HUMAN = """Caller country: {country}
Caller language code: {language_code}

Transcription:
<transcription>
{transcription}
</transcription>

Give the JSON dictionary."""


# ============================================================
# LAYER 4 — Selection from candidates
# ============================================================
SELECT_SYSTEM = """You are given:
- The caller's country (from an earlier layer; may be wrong — see COUNTRY-OVERRIDE).
- The original transcription (for context only).
- A list of CANDIDATE values per field, each with a count = how often that value appeared across multiple extraction runs.

The candidates were produced by an extraction step that ALREADY applied:
- name<->email letter consistency (token-level, edit-distance-based)
- origin-language name spelling
- phone prefix normalization

Your job: pick the SINGLE most probable value per field.

PRIME DIRECTIVE
If all candidates for a field agree on the same value, OUTPUT THAT VALUE VERBATIM. Do not second-guess unanimous candidates by inspecting the original transcription. The transcription is shown only for context when candidates disagree or when a country override is needed.

When candidates differ, apply these rules in priority order:

1. COUNTRY-OVERRIDE: Sanity-check the country. If it looks wrong (e.g., "United States" / "en" for an unmistakeably non-English name), mentally use the correct linguistic-origin country for the rules below. Do not trust "United States" for a clearly non-English name.

2. NAMES — diacritics: Among candidates that differ ONLY in diacritics (e.g., one is the accented form, another is the unaccented form of the same letters), pick the candidate WITH the diacritics if the (corrected) country's standard orthography uses them — even if it is a minority candidate. If NO candidate has the diacritics AND the country's orthography clearly requires them for that name, you may output the diacritic form yourself (this is the ONLY case where you may produce a value not in the candidate list, and only for the NAME fields, never for the email or phone).

3. NAMES — origin spelling: When candidates differ in their letter spelling (not just diacritics) and the (corrected) country is non-German, pick the form a native of that country would write.

4. PHONE — prefix sanity: Discard candidates starting with "+0", "+04", "0", or "00". Pick a candidate that begins with a valid country code ("+49", "+33", "+34", "+44", "+1", "+55", "+46", "+39", "+81", "+31", "+39", ...).

5. PHONE — length: Prefer the candidate whose digit count matches a plausible national format for the country.

6. EMAIL — choose from candidates only:
   a) Never modify the letters of an email yourself; pick from the candidates verbatim.
   b) NAME-TOKEN AGREEMENT WITH THE CHOSEN NAME (overrides majority): take the chosen NAME (first + last, lowercased, diacritic-stripped). For each email candidate, look at the alphabetic letter-runs inside the local-part (ignoring any digits, dots, hyphens, underscores). If one candidate's alphabetic letter-runs exactly match the chosen NAME's letters and another candidate's alphabetic letter-runs differ from the chosen NAME (extra/missing/substituted letter), PICK THE MATCHING CANDIDATE. This rule overrides frequency: a 1-of-5 minority candidate whose alphabetic letters match the chosen NAME beats a 4-of-5 majority candidate whose alphabetic letters do not match.
   c) HANDLE ELEMENTS (digit suffixes/prefixes, arbitrary non-name segments): when candidates differ on whether they retain non-alphabetic handle elements (digits, year suffixes, arbitrary segments) from the transcribed local-part, prefer the candidate that retains them.
   d) Do NOT revert a candidate's name-token spelling to whatever appeared in the transcript's email — the candidate is more reliable than the transcript for name-letter spelling.

7. Most-frequent wins by default. Override only when one of the rules above clearly favours another value.

Output JSON with a single string per key (NOT a list). Do not add extra keys."""

SELECT_HUMAN = """Caller country: {country}
Caller language code: {language_code}

Original transcription:
<transcription>
{transcription}
</transcription>

Candidates (each entry is value + count, ordered most-frequent first):
{candidates}

Output the final JSON dict, one string per key."""


# ============================================================
# LAYER 5 — Verification (NAME ↔ email reconciliation)
# ============================================================
VERIFY_SYSTEM = """You are the final verification layer. You receive a draft answer (first_name, last_name, email, phone_number) from earlier layers and the caller's country. Your job is to ensure the NAME fields and the email's name-tokens are consistent: when they are CLOSE but not equal, they MUST be reconciled to the country-standard spelling.

PROCEDURE — apply mechanically, no shortcuts:

1. Take the draft first_name (call it F) and last_name (call it L). Compute lowercased ASCII versions f = lowercase(strip_diacritics(F)) and l = lowercase(strip_diacritics(L)).

2. Take the draft email's local-part (everything before "@"). Split it by ".", "-", "_" into tokens. For each token, find the LONGEST contiguous run of alphabetic letters of length ≥ 2; collect these runs as the "email name-tokens". (Single-letter alphabetic runs and pure-digit substrings are NOT email name-tokens.)

3. For EACH email name-token T (lowercased):
   a) Compute character-level edit distance d_first = edit(T, f) and d_last = edit(T, l). Take min_d = min(d_first, d_last).
   b) If min_d == 0: T already agrees with the corresponding name; do nothing.
   c) If 0 < min_d <= 2 (CLOSE — they are clearly the same name spelled slightly differently; reconcile MANDATORY):
        - Identify which spelling — T or the matching name component — is the COUNTRY-STANDARD form (use the given country's orthography conventions for that name).
        - Case (i): the NAME field is the country-standard form and T is a near-miss (extra/missing/substituted letter). Action: REPLACE the alphabetic run inside the email's local-part with the name field's letters (lowercased ASCII). Preserve every other character (digits, separators, other letters) in their original positions. The NAME field is left as is.
        - Case (ii): T is the country-standard form and the NAME field is in a different language's spelling. Action: REPLACE the corresponding NAME field with T's letters (capitalised normally), then add origin-standard diacritics on the NAME (the email stays ASCII).
        - Case (iii): both T and the NAME field are valid spellings in different traditions (e.g., one is the German form, one is the Brazilian form). Action: pick the country-standard form and apply it to BOTH (modifying whichever side does not already use it).
   d) If min_d > 2: T is not a near-miss of any name field — it's a chosen handle, prefix, or unrelated segment. Leave it alone.

4. After all reconciliations, the email's local-part letters and the NAME field letters MUST agree (after lowercasing and stripping diacritics from the NAME). Verify and re-apply step 3 if they don't.

5. Phone number: pass through unchanged. Do not modify.

Output JSON with these keys: first_name, last_name, email, phone_number. Single string per key. No extra keys, no surrounding text."""

VERIFY_HUMAN = """Country: {country}
Language code: {language_code}

Original transcription (for context only — do NOT pull email letters from here; use the draft):
<transcription>
{transcription}
</transcription>

Draft answer:
first_name: {first_name}
last_name: {last_name}
email: {email}
phone_number: {phone_number}

Apply the verification procedure and output the fixed JSON dict."""


# ============================================================
# Build chains
# ============================================================
country_prompt = ChatPromptTemplate.from_messages([
    ("system", COUNTRY_SYSTEM),
    ("human", COUNTRY_HUMAN),
])
country_llm = ChatOpenAI(model=MODEL_NAME, temperature=0, max_tokens=400, base_url="https://us.api.openai.com/v1")
country_chain = country_prompt | country_llm | JsonOutputParser()

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", EXTRACT_SYSTEM),
    ("human", EXTRACT_HUMAN),
])
extract_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7, max_tokens=1200, base_url="https://us.api.openai.com/v1")
extract_chain = extract_prompt | extract_llm | JsonOutputParser()

select_prompt = ChatPromptTemplate.from_messages([
    ("system", SELECT_SYSTEM),
    ("human", SELECT_HUMAN),
])
select_llm = ChatOpenAI(model=MODEL_NAME, temperature=0, max_tokens=600, base_url="https://us.api.openai.com/v1")
select_chain = select_prompt | select_llm | JsonOutputParser()

verify_prompt = ChatPromptTemplate.from_messages([
    ("system", VERIFY_SYSTEM),
    ("human", VERIFY_HUMAN),
])
verify_llm = ChatOpenAI(model=MODEL_NAME, temperature=0, max_tokens=600, base_url="https://us.api.openai.com/v1")
verify_chain = verify_prompt | verify_llm | JsonOutputParser()


def aggregate_candidates(runs, keys):
    """Aggregate per-key candidate values across multiple runs into lists with counts."""
    aggregated = {}
    for k in keys:
        values = [str(r.get(k, "")) for r in runs if isinstance(r, dict)]
        counter = Counter(values)
        aggregated[k] = [{"value": v, "count": c} for v, c in counter.most_common()]
    return aggregated


# ============================================================
# Voxtral setup (Layer 1)
# ============================================================
device = "cuda"
repo_id = "mistralai/Voxtral-Mini-3B-2507"
processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(
    repo_id,
    dtype=torch.bfloat16,
    device_map=device
)
model.eval()


all_results = {"recordings": []}

recordings = sorted(f for f in os.listdir("data/recordings") if f.lower().endswith(".wav"))
for filename in tqdm(recordings):
    path = os.path.join("data/recordings", filename)

    # Layer 1: transcribe
    inputs = processor.apply_transcription_request(language="de", audio=path, model_id=repo_id)
    inputs = inputs.to(device, dtype=torch.bfloat16)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=500)
    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    transcription = " ".join(d.strip() for d in decoded)
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # Layer 2: country/origin
    try:
        country_info = country_chain.invoke({"transcription": transcription})
    except Exception:
        country_info = {"country": "", "language_code": "", "reasoning": ""}
    country = country_info.get("country", "") or ""
    language_code = country_info.get("language_code", "") or ""

    # Layer 3: self-consistent extraction (5 parallel runs, aggregate as candidate lists)
    extract_input = {
        "keys": KEYS,
        "country": country,
        "language_code": language_code,
        "transcription": transcription,
    }
    try:
        runs = extract_chain.batch([extract_input] * NUM_VOTES)
    except Exception:
        runs = []
        for _ in range(NUM_VOTES):
            try:
                runs.append(extract_chain.invoke(extract_input))
            except Exception:
                pass
    if not runs:
        continue
    candidates = aggregate_candidates(runs, KEYS)

    # Layer 4: selection
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
    try:
        final = select_chain.invoke({
            "country": country,
            "language_code": language_code,
            "transcription": transcription,
            "candidates": candidates_str,
        })
    except Exception:
        # Fallback: take the most-frequent candidate per key
        final = {k: (candidates[k][0]["value"] if candidates[k] else "") for k in KEYS}

    # Layer 5: verification (NAME ↔ email reconciliation when close-but-not-equal)
    pre_verify = dict(final)
    try:
        verified = verify_chain.invoke({
            "country": country,
            "language_code": language_code,
            "transcription": transcription,
            "first_name": final.get("first_name", ""),
            "last_name": final.get("last_name", ""),
            "email": final.get("email", ""),
            "phone_number": final.get("phone_number", ""),
        })
        if isinstance(verified, dict):
            for k in KEYS:
                if k in verified and verified[k]:
                    final[k] = verified[k]
    except Exception:
        pass

    # Phone post-processing
    phone_number = final.get("phone_number", "")
    if phone_number:
        phone_number = phone_number.replace(" ", "")
        # Prefix normalization
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
        # Stutter rule: only trigger on numbers that are clearly one digit too long
        # (>=14 digits after "+"; German mobiles are 13). Also require trailing repeat.
        digits_only = phone_number[1:]
        if len(digits_only) >= 14 and digits_only[-1] == digits_only[-2]:
            phone_number = "+" + digits_only[:-1]
        # Format
        k = min(3, len(phone_number) - 11)
        if k > 0:
            phone_number = phone_number[:3] + " " + phone_number[3:3+k] + " " + phone_number[3+k:]
        final["phone_number"] = phone_number

    all_results["recordings"].append({
        "id": filename.split(".")[0],
        "file": filename,
        "expected": final,
        "transcription": transcription,
        "country_info": country_info,
        "candidates": candidates,
        "pre_verify": pre_verify,
        "thinking_runs": [r.get("thinking", "") for r in runs if isinstance(r, dict)],
    })

with open("results2.json", "w") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
