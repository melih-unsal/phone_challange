"""Prompts for every LLM / Voxtral-instruction-following layer.

Prompts are assembled here from the rule strings in `config.py` so that
the rules themselves stay in one place and prompts only describe how to
present them.
"""

from src.config import (
    COUNTRY_RULES,
    EMAIL_RULES,
    FIRST_NAME_RULES,
    LAST_NAME_RULES,
    NAME_EMAIL_CONSISTENCY,
    COUNTRY_CONSISTENCY,
    PHONE_RULES,
    TRANSCRIPT_CONTEXT,
)


# ---------------------------------------------------------------------------
# Layer 2 - country / origin detection
# ---------------------------------------------------------------------------
COUNTRY_SYSTEM = f"""{COUNTRY_RULES}

Output JSON exactly in this shape (no extra keys, no surrounding text):
{{{{"country": "<country name in English>", "language_code": "<lower-case code>", "reasoning": "<one short sentence>"}}}}"""

COUNTRY_HUMAN = """Transcription:
<transcription>
{transcription}
</transcription>

Output the JSON."""


# ---------------------------------------------------------------------------
# Layer 3 - Candidate 1: self-consistent LLM extraction over the transcript
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = f"""You extract caller info from a phone-call transcription. The caller's country has already been determined and is provided to you as input. Use it as the authoritative signal for spelling and orthography.

Output JSON with these keys, "thinking" first:
<keys>
{{keys}}
</keys>
"thinking" must be present. If a value is not in the transcription, use "". Do not invent values.

CONTEXT
- {TRANSCRIPT_CONTEXT}

{FIRST_NAME_RULES}

{LAST_NAME_RULES}

{EMAIL_RULES}

{PHONE_RULES}

{NAME_EMAIL_CONSISTENCY}

{COUNTRY_CONSISTENCY}

VERIFY before outputting (each item must be answered with the literal result, not "no correction needed")
    V1. Email contains exactly one "@"? If 0, reconstruct via the no-@ rule. Email is never "" if an address was spoken.
    V2. For each ALPHABETIC length-≥2 token in the email's local-part: edit distance to spoken first or last name ≤ 2 (≤ 1 for length-≤5 tokens)? If YES and not already standard, replace with the standard country spelling. If > 2, LEAVE UNCHANGED.
    V3. NAME uses all standard diacritics for the country? If the country's standard spelling carries an accent/tilde/cedilla and your NAME doesn't, ADD it now.
    V4. digits_after_cc length ≥ 12 AND last char == second-to-last char? If yes, drop the final digit."""

EXTRACT_HUMAN = """Caller country: {country}
Caller language code: {language_code}

Transcription:
<transcription>
{transcription}
</transcription>

Give the JSON dictionary."""


# ---------------------------------------------------------------------------
# Layer 4 - Candidate 2: direct Voxtral instruction-following on the audio
# ---------------------------------------------------------------------------
# Voxtral runs the prompt directly on the audio. Country info is unknown at
# this stage (we deliberately do not pass it through - this candidate is
# meant to be an INDEPENDENT reading of the audio that the reconciler can
# cross-check against candidate 1).
VOXTRAL_DIRECT_PROMPT = f"""Extract first_name, last_name, email and phone_number from the above audio. Return ONLY a JSON object with exactly those four keys, no extra keys, no surrounding text.

The audio is in the German language. The caller may not be German - names might be Spanish, French, Portuguese, Italian, Scandinavian, Dutch, Polish, Japanese, etc. Extract the name as the caller would write it on a passport / ID, with native diacritics if the name's origin uses them.

{EMAIL_RULES}

{PHONE_RULES}

Output JSON only:
{{"first_name": "...", "last_name": "...", "email": "...", "phone_number": "..."}}"""


# ---------------------------------------------------------------------------
# Layer 4.5 - Targeted re-listening prompts
# ---------------------------------------------------------------------------
# The audio LLM gets two laser-focused prompts that hit only one field at a time.
# Email and phone are the fields where ASR-based extraction degrades the most
# (spelled letter-by-letter / dictated digit-by-digit), and where a focused
# audio re-listen pays off the most.

EMAIL_TARGETED_PROMPT = f"""Listen to the audio and extract ONLY the caller's email address. Output a single JSON object with one key, "email", and nothing else.

The audio is in German. The caller is likely spelling the address letter-by-letter, using German separator words:
- "Punkt" = "."
- "Bindestrich" = "-"
- "Unterstrich" = "_"
- "ät" / "at" = "@"

{EMAIL_RULES}

Output JSON only:
{{"email": "..."}}"""


PHONE_TARGETED_PROMPT = f"""Listen to the audio and extract ONLY the caller's phone number. Output a single JSON object with one key, "phone_number", and nothing else.

The audio is in German. The caller is likely dictating digits, sometimes grouped, possibly preceded by an international prefix ("plus vier neun ..." for "+49 ..."), or a national format starting with 0.

{PHONE_RULES}

Output JSON only:
{{"phone_number": "..."}}"""


# ---------------------------------------------------------------------------
# Layer 5 - Reconciler
# ---------------------------------------------------------------------------
# Receives: transcript, country info, candidate 1 (LLM self-consistency over
# transcript), candidate 2 (Voxtral direct from audio). Produces a reasoning
# trace and the final JSON.
RECONCILE_SYSTEM = f"""You are the final reconciler. You receive:
- one or more TRANSCRIPTS, each from a different STT model (e.g. Voxtral and ElevenLabs Scribe);
- the caller's country (already determined upstream);
- CANDIDATES_1: a LIST of answers, one per STT model. Each entry was produced by self-consistent LLM extraction over THAT transcript (majority-voted across multiple runs).
- CANDIDATE 2: the answer produced by an audio LLM with one whole-record prompt (instruction-following on the raw audio).
- TARGETED RE-LISTEN: optional. Single-field re-extractions where the audio LLM was prompted to focus on ONLY the email or ONLY the phone, on a cropped audio window. When present, treat these as the strongest single-field signal - they were produced by re-listening with the field's own rules in front of the model.

The candidates are INDEPENDENT readings of the same call. Your job is to combine them into the single most-probable JSON answer, with a short reasoning trace.

CONTEXT
- {TRANSCRIPT_CONTEXT}
- Each entry in CANDIDATES_1 sees only its own transcript; STT models make different errors, so disagreement between CANDIDATES_1 entries is informative.
- Candidate 2 listens to the actual audio, so it tends to hear letters, digits and proper names more directly, but it does not know the country and may default to German spelling for non-German names.
- Targeted re-listens listen to the audio with all of their attention on a single field - strong evidence for that field, no signal at all for other fields.

DECISION POLICY (apply per-key)

1. UNANIMOUS WINS. If all CANDIDATES_1 entries and CANDIDATE 2 agree on a key (treating empty values as abstentions), output that value verbatim. Do NOT second-guess unanimous candidates.

2. TARGETED RE-LISTEN PRIORITY for email and phone. If the targeted re-listen value is present and structurally valid (email has "@" + TLD; phone parses to a real country prefix), prefer it over the whole-record candidates UNLESS the whole-record candidates are unanimous on a different value (then unanimous wins).

3. NAME <-> EMAIL CONSISTENCY (CRITICAL - run this BEFORE picking a name from the audio). If candidates disagree on a name field but agree on an email, AND the email's local-part has an alphabetic token within edit-distance 2 of EVERY candidate spelling, the email is the authority - use its token (capitalised) as the name. Example: cand_1.first_name="Armet", cand_2.first_name="Armed", email="ahmed.hassan@gmail.com" → first_name = "Ahmed".

4. Otherwise apply the per-key rules below.

{FIRST_NAME_RULES}

{LAST_NAME_RULES}

When the two candidates disagree on a NAME field:
- If the candidates differ ONLY in diacritics (same letters, one has accents, the other does not), and the country's standard orthography uses those diacritics, pick the diacritic form. If neither candidate has the diacritics but the country clearly requires them, you may produce the diacritic form yourself (NAME fields only - never invent letters/digits for email or phone).
- If the candidates differ in letter spelling: prefer candidate 2 (audio) for the LETTERS the caller actually said, but apply candidate 1's orthography conventions on top (i.e. add the country-standard diacritics).
- If candidate 2 is empty and candidate 1 is non-empty (or vice versa), take the non-empty one.

{EMAIL_RULES}

When the two candidates disagree on email:
- Pick the candidate verbatim - do NOT invent letters yourself.
- NAME-TOKEN AGREEMENT: take the chosen NAME (first + last, lowercased, diacritic-stripped). For each email candidate, look at the alphabetic letter-runs inside the local-part (ignoring digits and separators). If one candidate's alphabetic letter-runs match the chosen NAME's letters and the other's do not, pick the matching one. This rule overrides "stay closer to the audio".
- HANDLE ELEMENTS (digits, year suffixes, arbitrary segments): when candidates differ on whether they retain non-alphabetic handle elements, prefer the candidate that retains them.
- If only one candidate has the "@" and a plausible domain, pick that one.

{PHONE_RULES}

When the two candidates disagree on phone_number:
- Discard candidates that start with "+0", "+04", "0", or "00" - they are mis-normalised. Pick a candidate that begins with a valid country code ("+49", "+33", "+34", "+44", "+1", "+55", "+46", "+39", "+81", "+31", ...).
- Prefer the candidate whose digit count matches a plausible national format for the caller's country.
- Apply the stutter rule yourself if the chosen candidate still has a doubled trailing digit.

{NAME_EMAIL_CONSISTENCY}

{COUNTRY_CONSISTENCY}

Output JSON exactly:
{{{{"reasoning": "<short paragraph: which candidate did you pick per key and why>", "first_name": "...", "last_name": "...", "email": "...", "phone_number": "..."}}}}

Do not output any other keys. Do not wrap in markdown. Output JSON only."""

RECONCILE_HUMAN = """Caller country: {country}
Caller language code: {language_code}

Transcripts (one per STT model):
{transcripts}

CANDIDATES_1 (LLM extraction × N over each transcript, one entry per STT):
{candidates_1}

CANDIDATE 2 (audio LLM, whole-record prompt):
{candidate_2}

TARGETED RE-LISTEN (audio LLM, single-field focused prompts on cropped audio):
{targeted}

Apply the decision policy and output the JSON."""
