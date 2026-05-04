"""Configuration: model IDs, runtime knobs, and the rule strings that
get composed into the prompts in `prompts.py`.

Each key has its own block of rules. Inter-key consistency rules are
kept separate so that prompts that only need a subset of the rules
(e.g. the Voxtral direct-extraction prompt) can include exactly what
they need.
"""

# Auto-load a .env file if python-dotenv is available, so API keys
# (OPENAI_API_KEY, ELEVENLABS_API_KEY, ...) come along with config.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Models / runtime
# ---------------------------------------------------------------------------
LLM_MODEL_NAME = "gpt-5.4-mini"
LLM_BASE_URL = "https://us.api.openai.com/v1"

# Audio-model specs are model-agnostic: "<backend>:<model_id>".
#   backend = "voxtral"      -> Voxtral via HF transformers (DEFAULT).
#                               Has rare SIGSEGV in sdpa_attention_forward;
#                               mitigated by VOXTRAL_ATTN_IMPL="eager" below
#                               and recoverable via main.py's AUTO_RESUME.
#                               This is the path the 29/30 headline result
#                               was produced on, across 8 dev runs.
#   backend = "voxtral-vllm" -> Voxtral served by vLLM. More stable in
#                               theory and on torch 2.4 + CUDA 12 stacks;
#                               vLLM 0.20.x has a regression that SIGSEGVs
#                               on Voxtral model registry inspection on
#                               torch 2.11 + CUDA 13. Use this when your
#                               vLLM/torch combo is known to work.
#   backend = "elevenlabs"   -> ElevenLabs Speech-to-Text API (transcription only)
#   backend = "whisper"      -> local HuggingFace Whisper (timestamps only)
#
# TRANSCRIPTION_MODELS is a LIST. Each entry produces its own transcript and
# its own candidate_1 (Layer 3 extracts independently from each transcript).
# More diverse transcripts -> more independent readings for the reconciler
# to triangulate from. ASR errors are usually correlated within a single
# model and uncorrelated across models, so adding a second STT is high-value.
#
# On a machine without a CUDA GPU, Voxtral and Whisper backends will fail
# to build at startup; main.py logs a warning and continues with whatever
# backends did build (typically just ElevenLabs Scribe).
TRANSCRIPTION_MODELS = [
    "voxtral:mistralai/Voxtral-Mini-3B-2507",
    "elevenlabs:scribe_v2",
]

INSTRUCT_MODEL = "voxtral:mistralai/Voxtral-Mini-3B-2507"

# Optional dedicated timestamping model (Layer 1b). Used to fill in word
# timestamps for Layer 4.5 when the main TRANSCRIPTION_MODEL doesn't expose
# them (e.g. Voxtral). Set to "" to disable - Layer 4.5 then falls back to
# whole-audio prompts.
#   "whisper:openai/whisper-large-v3-turbo"   (local HF, fast - DEFAULT)
#   "whisper:openai/whisper-large-v3"         (local HF, more accurate)
#   "elevenlabs:scribe_v2"                    (API)
#   ""                                        (disabled - Layer 4.5 uses full audio)
TIMESTAMP_MODEL = "whisper:openai/whisper-large-v3-turbo"

# Language used by the transcriber. Voxtral expects ISO-639-1 ("de"),
# ElevenLabs expects ISO-639-3 ("deu"). Each backend translates as needed.
TRANSCRIPTION_LANGUAGE = "de"

# Env var names for API-keyed backends
ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"

# Optional ElevenLabs Scribe keyterms (max 1000, ≤50 chars each). The model
# is biased toward recognising these literal strings - useful when the audio
# contains specific words a generic ASR mishears. For German caller recordings,
# the separator words people say while spelling emails are good candidates.
ELEVENLABS_KEYTERMS = [
    "Punkt", "Bindestrich", "Unterstrich",
    "ät", "at",
    "gmail.com", "gmx.de", "web.de", "hotmail.com", "outlook.com", "yahoo.com",
]

# Voxtral runtime
VOXTRAL_DEVICE = "cuda"
# Voxtral attention implementation. Recent torch+CUDA combos have hit
# intermittent SDPA segfaults inside `sdpa_attention_forward` during
# generation; "eager" is slower but rock-solid. Override to "sdpa" or
# "flash_attention_2" once the upstream bug is fixed.
VOXTRAL_ATTN_IMPL = "eager"

# Whisper runtime (only matters when TIMESTAMP_MODEL = "whisper:...")
WHISPER_DEVICE = "cuda"

# Auto-resume: on startup, scan RESULTS_DIR for the most recent unfinished
# report (no `evaluation` block, fewer recordings than the dataset) and
# pick up where it left off. Set False to always start a fresh run.
AUTO_RESUME = True

KEYS = ["first_name", "last_name", "email", "phone_number"]

# Self-consistency for candidate 1 (LLM extraction over the transcript)
NUM_LLM_VOTES = 5

# Self-consistency for candidate 2 (audio-instructor direct extraction)
NUM_INSTRUCT_VOTES = 3

# Self-consistency for targeted re-listen (Layer 4.5 - email + phone focused).
# Cheap to run and high-signal, so a small N is enough.
NUM_TARGETED_VOTES = 3

# Padding (seconds) added on each side of a detected email/phone segment
# before cropping the audio for the targeted re-listen. Larger pad =
# safer (won't cut off the trailing ".com" / ".fr") but gives the audio LLM
# a little more unrelated speech to ignore.
TARGETED_PAD_SECONDS = 3.0

# Master switch for Layer 4.5. Set False to skip targeted re-extraction (e.g.
# while A/B-testing the value of the layer).
USE_TARGETED_EXTRACTION = True

# phonenumbers default region for parsing local-format phone numbers.
DEFAULT_PHONE_REGION = "DE"

# Prompt-version tag stored in the report. Bump this when prompts change so
# downstream A/B comparisons can group runs by prompt version.
PROMPT_VERSION = "2026-05-03.v1"

RECORDINGS_DIR = "data/recordings"
GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Per-key rules
# ---------------------------------------------------------------------------
FIRST_NAME_RULES = """FIRST_NAME
- Use the STANDARD spelling of the first name in the caller's country/language. Never default to the German spelling for a non-German caller.
- Diacritics are mandatory whenever the country's standard spelling has them. The transcript and the email are ASCII and cannot encode diacritics, so their absence is NOT evidence the name lacks diacritics.
- Diacritic patterns by origin:
    * Spanish/Portuguese: á, é, í, ó, ú, ñ, ã, õ, ç where standard.
    * French: é, è, ê, à, ç where standard.
    * Scandinavian: å, ä, ö, ø, æ where standard.
    * German: ä, ö, ü, ß where standard.
- Output the form a native of the country would write on a passport / ID."""

LAST_NAME_RULES = """LAST_NAME
- Same origin-correct, diacritic-aware rules as FIRST_NAME.
- Additional Spanish patterns:
    * Spanish surnames ending in -ez -> acute accent on the vowel before -ez (-áez / -éez / -íez / -óez / -úez).
    * Spanish surnames whose stress falls on a non-default syllable (e.g. "-ía-" with stressed í, or final stressed vowel) carry an acute accent.
- Output the form a native of the country would write on a passport / ID."""

EMAIL_RULES = """EMAIL
- Always lowercase.
- Exactly one "@". If the transcript contains a name-shaped string ending in a recognised TLD (.de, .com, .fr, .es, .it, .nl, .uk, .br, .se, .no, .pt, .net, .org, .eu, .ch, .at) but no "@", the ASR dropped it: insert "@" before the rightmost two dot-segments (or three for known multi-level TLDs like .co.uk, .com.br). The email is never "" if an address was clearly spoken.
- Letter-by-letter spellings: a sequence of single letters joined by hyphens means the caller spelled the name; concatenate those letters and drop the hyphens between them. A hyphen between multi-letter tokens is a real hyphen - keep it.
- Separators between multi-letter name tokens (".", "-", "_") are preserved exactly as transcribed. In German speech "Punkt", "Bindestrich", "Unterstrich" are distinct words; do not normalise one to another.
- Digits, year suffixes, and other handle elements (e.g. "h47", "1983", "user2") are part of the caller's chosen handle. Keep them in place verbatim. Do NOT strip them."""

PHONE_RULES = """PHONE_NUMBER

Prefix normalisation (apply BEFORE any other phone rule):
- "+CC..." -> already international, keep as-is.
- "00CC..." (no "+") -> replace "00" with "+".
- "0CC..." (no "+", where CC is a valid country code such as 49) -> drop the leading "0", prepend "+".
- "0NNN..." (no "+", where NNN is a national-format number such as German mobile prefixes 0151/0152/0157/0159/0160/0162/0163/0170/0171/0172/0173/0174/0175/0176/0177/0178/0179, or German landlines 030/040/069/089/...) -> drop the leading "0" and prepend the country code that matches the caller's country (default "+49" for German national-format numbers).

After prefix normalisation the number is "+CC" + digits_after_cc.

Stutter rule:
    if len(digits_after_cc) >= 12 and digits_after_cc[-1] == digits_after_cc[-2]:
        digits_after_cc = digits_after_cc[:-1]    # drop the final stutter digit

Otherwise preserve every transcribed digit. The final phone field is "+" + country_code + digits_after_cc."""


# ---------------------------------------------------------------------------
# Inter-key consistency rules
# ---------------------------------------------------------------------------
NAME_EMAIL_CONSISTENCY = """NAME <-> EMAIL CONSISTENCY (token-level - modify the email ONLY when its alphabetic substring is a near-miss of a spoken name)

Step A: split the email's local-part by ".", "-", "_" into tokens.

Step B: for each token, find a single "candidate string" - one contiguous run of alphabetic letters within the token that the edit-distance test will apply to:
- Token is purely alphabetic length ≥ 2 -> candidate string is the whole token.
- Token is purely digits OR a single alphabetic letter -> no candidate string; keep the token unchanged.
- Token is mixed (letters + digits) -> take the LONGEST contiguous alphabetic run inside it. If that run has length ≥ 2 it is the candidate string; everything else in the token (digits, other letters) is preserved in place. If the longest run has length < 2, no candidate string; keep the token unchanged.

Step C: for the candidate string (lowercased ASCII), compute the minimum edit distance to (a) the spoken first name and (b) the spoken last name (both lowercased ASCII).
- min_distance == 0 -> already correct, leave the token unchanged.
- min_distance ≤ 2, OR ≤ 1 when the candidate string has length ≤ 5 -> NAME-LIKE substring. Replace the candidate string inside the token with the standard country spelling of the matching name (lowercase ASCII, no diacritics). Every other character of the token (digits, other letters) keeps its original position.
- min_distance > 2 -> does NOT resemble the spoken name. Leave the entire token unchanged.

Step D: NAME fields use the standard country spelling WITH diacritics. The email's local-part stays ASCII (no diacritics) regardless."""

COUNTRY_CONSISTENCY = """COUNTRY <-> NAME / EMAIL / PHONE CONSISTENCY
- The caller's country has been determined from the transcript (name heritage + email TLD/domain). It is the authoritative signal for orthography.
- NAME spelling and diacritics follow the country's conventions, NOT the German transcript's spelling.
- Phone country code defaults to the caller's country when the number is in national format and the country is unambiguous; otherwise default to +49 for German national-format numbers."""


# ---------------------------------------------------------------------------
# Country-detection rules (used by Layer 2)
# ---------------------------------------------------------------------------
COUNTRY_RULES = """You determine the LINGUISTIC ORIGIN of a phone caller's name from a German speech-to-text transcript. The goal is to identify which spelling conventions / orthography to apply to the caller's NAME - NOT to guess where they currently live.

Use these signals, in order of strength:
1. Email country-code TLD (.fr, .br, .se, .es, .it, .nl, .uk, .pt, .jp, etc.) and full domain (uol.com.br, free.fr, gmx.de, gmail.com, ...).
2. Cultural pattern of the NAME itself (given name + family name).
3. Phone country code (WEAKEST - many callers use a German number even if not German).

HARD RULES:
- The country you return determines what orthography to apply to the name.
- NEVER return "United States" / "USA" / language_code "en" as a default for ambiguous-or-Hispanic-or-foreign names. The USA gives no orthographic guidance because its names span every tradition. If you would otherwise pick USA, pick the country of the name's actual heritage instead.
- Map name heritage to a single linguistic-origin country:
    * Hispanic / Spanish-language pattern -> "Spain", "es" (regardless of where the caller lives).
    * French pattern -> "France", "fr".
    * Portuguese-language pattern: email TLD .br -> "Brazil", "pt-br"; otherwise -> "Portugal", "pt".
    * Scandinavian pattern -> "Sweden", "sv" (or Norway/Denmark/Iceland with corresponding code if the pattern is unmistakeably one of those).
    * Italian pattern -> "Italy", "it".
    * Japanese pattern -> "Japan", "ja".
    * Polish pattern -> "Poland", "pl".
    * Dutch / Flemish pattern -> "Netherlands", "nl".
    * Clearly German with no foreign signal -> "Germany", "de".
    * Genuinely ambiguous English (Smith, Brown, Johnson) with no foreign signal -> "United Kingdom", "en" - NOT "United States".
- Trust signal 1 (email TLD/domain) most. Trust signal 2 (name heritage) when the email is a generic global domain like gmail.com / hotmail.com / outlook.com / yahoo.com.
- Country of residence is irrelevant. Linguistic origin of the NAME is the only thing that matters."""


# ---------------------------------------------------------------------------
# Transcript context (the same sentence is true for every layer)
# ---------------------------------------------------------------------------
TRANSCRIPT_CONTEXT = """The transcript was produced by a German speech-to-text system; it uses German orthography and German phonetics regardless of the caller's actual country. Trust its WORDS, distrust its SPELLING of non-German content."""
