# Phonebot Caller-Info Extraction Pipeline

> Submission for the Phonebot AI Engineer Technical Challenge.
> Prepared by **Melih Unsal** for **JUPUS**.

This repository contains my solution to the challenge: extract
`first_name`, `last_name`, `email`, and `phone_number` from a corpus
of 30 German phone-call recordings.

The strongest configuration of the pipeline reaches **29 of 30
full-record matches** (96.7%) on the dataset, with 100% accuracy on
the phone-number and first-name fields and 96.7% on email and last
name. A transcript-only baseline (no audio LLM, no re-listen)
reaches 26 of 30, so the audio-grounded layers contribute three
additional recordings.

A full write-up lives in [`study/paper.pdf`](study/paper.pdf) and a
slide deck for the Loom walkthrough is in
[`study/presentation.pptx`](study/presentation.pptx).

---

## TL;DR - three commands to run everything

```bash
# 1. Install
pip install -e '.[gpu,observability]'

# 2. Configure
cp .env.example .env                  # add OPENAI_API_KEY + ELEVENLABS_API_KEY
                                      # plus LANGFUSE_* keys (see below)

# 3. Run
phonebot run                          # batch over data/recordings/
```

After installation, every CLI subcommand becomes available:

```bash
phonebot doctor                                       # GPU / API / Langfuse status
phonebot run                                          # batch over data/recordings/
phonebot extract data/recordings/call_01.wav --json   # one recording -> stdout JSON
phonebot evaluate results/report_2026-05-03_180227.json
phonebot ablate --transcription elevenlabs:scribe_v2 --no-targeted --tag scribe_only
```

Every LLM and audio call is traced to Langfuse when the
`LANGFUSE_*` env vars are present (free tier on
[`cloud.langfuse.com`](https://cloud.langfuse.com) works out of the
box).

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Results](#results)
4. [Repository layout](#repository-layout)
5. [Installation](#installation)
6. [The `phonebot` CLI](#the-phonebot-cli)
7. [Running the full pipeline](#running-the-full-pipeline)
8. [Running ablation experiments via the CLI](#running-ablation-experiments-via-the-cli)
9. [Observability with Langfuse](#observability-with-langfuse)
10. [Voxtral backend choice (HF transformers, vLLM optional)](#voxtral-backend-choice-hf-transformers-vllm-optional)
11. [Troubleshooting](#troubleshooting)
12. [Reproducing the figures](#reproducing-the-figures)
13. [Configuration knobs](#configuration-knobs)
14. [Adding a new entity to the pipeline](#adding-a-new-entity-to-the-pipeline)
15. [Future improvements](#future-improvements)
16. [Files of interest](#files-of-interest)

---

## What it does

For every German phone-call recording in `data/recordings/`, the
pipeline produces a JSON record with four fields:

```json
{
  "first_name": "Marie",
  "last_name":  "Lefevre",
  "email":      "marie.lefevre@yahoo.fr",
  "phone_number": "+4917655012345"
}
```

The pipeline is robust to:

- callers from 11 different language traditions (German, French,
  Italian, Spanish, Brazilian, Swedish, Japanese, Polish, English,
  Arabic, Irish);
- diacritic-heavy names that German STT systems strip
  (\textit{García}, \textit{Lefèvre}, \textit{Martínez}, \textit{Schröder});
- spelled-out emails (`Punkt`, `Bindestrich`, `Unterstrich`, `ät`,
  letter-by-letter spelling);
- long digit sequences with stutters in the phone number;
- LLM stochasticity (self-consistency voting + reconciler).

---

## Architecture at a glance

The pipeline has seven layers. The full diagram is on page 3 of
[`study/paper.pdf`](study/paper.pdf); the short version:

```
WAV  ->  Layer 1a (Voxtral STT)  +  Layer 1b (Scribe v2 STT)
              |                            |
          transcript A                 transcript B
              |                            |
              +----- Layer 2 ---------+
                  Country / origin
              |                            |
          Layer 3a              Layer 3b
        (Cand 1A, 5x LLM)   (Cand 1B, 5x LLM)
                       \  /
                        \/      Layer 4 (Cand 2)
                        /\      Voxtral SpeechLM
                       /  \     (whole-record)
                      v    v
                  any candidate
                  disagrees on
                  email or phone?  --(no)--> Layer 5
                       |
                      yes
                       |
                       v
              Layer 4.5 (targeted re-listen)
              Voxtral SpeechLM on cropped audio
              (Whisper word timestamps from Layer 1c)
                       |
                       v
                   Layer 5 (Reconciler)
                   gpt-5.4-mini + 4 deterministic safety nets
                       |
                       v
                   Layer 6 (Schema)
                   Pydantic + phonenumbers
                   E.164 / NFC / lower-case
                       |
                       v
                   final JSON
```

A few key ideas:

- **Two STT models in parallel.** ASR errors are correlated within a
  single STT model and uncorrelated across models, so adding a second
  STT was the cheapest accuracy lever in the pipeline.
- **Country decided once.** A dedicated layer maps name pattern + email
  TLD to a single linguistic-origin country, which becomes the
  authority for all downstream orthography rules.
- **Self-consistency voting on every LLM call.** Five votes per
  transcript at temperature 0.7, three votes for the audio LLM at
  temperature 0.01.
- **Conditional re-listen.** Layer 4.5 only fires on records where
  the whole-record candidates disagree on email or phone. Whisper is
  invoked lazily, only on those records.
- **Four deterministic safety nets after the LLM reconciler.** The
  reconciler is an LLM and can be charmed by a confident-but-wrong
  re-listen. Net 1 filters structurally invalid targeted output; Net
  2 enforces unanimous candidates; Net 3 enforces strict-majority;
  Net 4 aligns name fields with the email's local-part using
  edit-distance.

The full methodology is in Section 4 of
[`study/paper.pdf`](study/paper.pdf).

---

## Results

### Headline accuracy (best of every recorded run per configuration)

| Configuration                                  | n runs | Best |
|------------------------------------------------|-------:|-----:|
| **Voxtral + Scribe \| full pipeline**          | 3      | **29** |
| Voxtral \| +targeted, no Whisper               | 1      | 28   |
| Voxtral \| +targeted, +Whisper                 | 2      | 27   |
| Voxtral \| no targeted (ablation)              | 1      | 26   |
| Scribe  \| no audio LLM, no targeted (ablation)| 1      | 26   |
| Scribe  \| +targeted                           | 1      | 26   |
| Scribe  \| no targeted                         | 2      | 25   |

Table is computed by [`study/analyze.py`](study/analyze.py) from every
report under `results/` and `study/results/`.

### Per-key accuracy on the best run (29/30)

| Field          | Correct | Total | Accuracy |
|----------------|--------:|------:|---------:|
| first\_name    |   30    |   30  |  100.0%  |
| last\_name     |   29    |   30  |   96.7%  |
| email          |   29    |   30  |   96.7%  |
| phone\_number  |   30    |   30  |  100.0%  |

The single residual error is `call_23`
(`emma.andersson@gmail.com`/`Andersson`): both STTs hear a single
`n` (`Anderson`), and the email's local-part also says `anderson`,
so the email-name alignment net has nothing to correct against.
A discussion of this case is in Section 8 of the paper.

### Figures

All figures used in the paper live in
[`study/figures/`](study/figures/) and are regenerated by
`python -m study.analyze`:

- `full_match_by_config.png` - Full-record accuracy per config.
- `per_key_accuracy.png` - Per-key accuracy across configs.
- `timing_breakdown.png` - Per-stage wall-clock latency.
- `agreement_distribution.png` - Inter-candidate agreement on the
  best run.
- `targeted_mode_distribution.png` - Layer 4.5 firing modes
  (cropped / skipped\_unanimous / skipped).
- `error_taxonomy.png` - Aggregated error counts per field across
  every reported run.
- `pipeline_only.png` - Cropped TikZ pipeline diagram for the slides.

---

## Repository layout

```
phonebot_challenge/
├── data/
│   ├── ground_truth.json           # 30 ground-truth records
│   └── recordings/                 # call_01.wav ... call_30.wav
├── src/                            # the pipeline source
│   ├── config.py                   # central config (all knobs live here)
│   ├── prompts.py                  # composed LLM prompts
│   ├── transcription.py            # Voxtral / Scribe / Whisper backends
│   ├── observability.py            # Langfuse integration (opt-in)
│   ├── country.py                  # Layer 2
│   ├── extract_llm.py              # Layer 3 (transcript-LLM, candidate-1)
│   ├── extract_voxtral.py          # Layer 4 (SpeechLM, candidate-2)
│   ├── targeted_extract.py         # Layer 4.5 (cropped re-listen)
│   ├── segments.py                 # email/phone segment detection
│   ├── audio_crop.py               # soundfile-based audio cropping
│   ├── reconcile.py                # Layer 5 + 4 safety nets
│   ├── schema.py                   # Layer 6 (Pydantic CallerInfo)
│   ├── validators.py               # shared by schema + evaluator
│   ├── evaluate.py                 # per-key normalised matching
│   └── cli.py                      # `phonebot` console script
├── main.py                         # end-to-end orchestrator
├── results/                        # development reports (timestamped JSON)
├── study/                          # the ablation study
│   ├── cli.py                      # CLI override harness for src/config.py
│   ├── analyze.py                  # aggregates reports -> CSVs + figures
│   ├── build_pptx.py               # generates presentation.pptx
│   ├── paper.tex                   # LaTeX source of the paper
│   ├── paper.pdf                   # built paper (11 pages)
│   ├── presentation.pptx           # 19-slide deck for the Loom walkthrough
│   ├── figures/                    # generated PNGs + TikZ pipeline diagram
│   ├── data/                       # aggregated CSVs (runs, per_key, mismatches)
│   └── results/                    # ablation-run reports
├── PIPELINE.md                     # design narrative (mermaid diagrams)
├── architecture.md                 # earlier architecture notes
├── pyproject.toml                  # `pip install -e .` entry point
├── requirements.txt                # CPU-friendly base dependencies
├── requirements-gpu.txt            # GPU-only deps (transformers, librosa, ...)
├── .env.example                    # template for OPENAI / ELEVENLABS / LANGFUSE
└── README.md                       # this file
```

---

## Installation

The pipeline runs on Python 3.10+. There are two install profiles
depending on whether you have a CUDA GPU.

```bash
git clone <this-repo>
cd phonebot_challenge

python3 -m venv .venv
source .venv/bin/activate

# Base install: works on every machine. Uses the Scribe + OpenAI path.
# Pipeline ceiling on this dataset is 26/30.
pip install -e .

# GPU install: adds transformers / accelerate / mistral-common / librosa
# for Voxtral and Whisper. Install torch separately for your CUDA version
# from https://pytorch.org/get-started/ (default config uses torch 2.5.1+cu124).
pip install -e '.[gpu]'

# Observability: adds the Langfuse SDK (no-op until LANGFUSE_* env vars
# are set).
pip install -e '.[observability]'

# Optional vLLM Voxtral backend (only for stacks where vLLM works).
pip install -e '.[gpu,gpu_vllm]'

# Everything (GPU + observability + analysis tooling).
pip install -e '.[all]'
```

### Required environment variables

| Variable | Used by | Notes |
|---|---|---|
| `OPENAI_API_KEY` | LLM layers (country / candidate-1 / reconciler) | required |
| `ELEVENLABS_API_KEY` | Scribe v2 transcriber | required (the Scribe path is the only fallback when GPU is unavailable) |
| `LANGFUSE_PUBLIC_KEY` | observability | optional; pipeline runs identically without it |
| `LANGFUSE_SECRET_KEY` | observability | optional |
| `LANGFUSE_HOST` | observability | defaults to `https://cloud.langfuse.com` |

A `.env` file at the repo root is auto-loaded by `python-dotenv`.
See [`.env.example`](.env.example) for the full template.

### GPU prerequisites (optional)

If you want the full pipeline (Voxtral + Whisper instead of just
Scribe), the host needs:

- **NVIDIA driver ≥ 535** (any recent install satisfies this).
- **About 10 GB of free VRAM.** Voxtral-Mini-3B in bf16 plus
  Whisper-large-v3-turbo in fp32 share the GPU; the dev machine was
  an RTX 4090 with 24 GB.
- **CUDA via torch wheels.** Install torch from the matching CUDA
  index (`pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1`).
  No system CUDA install needed - the wheel bundles the runtime libs.

If neither a GPU nor torch is available, `phonebot run` still works
through the Scribe + OpenAI path; you just cap at 26/30 instead of 29/30.

---

## The `phonebot` CLI

After `pip install -e .`, the `phonebot` console script becomes
available with five subcommands:

```bash
phonebot doctor                                       # resolved config + GPU/API/Langfuse status
phonebot run                                          # batch over data/recordings/
phonebot extract data/recordings/call_01.wav --json   # one recording -> JSON
phonebot evaluate results/report_*.json               # re-evaluate a report
phonebot ablate --transcription elevenlabs:scribe_v2 \
                --instruct "" --no-targeted \
                --tag scribe_only                     # ablation run (wraps study.cli)
```

Run `phonebot <cmd> --help` for per-subcommand flags. The
`doctor` subcommand is the recommended first step on a new machine -
it prints the resolved config, the available GPU, and whether
OpenAI / ElevenLabs / Langfuse are reachable.

Example `doctor` output on a working setup:

```text
== phonebot doctor ==
  repo root           : /home/you/phonebot_challenge
  python              : 3.12.x
  torch               : 2.5.1+cu124  cuda=True devices=1
    [0] NVIDIA GeForce RTX 4090
  vllm                : not installed
  OPENAI_API_KEY      : set
  ELEVENLABS_API_KEY  : set
  Langfuse            : enabled
    host              : https://cloud.langfuse.com
  TRANSCRIPTION_MODELS: ['voxtral:mistralai/Voxtral-Mini-3B-2507', 'elevenlabs:scribe_v2']
  INSTRUCT_MODEL      : 'voxtral:mistralai/Voxtral-Mini-3B-2507'
  TIMESTAMP_MODEL     : 'whisper:openai/whisper-large-v3-turbo'
  USE_TARGETED_EXTRACTION: True
```

---

## Running the full pipeline

The default configuration in `src/config.py` runs the dual-STT
pipeline with the audio LLM, the targeted re-listen, and the
Whisper timestamper enabled (the strongest configuration).

```bash
phonebot run
```

This processes every WAV in `data/recordings/` and writes a
timestamped JSON report to `results/report_YYYY-MM-DD_HHMMSS.json`.
On a successful run the report ends with an `evaluation` block that
compares predictions against `data/ground_truth.json`.

The pipeline auto-resumes from the most recent unfinished report,
so a crash mid-run does not throw away earlier work. Pass
`--no-resume` to start fresh, or `--results-dir <path>` to write
elsewhere.

**No GPU on the host?** The pipeline detects this at startup, logs a
warning that Voxtral / Whisper backends could not be built, and runs
the Scribe + OpenAI path automatically. No code change required.

---

## Running ablation experiments via the CLI

Every config knob in `src/config.py` is exposed as a flag through the
`phonebot ablate` subcommand (which wraps the lower-level
`study.cli` harness). The wrapper mutates the live `src.config`
module before `main.py` is imported, so the override is picked up by
every downstream layer.

### Examples

**Transcript-only baseline (no GPU, fast).**

```bash
phonebot ablate \
    --transcription elevenlabs:scribe_v2 \
    --instruct "" --timestamp "" \
    --no-targeted \
    --no-auto-resume \
    --results-dir study/results \
    --tag scribe_only_no_audiollm
```

This run uses no GPU memory at all - only the ElevenLabs API and
the OpenAI API. It is the natural floor of the study.

**Voxtral-only without targeted re-listen.**

```bash
phonebot ablate \
    --transcription voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --instruct voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --timestamp "" \
    --no-targeted \
    --no-auto-resume \
    --results-dir study/results \
    --tag voxtral_only_no_targeted
```

**Best-known dual-STT configuration.**

```bash
phonebot ablate \
    --transcription voxtral:mistralai/Voxtral-Mini-3B-2507,elevenlabs:scribe_v2 \
    --instruct voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --timestamp whisper:openai/whisper-large-v3-turbo \
    --no-auto-resume \
    --results-dir study/results \
    --tag dual_full_pipeline
```

### All CLI flags

```
phonebot ablate --help
```

| Flag | Maps to | Purpose |
|---|---|---|
| `--transcription <list>` | `TRANSCRIPTION_MODELS` | comma-separated list of STT specs |
| `--instruct <spec>` | `INSTRUCT_MODEL` | empty string disables Layer 4 + 4.5 |
| `--timestamp <spec>` | `TIMESTAMP_MODEL` | empty string disables Whisper Layer 1c |
| `--targeted` / `--no-targeted` | `USE_TARGETED_EXTRACTION` | toggles Layer 4.5 |
| `--targeted-pad <s>` | `TARGETED_PAD_SECONDS` | crop window padding |
| `--num-llm-votes <n>` | `NUM_LLM_VOTES` | candidate-1 self-consistency |
| `--num-instruct-votes <n>` | `NUM_INSTRUCT_VOTES` | candidate-2 self-consistency |
| `--num-targeted-votes <n>` | `NUM_TARGETED_VOTES` | Layer 4.5 self-consistency |
| `--language <iso>` | `TRANSCRIPTION_LANGUAGE` | ISO-639-1 code |
| `--phone-region <cc>` | `DEFAULT_PHONE_REGION` | default phone region |
| `--llm <name>` | `LLM_MODEL_NAME` | LLM model id |
| `--prompt-version <s>` | `PROMPT_VERSION` | tag stamped into the report |
| `--auto-resume` / `--no-auto-resume` | `AUTO_RESUME` | resume vs fresh run |
| `--results-dir <path>` | `RESULTS_DIR` | where reports are written |
| `--tag <s>` | --- | filename tag (`report_<tag>_<ts>.json`) |

---

## Observability with Langfuse

In production I had been using Portkey for prompt management and LLM
usage tracking. The open-source equivalent that fits this codebase
best is **[Langfuse](https://langfuse.com/)**: it has a drop-in
LangChain `CallbackHandler`, tracks tokens / cost / latency per call,
has a UI for inspection, and supports prompt versioning to replace
Portkey's prompt feature.

**Why Langfuse over alternatives:**

- **Helicone** - proxy-based; requires routing traffic through their
  proxy, less natural with LangChain.
- **Phoenix (Arize)** - broader ML observability; heavier weight
  than needed here.
- **OpenLLMetry** - OpenTelemetry-based; valuable for cross-vendor
  setups but the LangChain integration is less polished.

Langfuse is the lightest path that covers everything Portkey did.

**How it is wired in:** the small module
[`src/observability.py`](src/observability.py) exposes
`langchain_config()` (returns a `{"callbacks": [...]}` dict when
configured) and `span(name, **metadata)` (a context manager for
tracing audio-backend calls). Both are no-ops when the
`LANGFUSE_*` env vars are missing, so adding Langfuse never breaks
existing runs.

### Setup with Langfuse Cloud (recommended, ~5 minutes)

`pip install -e '.[gpu,observability]'` installed the **Langfuse
Python SDK**, but the SDK alone has nowhere to send traces - you also
need a **Langfuse server**. The fastest way is the free hosted
instance at [`cloud.langfuse.com`](https://cloud.langfuse.com):

1. **Sign up** at <https://cloud.langfuse.com> and create a new
   project (any name, e.g. `phonebot`).

2. In the project, go to **Settings → API Keys → Create new API
   keys**. Copy the public key (`pk-lf-…`) and the secret key
   (`sk-lf-…`). The secret is shown only once - copy it now.

3. **Add the keys to your `.env`** at the repo root:

   ```dotenv
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxx
   LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

4. **Verify the wiring:**

   ```bash
   phonebot doctor
   ```

   This should now print
   `Langfuse: enabled  host: https://cloud.langfuse.com`.
   If it still says `disabled`, the env vars haven't been loaded -
   make sure `.env` is at the repo root (or `export` the vars in
   your shell).

5. **Run any subcommand** that hits an LLM or audio backend, e.g.

   ```bash
   phonebot run
   ```

   On startup you'll see one extra line:
   `Langfuse session: report_2026-05-04_011234  tags: [...]`.
   Open the Langfuse UI, click **Sessions** in the sidebar, and you
   should see a row with that exact session id appearing within a
   few seconds. Click into it to see one trace per recording.

### Self-hosting Langfuse (optional)

If you'd rather self-host, the repo ships a `docker-compose.yml`
that brings up a local Langfuse + Postgres stack and seeds it with
the keys from `.env` on first start. After cloning:

```bash
docker compose --profile obs up -d langfuse langfuse-db
# Langfuse UI is now at http://localhost:3000
# Use LANGFUSE_HOST=http://localhost:3000 in .env
```

The bundled compose file uses Langfuse's `LANGFUSE_INIT_*` env
vars to auto-create the org / project / user / keys at first boot,
so no manual key-copy step is needed. Default login is
`admin@phonebot.local` / `changeme1234` (rotate before sharing).

### What ends up in Langfuse for one `phonebot run`

| Layer | UI grouping | Purpose |
|---|---|---|
| `phonebot run` invocation | one **Session** named after the report (e.g. `report_2026-05-04_011234`) | All traces from the same batch group together. |
| One recording (`call_NN.wav`) | one **Trace** | Top-level entry point per call. |
| `layer1.transcribe[voxtral:...]` | nested span | Voxtral STT call. |
| `layer1.transcribe[elevenlabs:scribe_v2]` | nested span | Scribe STT call. |
| `layer1b.timestamp[whisper:...]` | nested span (lazy) | Whisper word timestamps. |
| `layer2.country` | LLM span | gpt-5.4-mini one-shot. |
| `layer3.candidate_1` | LLM span × 5 votes × N transcripts | Self-consistent extraction. |
| `layer4.candidate_2[voxtral:...]` | nested span | SpeechLM whole-record. |
| `layer4_5.targeted_extract` | nested span (conditional) | Cropped re-listen. |
| `layer5.reconcile` | LLM span | Reasoning reconciler. |

Every LLM span carries **tokens, cost, latency, full prompt, full
response**. Every audio span carries **duration, audio path,
language, mode** (cropped / full / skipped).

The session is also tagged with the run config so you can filter:
`prompt=2026-05-03.v1`, `stt=voxtral+scribe`, `audio_llm=on`,
`targeted=on`, `whisper=on`. Ablation runs add an extra
`ablation=<your-tag>` chip.

### Comparing the ablation study in Langfuse

The ablation study is exactly what Langfuse's session/tag model is
built for. Each `phonebot ablate --tag <name>` run creates its own
Session with `ablation=<name>` as a filterable tag.

Suggested workflow for showing the team:

```bash
# 1. Strongest configuration (the headline 29/30 run)
phonebot ablate \
    --transcription voxtral:mistralai/Voxtral-Mini-3B-2507,elevenlabs:scribe_v2 \
    --instruct voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --timestamp whisper:openai/whisper-large-v3-turbo \
    --no-auto-resume \
    --tag dual_full_pipeline

# 2. Voxtral-only, no targeted re-listen
phonebot ablate \
    --transcription voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --instruct voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --timestamp "" \
    --no-targeted --no-auto-resume \
    --tag voxtral_only_no_targeted

# 3. Transcript-only baseline (no audio LLM, no re-listen)
phonebot ablate \
    --transcription elevenlabs:scribe_v2 \
    --instruct "" --timestamp "" \
    --no-targeted --no-auto-resume \
    --tag scribe_only_no_audiollm
```

Each invocation produces:

- a tagged **Session** in the Langfuse UI;
- a JSON report under `study/results/`;
- a row in `study/data/runs.csv` after `python -m study.analyze`.

In the Langfuse UI, **Filter by tag** (top-right of the Sessions
list) → pick e.g. `ablation=scribe_only_no_audiollm` to see only
that ablation. Pick two ablations side-by-side and the **Compare**
button shows you per-call diffs, token-cost diffs, and latency
diffs across the runs - exactly the view you want for the team
walkthrough.

### Prompt management

Langfuse's **Prompts** tab replaces what Portkey was doing for
prompt versioning: create / edit / version prompts in the UI, then
fetch them from Python at runtime. The current pipeline still keeps
prompts in [`src/prompts.py`](src/prompts.py) (composed from rules
in [`src/config.py`](src/config.py)), but moving any individual
prompt to Langfuse is a small change - example pattern:

```python
from langfuse import get_client
client = get_client()
prompt = client.get_prompt("country", version="latest")  # versioned
COUNTRY_SYSTEM = prompt.compile(...)
```

---

## Voxtral backend choice (HF transformers, vLLM optional)

Voxtral runs through **HF transformers** by default - that's the
backend the 29/30 headline was produced on (8 dev runs). The
`voxtral-vllm` backend exists as an opt-in for stacks where vLLM is
known to work, but it's not bundled in the default install because
vLLM's Voxtral support has been unstable on bleeding-edge torch (a
0.20.x model-registry SIGSEGV on torch 2.11+cu130).

```python
# default in src/config.py
TRANSCRIPTION_MODELS = [
    "voxtral:mistralai/Voxtral-Mini-3B-2507",
    "elevenlabs:scribe_v2",
]
INSTRUCT_MODEL = "voxtral:mistralai/Voxtral-Mini-3B-2507"
```

### Why HF + eager attention is reliable

The original concern with HF transformers was a rare SIGSEGV inside
`sdpa_attention_forward` during generation. Two safety layers in this
repo make that crash invisible in practice:

- `VOXTRAL_ATTN_IMPL = "eager"` (set in `src/config.py`) bypasses
  PyTorch's SDPA path entirely. Eager attention is slower per call
  but doesn't crash. The 8 dev reports under `results/` were all
  produced on this exact configuration.
- `AUTO_RESUME = True` (also in `src/config.py`) makes
  `phonebot run` pick up the most recent unfinished report and
  continue from where the previous run stopped, so even the
  rare residual crash is recoverable end-to-end.

### Switching to the vLLM backend (opt-in)

If you have a torch+CUDA stack where vLLM's Voxtral support is
stable (broadly, torch 2.4–2.7 with CUDA 12), you can switch by:

1. Install vLLM:

   ```bash
   pip install -e '.[gpu,gpu_vllm]'
   # or just: pip install vllm>=0.6
   ```

2. Edit `src/config.py`:

   ```python
   TRANSCRIPTION_MODELS = [
       "voxtral-vllm:mistralai/Voxtral-Mini-3B-2507",
       "elevenlabs:scribe_v2",
   ]
   INSTRUCT_MODEL = "voxtral-vllm:mistralai/Voxtral-Mini-3B-2507"
   ```

3. Re-run `phonebot doctor` to confirm vLLM loads cleanly.

If the model-registry inspection subprocess SIGSEGVs, your
torch+vLLM combo is one of the broken ones - fall back to
`voxtral:`. The HF backend is robust and produces the same 29/30
headline on this dataset.

The GPU detection in `src/transcription.py` ensures both backends
fail loudly with a clear "requires a CUDA GPU" message on hosts
without a GPU, instead of crashing late in inference.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `phonebot run` reports `Langfuse: disabled` | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` missing in `.env` | add the keys; restart the run. The pipeline is otherwise unaffected. |
| `every transcriber failed` on every record | `ELEVENLABS_API_KEY` missing/invalid AND no GPU available | check `.env`; on a no-GPU host Scribe is the only path, so a missing key is a hard fail. |
| Dual-STT run reports 28/30 instead of 29/30 (call_25 fails on `Lefebvre`/`Lefevre`) | ElevenLabs Scribe silently failed on most records (rate-limit / network), pipeline degraded to single-Voxtral, Net 4 lost the cross-STT signal | `phonebot run` now prints a transcriber-summary at the end of every run that flags partial failures. The ElevenLabs backend retries transient errors 5× with exponential backoff. If the summary still shows >10% Scribe failures, the ElevenLabs quota or rate-limit is the bottleneck - wait or upgrade the plan. |
| `cuda=False` in `phonebot doctor` despite having an NVIDIA GPU | torch was installed without CUDA wheels (default index gives the CPU-only build) | reinstall torch from the matching CUDA index: `pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1` |
| Rare `SIGSEGV` mid-run during Voxtral generation | torch SDPA-attention bug on a specific torch+CUDA combo | the default config sets `VOXTRAL_ATTN_IMPL = "eager"` which avoids the SDPA path. `AUTO_RESUME=True` makes the run resume from the last checkpoint anyway. |
| Switched to `voxtral-vllm:` and got `Model architectures … failed to be inspected` | vLLM 0.20.x V1 engine SIGSEGV on Voxtral with bleeding-edge torch | revert `TRANSCRIPTION_MODELS` / `INSTRUCT_MODEL` in `src/config.py` to the `voxtral:` prefix. The HF backend is the supported default. |
| First `phonebot run` is very slow | Voxtral / Whisper weights are downloading from HuggingFace into `~/.cache/huggingface` | one-time cost; subsequent runs reuse the cached weights. |
| Self-hosted Langfuse keeps restarting with `Error: CLICKHOUSE_URL is not configured. Migrating from V2?` | the `langfuse:latest` Docker tag now points at v3, which needs ClickHouse + Redis + MinIO in addition to Postgres | the bundled compose file pins to `langfuse:2` (v2 only needs Postgres and has every feature this repo uses). If you customised the compose file, restore the pin or migrate to a full v3 stack. |
| `localhost:3000` shows nothing after `docker compose --profile obs up -d` | container failed to start; check `docker compose logs langfuse --tail 50` | usually the v3/v2 mismatch above. After the fix, `docker compose --profile obs down && up -d`. |
| `phonebot doctor` reports `Langfuse: enabled` but no traces appear in the UI, AND `langfuse.Langfuse().auth_check()` raises `pydantic ValidationError ... metadata field required` | Python SDK is v3+ but the bundled server is v2; the v3 SDK calls v3-only endpoints | the included `requirements.txt` and `pyproject.toml` pin `langfuse>=2.0,<3`. Run `pip install -e '.[observability]' --upgrade` to install the matching SDK. |
| `phonebot extract` shows audio-span traces (`layer1.transcribe`, `layer4.candidate_2`) in Langfuse but no LangChain traces (`layer2.country`, `layer3.candidate_1`, `layer5.reconcile`) | LangChain ≥ 1.0 is installed; Langfuse v2's `CallbackHandler` was written against LangChain 0.x and silently no-ops on 1.x | the included deps pin `langchain>=0.3,<1.0` and `langchain-core>=0.3,<1.0`. Reinstall with `pip install -e '.[observability]' --upgrade`, or downgrade explicitly: `pip install 'langchain>=0.3,<1.0' 'langchain-core>=0.3,<1.0' 'langchain-openai>=0.3,<1.0'`. |

---

## Reproducing the figures

```bash
python -m study.analyze
```

This script:

1. Reads every `report_*.json` under `results/` and `study/results/`.
2. Writes:
   - `study/data/runs.csv` - one row per report (config + accuracy + timings).
   - `study/data/per_key.csv` - per-key accuracy per run.
   - `study/data/mismatches.csv` - every individual error across all runs.
3. Renders six PNG figures into `study/figures/` (the same ones
   embedded in the paper).

Re-running the analyzer after adding a new report will pick up the
new data automatically - there is no manual table maintenance.

---

## Configuration knobs

All knobs live in [`src/config.py`](src/config.py). The four most
important ones for accuracy:

| Constant | Default | Effect |
|---|---|---|
| `TRANSCRIPTION_MODELS` | `["voxtral:...", "elevenlabs:scribe_v2"]` | List of STT models. Adding a second STT was the cheapest accuracy improvement. |
| `INSTRUCT_MODEL` | `voxtral:...` | The audio LLM used for Layer 4 + 4.5. Empty string disables both. |
| `TIMESTAMP_MODEL` | `whisper:...-turbo` | Layer 1c. Required by Layer 4.5 cropping when the primary STT lacks word timestamps. |
| `USE_TARGETED_EXTRACTION` | `True` | Master switch for Layer 4.5. |

Self-consistency vote counts:

| Constant | Default | Notes |
|---|---|---|
| `NUM_LLM_VOTES` | 5 | candidate-1 (transcript-LLM) per transcript |
| `NUM_INSTRUCT_VOTES` | 3 | candidate-2 (SpeechLM whole-record) |
| `NUM_TARGETED_VOTES` | 3 | Layer 4.5 cropped re-listen |

Other knobs:

- `TARGETED_PAD_SECONDS` (default 3.0): seconds of context added on
  each side of a detected email/phone segment before cropping.
- `VOXTRAL_ATTN_IMPL` (default `"eager"`): switches Voxtral's
  attention path. Override to `"sdpa"` or `"flash_attention_2"`
  if your environment supports it.
- `AUTO_RESUME` (default `True`): on startup, resumes the most
  recent unfinished report instead of starting fresh.
- `PROMPT_VERSION` (default `"2026-05-03.v1"`): stamped into every
  report so A/B comparisons can group runs by prompt version.

---

## Adding a new entity to the pipeline

The pipeline is built around a single list of keys in
[`src/config.py`](src/config.py):

```python
KEYS = ["first_name", "last_name", "email", "phone_number"]
```

Every layer that produces or consumes structured data is driven by
this list - the prompts, the self-consistency vote, the reconciler,
and the schema. Adding a new entity (say `company_name`, `address`,
`appointment_date`, `case_topic`) is **three small changes**.

### Worked example: adding `company_name`

#### Step 1 - register the key + write its rule

In [`src/config.py`](src/config.py):

```python
KEYS = [
    "first_name", "last_name", "email", "phone_number",
    "company_name",                                       # NEW
]

# Rule string the LLM follows when extracting this field.
COMPANY_NAME_RULES = """COMPANY_NAME
- Use the company's legal name as it would appear on official documents.
- Capitalisation matches the company's own brand style (e.g. "BMW",
  not "Bmw"; "iPhone", not "Iphone").
- If the caller spells out an abbreviation letter-by-letter
  (e.g. "B-M-W"), reconstruct the abbreviation as one token.
- "" if no company name is mentioned in the recording."""
```

#### Step 2 - wire the rule into the extraction prompt

In [`src/prompts.py`](src/prompts.py), import the new rule string and
add it to the `EXTRACT_SYSTEM` template (and any other prompt that
should know about the new field, e.g. the audio LLM direct prompt
and the reconciler):

```python
from src.config import (
    ...,
    COMPANY_NAME_RULES,            # NEW
)

EXTRACT_SYSTEM = f"""...
{FIRST_NAME_RULES}
{LAST_NAME_RULES}
{EMAIL_RULES}
{PHONE_RULES}
{COMPANY_NAME_RULES}               # NEW
...
"""
```

That's it for the LLM side - `extract_candidate_1` already calls
`majority_vote(runs, KEYS)` so the new key is included automatically.

#### Step 3 - extend the schema

In [`src/schema.py`](src/schema.py), add the field to the Pydantic
model and (optionally) a normaliser:

```python
class CallerInfo(BaseModel):
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone_number: str = ""
    company_name: str = ""                                # NEW

    # Optional: collapse whitespace and strip stray punctuation.
    @field_validator("company_name", mode="before")
    @classmethod
    def _company(cls, v):
        return " ".join((v or "").split()).strip()
```

Then update `validate_caller_info()` so the new key is propagated:

```python
safe = {
    "first_name":  raw.get("first_name", "")  or "",
    ...
    "company_name": raw.get("company_name", "") or "",    # NEW
}
```

That's the entire change.

#### What the pipeline does for free

After those three edits, every layer picks up the new key automatically:

| Layer | What happens with `company_name` |
|---|---|
| Layer 3 (transcript-LLM × N votes) | LLM produces `company_name` per the rule; `majority_vote` collapses across votes |
| Layer 4 (audio LLM whole-record) | Voxtral receives the same key list and emits a value |
| Layer 5 (reconciler) | The reconciler picks per-key, so the new key joins the existing four |
| Layer 5 safety nets | Net 2 (unanimous) and Net 3 (strict-majority) work on every key in `KEYS`, no change needed |
| Layer 6 (schema) | Pydantic validates and normalises the new field |
| Evaluator | `field_match` compares against ground truth; just add the key to `data/ground_truth.json` to score it |
| Langfuse traces | New field appears in every per-layer trace's output payload |

### Other entity types and where the rule goes

| Entity | Suggested rule placement | Notes |
|---|---|---|
| `address`, `postal_code` | own `*_RULES` constant | Country-aware; reuse the country-detection result for postal-code formats |
| `appointment_date`, `appointment_time` | own `*_RULES` constant | Add a `dateparser`/ISO-8601 normaliser in `validators.py` |
| `case_topic`, `urgency` | classification-style; rule lists the allowed values | Use a Pydantic `Literal[...]` type for built-in validation |
| `id_number`, `case_number` | own `*_RULES` constant | Add a regex normaliser; benefits from Layer 4.5 cropped re-listen |

For fields that need **cross-field consistency** (like the existing
NAME ↔ EMAIL rule), add a new constant alongside `NAME_EMAIL_CONSISTENCY`
in `src/config.py` and reference it in the relevant prompts.

For fields that need **country-aware orthography**, the country code
returned by Layer 2 is already available to every downstream layer,
so country-specific rules can live inside the entity's `*_RULES`
string.

---

## Future improvements

The current pipeline is built for batch post-call extraction at high
accuracy. The following are the most concrete next steps to take it
toward a production phone-bot integration.

### Real-time streaming

The pipeline today processes a complete WAV after the call ends.
Inside a live phone bot, fields should be extracted incrementally as
the caller speaks - first name appears as soon as the caller says
their name, phone number as soon as it has been dictated, and so on.
This means swapping the batch-style STT call for a streaming one
(ElevenLabs Scribe and Voxtral both support partial-result streams),
running candidate-1 over the partial transcript on every commit, and
pushing the reconciler decisions to the UI as soon as the relevant
content has been heard. The four deterministic safety nets work
unchanged on incremental updates.

### Conversational integration with TTS-driven clarification

The most natural extension is to put the pipeline inside the agent's
turn loop, not after it. When a layer's confidence is low (for
example, candidate-1 disagrees on the email and the targeted re-listen
returns a structurally invalid value), the agent should ask a
clarifying question through the TTS channel - *"Sorry, did you say
M-U-E-L-L-E-R or M-Ü-L-L-E-R?"* - instead of guessing. The
disagreement signal is already produced by the existing Layer 4.5
gate; what's missing is wiring that signal to a follow-up turn in
the dialog manager. This turns the pipeline from a passive extractor
into an active interrogator and removes the residual error class
(call_23 / *Andersson*) entirely.

### Adaptive memory + dynamic keyterm prompting

Once the agent confirms a tricky token via the clarification turn
above, that confirmation is information the system should remember.
A short-lived per-call memory (the agent learned that the caller's
surname is *Andersson*, not *Anderson*) plus a longer-lived per-tenant
memory (this customer base over-indexes on Swedish surnames; bias
the keyterms accordingly) should be queryable from the prompts.
Concretely: store confirmed tokens in a key-value store keyed on
(caller, session), and pull the recent confirmed-token set into the
ElevenLabs `keyterms` parameter and the SpeechLM prompt prelude on
every subsequent turn. This makes the system *learn during the
call*, which is something neither STT nor SpeechLM does on its own.

### Multilingual support (Danish, Finnish, ...)

The pipeline is currently configured for German
(`TRANSCRIPTION_LANGUAGE = "de"`). Expanding to Danish, Finnish, or
any other language is mostly a configuration change: switch the STT
language code, swap the Voxtral / Whisper language to match, and
extend the country-detection prompt so it can map names with the
relevant heritage signals. The country-aware orthography rules in
`src/config.py` already separate "what the STT heard" from "how the
caller's country writes it"; the same separation works for any
language pair as long as the rules are translated. The harder part
is data - Danish and Finnish callers tend to use different separator
words for spelling out emails (`piste` for "dot" in Finnish, `prik`
in Danish), which need to be added to the keyterm and segment-detection
lists.

### Concurrent users (target: 20 simultaneous calls)

Today the pipeline loads Voxtral and Whisper into a single GPU and
processes one recording at a time. A 20-call concurrent target
needs a deployment shape closer to a service: split each backend
into its own process, batch incoming requests at the model level,
and let the orchestrator (one phonebot worker per call) talk to the
backends over an RPC. vLLM's continuous batching is the right
primitive on the SpeechLM side; an ElevenLabs API pool with a
per-tenant rate limiter handles the Scribe traffic. Each phonebot
worker stays tiny and CPU-bound; the GPU work is amortised across
all calls. Langfuse already groups traces by session, so 20
concurrent calls produce 20 distinct sessions in the UI without
any code change.

### Smaller, faster models for concurrency

The cleanest accuracy/latency tradeoff for a 20-user concurrent
target is to swap the heavy backbones for smaller variants of the
same family: Voxtral → a smaller SpeechLM (or Voxtral-Mini at a
shorter `max_tokens`), `gpt-5.4-mini` → an even smaller open model
served via vLLM, Whisper-large-v3 → distil-whisper. Each of these
trades a few accuracy points for substantially lower latency per
call, which is the right shape under load. The ablation harness
(`phonebot ablate --llm <model>`) already supports comparing models
side-by-side, so this is empirical work, not a redesign.

### Long-conversation optimisation

The dataset's recordings are short (≤30 s) but a real phone call can
run several minutes. For long calls, two things help: (1) chunk the
audio into 30-60 s windows and run the pipeline incrementally (the
streaming path above), and (2) maintain a per-call "extracted so
far" cache so subsequent layers don't re-process audio that was
already covered. Layer 4.5's targeted re-listen is already a model
of how to do this - only re-listen to the contested segment, not
the whole call.

---

## Files of interest

If you only have a few minutes:

- [`study/paper.pdf`](study/paper.pdf) - the full write-up
  (11 pages, with the architecture diagram, methodology, ablation
  study, and error analysis).
- [`study/presentation.pptx`](study/presentation.pptx) - the
  19-slide deck for the Loom walkthrough.
- [`PIPELINE.md`](PIPELINE.md) - design narrative with mermaid
  diagrams for every layer.
- [`src/config.py`](src/config.py) - central config; all the
  rules and orthography priors that the prompts compose from.
- [`src/reconcile.py`](src/reconcile.py) - the four deterministic
  safety nets, in code.
- [`src/observability.py`](src/observability.py) - the Langfuse
  integration (only ~140 lines; no-op when keys are absent).
- [`src/cli.py`](src/cli.py) - the `phonebot` console script.
- [`study/cli.py`](study/cli.py) - the ablation override harness.
- [`study/analyze.py`](study/analyze.py) - the aggregator that
  builds the CSVs and figures from raw reports.

If you want to dig deeper, the `results/` and `study/results/`
directories contain the full per-recording trace for every run that
appears in the paper, including the transcripts, the candidate
distribution, the targeted re-listen output, and the reconciler's
reasoning paragraph.

---

> **Note on Docker:** the repository also includes a `Dockerfile`,
> `docker-compose.yml`, and a `phonebot` shell wrapper for
> teams that prefer running the pipeline in containers. Those files
> are intact and functional but optional - the README above
> documents the simpler native-Python path that works on any system
> with Python 3.10+ and a CUDA GPU (optional).
