# Phonebot Caller-Info Extraction Pipeline

> Submission for the Phonebot AI Engineer Technical Challenge.
> Prepared by **Melih Unsal** for **JUPUS**.

This repository contains my solution to the challenge: extract
`first_name`, `last_name`, `email`, and `phone_number` from a corpus
of 30 German phone-call recordings.

The strongest configuration of the pipeline reaches **29 of 30
full-record matches** (96.7%) on the dataset, with 100% accuracy on
the phone-number and first-name fields, and 96.7% on email and last
name. A transcript-only baseline (no audio LLM, no re-listen) reaches
26 of 30, so the audio-grounded layers contribute three additional
recordings.

A full write-up lives in [`study/paper.pdf`](study/paper.pdf) and a
slide deck for the Loom walkthrough is in
[`study/presentation.pptx`](study/presentation.pptx).

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Results](#results)
4. [Repository layout](#repository-layout)
5. [Installation](#installation)
6. [Running the full pipeline](#running-the-full-pipeline)
7. [Running ablation experiments via the CLI](#running-ablation-experiments-via-the-cli)
8. [Reproducing the figures](#reproducing-the-figures)
9. [Building the paper](#building-the-paper)
10. [Configuration knobs](#configuration-knobs)
11. [Files of interest](#files-of-interest)

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

### Headline accuracy

| Configuration                                  | n | Mean | Std | Best |
|------------------------------------------------|--:|-----:|----:|-----:|
| **Voxtral + Scribe \| full pipeline**          | 2 | 26.5 | 2.5 | **29** |
| Voxtral \| +targeted, no Whisper               | 1 | 28.0 |  -  | 28   |
| Voxtral \| +targeted, +Whisper                 | 2 | 26.5 | 0.5 | 27   |
| Voxtral \| no targeted (ablation)              | 1 | 26.0 |  -  | 26   |
| Scribe  \| no audio LLM, no targeted (ablation)| 1 | 26.0 |  -  | 26   |
| Scribe  \| +targeted                           | 1 | 26.0 |  -  | 26   |
| Scribe  \| no targeted                         | 2 | 25.0 | 0.0 | 25   |

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

- `full_match_by_config.png` — Mean full-record accuracy per config.
- `per_key_accuracy.png` — Per-key accuracy across configs.
- `timing_breakdown.png` — Per-stage wall-clock latency.
- `agreement_distribution.png` — Inter-candidate agreement on the
  best run.
- `targeted_mode_distribution.png` — Layer 4.5 firing modes
  (cropped / skipped\_unanimous / skipped).
- `error_taxonomy.png` — Aggregated error counts per field across
  every reported run.
- `pipeline_only.png` — Cropped TikZ pipeline diagram for the slides.

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
│   ├── country.py                  # Layer 2
│   ├── extract_llm.py              # Layer 3 (transcript-LLM, candidate-1)
│   ├── extract_voxtral.py          # Layer 4 (SpeechLM, candidate-2)
│   ├── targeted_extract.py         # Layer 4.5 (cropped re-listen)
│   ├── segments.py                 # email/phone segment detection
│   ├── audio_crop.py               # soundfile-based audio cropping
│   ├── reconcile.py                # Layer 5 + 4 safety nets
│   ├── schema.py                   # Layer 6 (Pydantic CallerInfo)
│   ├── validators.py               # shared by schema + evaluator
│   └── evaluate.py                 # per-key normalised matching
├── main.py                         # end-to-end orchestrator
├── results/                        # development reports (timestamped JSON)
├── study/                          # the ablation study
│   ├── cli.py                      # CLI override harness for src/config.py
│   ├── analyze.py                  # aggregates reports -> CSVs + figures
│   ├── build_pptx.py               # generates presentation.pptx
│   ├── paper.tex                   # LaTeX source of the paper
│   ├── paper.pdf                   # built paper (11 pages)
│   ├── presentation.pptx           # 19-slide deck for the Loom walkthrough
│   ├── README.md                   # study-specific reproduction notes
│   ├── figures/                    # generated PNGs + TikZ pipeline diagram
│   ├── data/                       # aggregated CSVs (runs, per_key, mismatches)
│   └── results/                    # ablation-run reports
├── PIPELINE.md                     # design narrative (mermaid diagrams)
├── architecture.md                 # earlier architecture notes
├── requirements.txt
└── README.md                       # original challenge README
```

---

## Installation

The pipeline runs against:

- Python 3.12 (the `.venv` was built with 3.12 originally; 3.10+
  should also work).
- A CUDA GPU with at least ~10 GB of free VRAM (Voxtral-Mini-3B in
  bf16 plus Whisper-large-v3-turbo in fp32 share the GPU; the dev
  machine was an RTX 4090 with 24 GB).
- `OPENAI_API_KEY` for the LLM layers (country, candidate-1,
  reconciler).
- `ELEVENLABS_API_KEY` for the Scribe v2 transcriber.

Quick install:

```bash
git clone <this-repo>
cd phonebot_challenge

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...
```

A `.env` file at the repo root is auto-loaded by `python-dotenv`,
so you can put the keys there instead.

---

## Running the full pipeline

The default configuration in `src/config.py` runs the dual-STT
pipeline with the audio LLM, the targeted re-listen, and the
Whisper timestamper enabled (the strongest configuration).

```bash
python main.py
```

This processes every WAV in `data/recordings/` and writes a
timestamped JSON report to `results/report_YYYY-MM-DD_HHMMSS.json`.
On a successful run the report ends with an `evaluation` block that
compares predictions against `data/ground_truth.json`.

The pipeline auto-resumes from the most recent unfinished report,
so a crash mid-run does not throw away earlier work. Set
`AUTO_RESUME = False` in `src/config.py` (or pass
`--no-auto-resume` through the CLI harness) to always start fresh.

---

## Running ablation experiments via the CLI

Every config knob in `src/config.py` is exposed as a flag through
`study/cli.py`. The wrapper mutates the live `src.config` module
before `main.py` is imported, so the override is picked up by every
downstream layer.

### Examples

**Transcript-only baseline (no GPU, fast).**

```bash
python -m study.cli \
    --transcription elevenlabs:scribe_v2 \
    --instruct "" --timestamp "" \
    --no-targeted \
    --no-auto-resume \
    --results-dir study/results \
    --tag scribe_only_no_audiollm
```

This run uses no GPU memory at all — only the ElevenLabs API and
the OpenAI API. It is the natural floor of the study.

**Voxtral-only without targeted re-listen.**

```bash
python -m study.cli \
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
python -m study.cli \
    --transcription voxtral:mistralai/Voxtral-Mini-3B-2507,elevenlabs:scribe_v2 \
    --instruct voxtral:mistralai/Voxtral-Mini-3B-2507 \
    --timestamp whisper:openai/whisper-large-v3-turbo \
    --no-auto-resume \
    --results-dir study/results \
    --tag dual_full_pipeline
```

### All CLI flags

```
python -m study.cli --help
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

## Reproducing the figures

```bash
python -m study.analyze
```

This script:

1. Reads every `report_*.json` under `results/` and `study/results/`.
2. Writes:
   - `study/data/runs.csv` — one row per report (config + accuracy + timings).
   - `study/data/per_key.csv` — per-key accuracy per run.
   - `study/data/mismatches.csv` — every individual error across all runs.
3. Renders six PNG figures into `study/figures/` (the same ones
   embedded in the paper).

Re-running the analyzer after adding a new report will pick up the
new data automatically — there is no manual table maintenance.

---

## Building the paper

The paper is plain LaTeX with a TikZ pipeline diagram embedded as
Figure 1. To rebuild the PDF after editing `study/paper.tex` or
regenerating figures:

```bash
cd study
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex   # second pass for cross-refs
```

The required LaTeX packages are part of TeX Live's `texlive-latex-extra`
and `texlive-pictures` (booktabs, tabularx, hyperref, microtype,
tikz, etc.). On Debian/Ubuntu:

```bash
sudo apt install texlive-latex-extra texlive-pictures texlive-science
```

To rebuild the slide deck:

```bash
python study/build_pptx.py
```

This writes `study/presentation.pptx`. The deck is 19 slides at
16:9 widescreen and embeds the same figures as the paper.

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
- `PROMPT_VERSION` (default `"2026-05-03.v1"`): stamped into every
  report so A/B comparisons can group runs by prompt version.

---

## Files of interest

If you only have a few minutes:

- [`study/paper.pdf`](study/paper.pdf) — the full write-up
  (11 pages, with the architecture diagram, methodology, ablation
  study, and error analysis).
- [`study/presentation.pptx`](study/presentation.pptx) — the
  19-slide deck for the Loom walkthrough.
- [`PIPELINE.md`](PIPELINE.md) — design narrative with mermaid
  diagrams for every layer.
- [`src/config.py`](src/config.py) — central config; all the
  rules and orthography priors that the prompts compose from.
- [`src/reconcile.py`](src/reconcile.py) — the four deterministic
  safety nets, in code.
- [`study/cli.py`](study/cli.py) — the CLI override harness.
- [`study/analyze.py`](study/analyze.py) — the aggregator that
  builds the CSVs and figures from raw reports.

If you want to dig deeper, the `results/` and `study/results/`
directories contain the full per-recording trace for every run that
appears in the paper, including the transcripts, the candidate
distribution, the targeted re-listen output, and the reconciler's
reasoning paragraph.
