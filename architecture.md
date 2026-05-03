# Pipeline architectures

Two pipelines extract caller info (`first_name`, `last_name`, `email`, `phone_number`) from a German-language phone-call recording. Both use Voxtral-Mini-3B for transcription and `gpt-5.4-mini` via langchain for the structured-extraction stages.

## test.py - single-stage extraction with majority voting

```mermaid
flowchart TD
    A([WAV file]) --> B["<b>Layer 1 - Voxtral</b><br/>VoxtralForConditionalGeneration<br/>language=de, bf16"]
    B --> T["transcript<br/><i>(plain text)</i>"]
    T --> X["<b>Layer 2 - Extraction × 3</b><br/>chain.batch([input] × 3)<br/>gpt-5.4-mini, temp=0.7<br/>system prompt holds:<br/>· diacritic patterns<br/>· name<->email token-level consistency<br/>· phone prefix normalization<br/>· VERIFY checklist V1–V4"]
    X --> R1["run 1<br/>{first, last, email, phone}"]
    X --> R2["run 2"]
    X --> R3["run 3"]
    R1 & R2 & R3 --> M["<b>majority_vote()</b><br/>per-key Counter.most_common(1)"]
    M --> P["<b>Phone post-processing</b><br/>· strip spaces<br/>· prefix: +00→+, +0→+, 00→+, 0→+49<br/>· stutter: drop final repeat if ≥14 digits<br/>· format as +CC AAA NNNNNNNN"]
    P --> O([results.json])

    style B fill:#e1f5ff,stroke:#0288d1
    style X fill:#fff3e0,stroke:#f57c00
    style M fill:#f3e5f5,stroke:#7b1fa2
    style P fill:#e8f5e9,stroke:#388e3c
```

Behaviour: a single LLM does origin inference + extraction + verification in one pass. Three runs are aggregated with naive per-key majority voting. The pipeline depends entirely on the extraction prompt to enforce all the rules; if a rule is silently dropped in 2 of 3 runs, the wrong answer wins.

## test2.py - five-layer pipeline with self-consistency, selection and verification

```mermaid
flowchart TD
    A([WAV file]) --> L1["<b>Layer 1 - Voxtral</b><br/>transcription"]
    L1 --> T["transcript"]

    T --> L2["<b>Layer 2 - Country detection</b><br/>gpt-5.4-mini, temp=0<br/>maps name pattern + email TLD<br/>→ country + language_code<br/>(USA forbidden as default)"]
    L2 --> CI["country_info<br/>{country, language_code, reasoning}"]

    T --> L3
    CI --> L3["<b>Layer 3 - Self-consistent extraction</b><br/>chain.batch([input] × 5)<br/>gpt-5.4-mini, temp=0.7<br/>uses given country (no inference)<br/>applies token-level consistency rule"]
    L3 --> RUNS["runs[5]<br/>each: {thinking, first, last, email, phone}"]

    RUNS --> AGG["<b>aggregate_candidates()</b><br/>per-key Counter.most_common()<br/>→ {key: [{value, count}, ...]}"]
    AGG --> CAND["candidates dict<br/><i>(values are lists)</i>"]

    CAND --> L4
    CI --> L4
    T --> L4["<b>Layer 4 - Selection</b><br/>gpt-5.4-mini, temp=0<br/>PRIME DIRECTIVE: unanimous wins<br/>· country-override<br/>· diacritic preference<br/>· phone prefix sanity<br/>· name-token agreement (overrides majority)"]
    L4 --> DRAFT["draft answer<br/>{first, last, email, phone}"]

    DRAFT --> L5
    CI --> L5
    T --> L5["<b>Layer 5 - Verification</b><br/>gpt-5.4-mini, temp=0<br/>NAME <-> email reconciliation<br/>· edit-distance per email name-token<br/>· dist=0 → ok; 1–2 → reconcile to country standard<br/>· >2 → leave alone (handle/prefix)"]
    L5 --> VER["verified answer"]

    VER --> P["<b>Phone post-processing</b><br/>same rules as test.py"]
    P --> O([results2.json])

    style L1 fill:#e1f5ff,stroke:#0288d1
    style L2 fill:#fff8e1,stroke:#fbc02d
    style L3 fill:#fff3e0,stroke:#f57c00
    style AGG fill:#f3e5f5,stroke:#7b1fa2
    style L4 fill:#fce4ec,stroke:#c2185b
    style L5 fill:#e8eaf6,stroke:#3f51b5
    style P fill:#e8f5e9,stroke:#388e3c
```

Differences from test.py:
- Country is decided **once** (Layer 2) and passed as input to all later layers, instead of being re-inferred 3× independently.
- Self-consistency runs are aggregated as **candidate lists with counts**, not collapsed by majority. Layer 4 sees the full distribution.
- Layer 4 is a dedicated **selection** step that can override majority (e.g., a 1-of-5 candidate with the right diacritics beats a 4-of-5 candidate without them).
- Layer 5 is a dedicated **verification** step that reconciles NAME <-> email when their letters are within edit-distance 2 - catches the case where extraction got one side right and the other side wrong.

## Per-stage settings

| Stage                | model         | temp | runs | output                                |
|----------------------|---------------|------|------|---------------------------------------|
| Voxtral (both)       | Voxtral-Mini-3B-2507 | -    | 1    | transcript                            |
| test.py extract      | gpt-5.4-mini  | 0.7  | 3    | full record (voted per-key)           |
| test2 country        | gpt-5.4-mini  | 0    | 1    | `{country, language_code, reasoning}` |
| test2 extract        | gpt-5.4-mini  | 0.7  | 5    | candidate list per key                |
| test2 select         | gpt-5.4-mini  | 0    | 1    | draft answer                          |
| test2 verify         | gpt-5.4-mini  | 0    | 1    | reconciled answer                     |
