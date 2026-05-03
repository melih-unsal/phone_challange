"""Build study/presentation.pptx for the Loom walkthrough.

Slides target a ~10-minute walkthrough:
  1. Title
  2. Problem
  3. Why it's hard
  4. Pipeline overview (image)
  5. Layer 1 - Multi-backend STT
  6. Layer 2 - Country detection
  7. Layer 3 - Candidate-1 (transcript-LLM, self-consistency)
  8. Layer 4 - Candidate-2 (SpeechLM)
  9. Layer 4.5 - Targeted re-listen
 10. Layer 5 - Reconciler + 4 safety nets
 11. Layer 6 - Schema
 12. CLI override harness
 13. Headline numbers (table + chart image)
 14. Per-key accuracy chart
 15. Latency breakdown chart
 16. Agreement + targeted-mode chart
 17. Ablation takeaways
 18. Error analysis (call_23 Andersson)
 19. Conclusion
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent
FIG  = ROOT / "figures"
OUT  = ROOT / "presentation.pptx"

# 16:9 widescreen
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

PRIMARY      = RGBColor(0x0B, 0x3D, 0x91)
ACCENT       = RGBColor(0xC6, 0x28, 0x28)
TEXT_DARK    = RGBColor(0x22, 0x2B, 0x35)
TEXT_LIGHT   = RGBColor(0x55, 0x5F, 0x6B)
BG           = RGBColor(0xFA, 0xFB, 0xFD)
RULE         = RGBColor(0xCC, 0xD0, 0xD8)


def new_pres() -> Presentation:
    p = Presentation()
    p.slide_width  = SLIDE_W
    p.slide_height = SLIDE_H
    return p


def add_blank(p: Presentation):
    layout = p.slide_layouts[6]  # blank
    return p.slides.add_slide(layout)


def add_bg(slide):
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H,
    )
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.shadow.inherit = False
    return bg


def add_textbox(slide, x, y, w, h, text, *, size=18, bold=False,
                color=TEXT_DARK, align=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return tb


def add_bullets(slide, x, y, w, h, lines, *, size=18, color=TEXT_DARK):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = "•  " + line
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def add_rule(slide, x, y, w, h=Pt(2), color=PRIMARY):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    return bar


def add_header(slide, kicker, title):
    add_textbox(slide, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4),
                kicker.upper(), size=12, bold=True, color=PRIMARY)
    add_textbox(slide, Inches(0.6), Inches(0.7), Inches(12), Inches(0.7),
                title, size=28, bold=True, color=TEXT_DARK)
    add_rule(slide, Inches(0.6), Inches(1.43), Inches(2.5))


def add_footer(slide, page_num, total):
    add_textbox(slide, Inches(0.6), Inches(7.1), Inches(8), Inches(0.3),
                "Phonebot Caller-Info Pipeline  -  Melih Unsal  -  JUPUS",
                size=10, color=TEXT_LIGHT)
    add_textbox(slide, Inches(11.8), Inches(7.1), Inches(1.4), Inches(0.3),
                f"{page_num} / {total}", size=10, color=TEXT_LIGHT,
                align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def build():
    p = new_pres()
    total_slides = 19  # used for footer

    # --- Slide 1: Title ---
    s = add_blank(p); add_bg(s)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.6),
                              SLIDE_W, Inches(2.0))
    band.line.fill.background()
    band.fill.solid()
    band.fill.fore_color.rgb = PRIMARY
    add_textbox(s, Inches(0.6), Inches(2.75), Inches(12), Inches(0.5),
                "PHONEBOT TECHNICAL CHALLENGE", size=14, bold=True,
                color=RGBColor(0xFF, 0xFF, 0xFF))
    add_textbox(s, Inches(0.6), Inches(3.10), Inches(12), Inches(1.2),
                "A Multi-Transcript, Self-Consistent,\nAudio-Grounded Pipeline",
                size=34, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    add_textbox(s, Inches(0.6), Inches(4.85), Inches(12), Inches(0.5),
                "Caller-info extraction from German phone-call recordings",
                size=20, color=TEXT_LIGHT)
    add_textbox(s, Inches(0.6), Inches(5.35), Inches(12), Inches(0.5),
                "Prepared by Melih Unsal for JUPUS",
                size=18, bold=True, color=TEXT_DARK)
    add_textbox(s, Inches(0.6), Inches(5.85), Inches(12), Inches(0.5),
                "2026-05-03", size=14, color=TEXT_LIGHT)
    add_footer(s, 1, total_slides)

    # --- Slide 2: Problem ---
    s = add_blank(p); add_bg(s)
    add_header(s, "the task", "What I had to extract")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "Input: 30 German phone-call recordings (16-bit WAV)",
                    "Output per call: first name, last name, email, phone number",
                    "Phone must be E.164; email lower-cased; names NFC-normalised",
                    "Ground truth allows multiple acceptable spellings per field",
                ], size=20)
    add_footer(s, 2, total_slides)

    # --- Slide 3: Why hard ---
    s = add_blank(p); add_bg(s)
    add_header(s, "why it is hard", "The dataset is acoustically and linguistically diverse")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "Callers from 11 language traditions: German, French, Italian, Spanish, "
                    "Brazilian, Swedish, Japanese, Polish, English, Arabic, Irish",
                    "Diacritics are mandatory (Garcia -> Garcia, Lefevre -> Lefevre)",
                    "Spelled-out emails: Punkt, Bindestrich, Unterstrich, at",
                    "Long digit sequences with stutters (e.g. ...12313)",
                    "German STT systems have no incentive to reproduce non-German orthography",
                ], size=18)
    add_footer(s, 3, total_slides)

    # --- Slide 4: Pipeline overview (big image) ---
    # Layout edited manually by Melih: image moved to the right half of the
    # slide, no side-text block, single-line caption sits just above the
    # footer.
    s = add_blank(p); add_bg(s)
    add_header(s, "architecture", "Full pipeline at a glance")
    pipe = FIG / "pipeline_only.png"
    if pipe.exists():
        s.shapes.add_picture(str(pipe), Inches(5.43), Inches(0.11),
                             width=Inches(7.57))
    add_textbox(s, Inches(1.33), Inches(7.07), Inches(12), Inches(0.2),
                "Solid arrows = unconditional flow.   Dashed arrows = "
                "lazy paths (Whisper, targeted re-listen).",
                size=10, color=TEXT_LIGHT)
    add_footer(s, 4, total_slides)

    # --- Slide 5: Layer 1 STT ---
    # Edited manually by Melih: dropped the TRANSCRIPTION_MODELS bullet,
    # simplified the Whisper bullet, added a benchmark link footer.
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 1", "Two STT models in parallel = uncorrelated errors")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(1.45),
                [
                    "Voxtral-Mini-3B-2507 (local, bf16): strong on letters, no timestamps",
                    "ElevenLabs Scribe v2 (API): word timestamps + keyterms biasing "
                    "(Punkt, Bindestrich, gmail.com, ...)",
                    "Errors correlate within one model, uncorrelate across models",
                    "Whisper-large-v3-turbo generates the timestamps lazily (Layer 1c)",
                ], size=18)
    bench = s.shapes.add_textbox(Inches(6.0), Inches(7.0),
                                  Inches(6.41), Inches(0.30))
    bench.text_frame.word_wrap = True
    par = bench.text_frame.paragraphs[0]
    r1 = par.add_run()
    r1.text = "Benchmark Link: "
    r1.font.size = Pt(11); r1.font.color.rgb = TEXT_LIGHT
    r1.font.name = "Calibri"
    r2 = par.add_run()
    r2.text = "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard"
    r2.font.size = Pt(11); r2.font.italic = True
    r2.font.color.rgb = PRIMARY; r2.font.name = "Calibri"
    add_footer(s, 5, total_slides)

    # --- Slide 6: Layer 2 country ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 2", "Country detection drives orthography")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "Single LLM call decides the caller's linguistic-origin country once",
                    "Signals (priority order): email TLD/domain, name pattern, phone code",
                    "Hard rule: USA is forbidden as a default for ambiguous names",
                    "Output is a single country string used by every later layer",
                    "Removes a class of failure where 5 LLM votes pick 3 different countries",
                ], size=18)
    add_footer(s, 6, total_slides)

    # --- Slide 7: Layer 3 candidate-1 ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 3", "Candidate-1: self-consistent transcript-LLM")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "For each transcript, run gpt-5.4-mini at temperature 0.7, five times",
                    "Per-key majority vote -> one candidate per transcript",
                    "Prompt embeds country-specific orthography, name<->email rules, "
                    "phone prefix rules",
                    "Self-consistency averages out single-shot LLM noise",
                    "Number of candidate-1s = number of STT models",
                ], size=18)
    add_footer(s, 7, total_slides)

    # --- Slide 8: Layer 4 candidate-2 SpeechLM ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 4", "Candidate-2: SpeechLM directly on the audio")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "Voxtral consumes the WAV plus a single text instruction",
                    "Three runs at temperature 0.01 (near-greedy), majority-voted",
                    "Strong on literal letters and digits the caller spoke",
                    "Weak on diacritics and country-aware orthography "
                    "(audio carries no orthographic signal)",
                    "Different error profile from candidate-1 - this is the point",
                ], size=18)
    add_footer(s, 8, total_slides)

    # --- Slide 9: Layer 4.5 targeted re-listen ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 4.5", "Targeted re-listen: rewind the parts you are unsure about")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4.5),
                [
                    "Gate: opens only if whole-record candidates disagree on email or phone",
                    "Word timestamps come from Scribe or from Whisper (Layer 1c)",
                    "Segment detection looks for Punkt/Bindestrich/at + domain hints, "
                    "or digit-words and 'plus'",
                    "Crop the audio to the detected window plus 3 seconds context",
                    "Re-prompt the SpeechLM with a single-field prompt, three times, "
                    "majority-voted",
                    "Most records skip the layer entirely - it is selective by design",
                ], size=18)
    add_footer(s, 9, total_slides)

    # --- Slide 10: Layer 5 reconciler + safety nets ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 5", "Reconciler + 4 deterministic safety nets")
    add_bullets(s, Inches(0.6), Inches(1.7), Inches(12), Inches(5),
                [
                    "LLM reconciler (gpt-5.4-mini, T=0) sees: transcripts, country, "
                    "every candidate-1, candidate-2, targeted re-listen",
                    "Net 1 - drop targeted email/phone if it is structurally invalid "
                    "(no @, won't parse)",
                    "Net 2 - if every whole-record candidate agrees, that value wins "
                    "regardless of the LLM",
                    "Net 3 - strict majority overrides the LLM (e.g. 2 of 3 candidates "
                    "agree on the right phone)",
                    "Net 4 - email <-> name alignment using ASCII-folded edit distance "
                    "(fixes Lefevre/Lefebvre/Le Fevre)",
                    "Each net catches a class of obvious cases the LLM might miss",
                ], size=17)
    add_footer(s, 10, total_slides)

    # --- Slide 11: Layer 6 schema ---
    s = add_blank(p); add_bg(s)
    add_header(s, "layer 6", "Schema validation and normalisation")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(4),
                [
                    "Pydantic CallerInfo model is the single normalisation point",
                    "Names: NFC normalised",
                    "Emails: lower-cased",
                    "Phone numbers: parsed by phonenumbers, re-emitted in E.164",
                    "Same validators.py used by the evaluator -> formatting can never "
                    "lose points",
                ], size=18)
    add_footer(s, 11, total_slides)

    # --- Slide 12: CLI override harness ---
    s = add_blank(p); add_bg(s)
    add_header(s, "ablation tooling", "CLI harness exposing every config knob")
    add_bullets(s, Inches(0.6), Inches(1.8), Inches(12), Inches(2.5),
                [
                    "study/cli.py mutates src.config before main.py is imported",
                    "Every relevant constant in src/config.py becomes a flag",
                    "Tagged reports go to study/results/ for aggregation",
                ], size=18)
    code_box = s.shapes.add_textbox(Inches(0.6), Inches(3.5),
                                     Inches(12), Inches(2.8))
    tf = code_box.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    r = p1.add_run()
    r.text = ("python -m study.cli \\\n"
              "    --transcription elevenlabs:scribe_v2 \\\n"
              "    --instruct \"\" --timestamp \"\" \\\n"
              "    --no-targeted --no-auto-resume \\\n"
              "    --tag scribe_only_no_audiollm")
    r.font.name = "Consolas"
    r.font.size = Pt(16)
    r.font.color.rgb = TEXT_DARK
    add_footer(s, 12, total_slides)

    # --- Slide 13: Headline numbers (table) ---
    s = add_blank(p); add_bg(s)
    add_header(s, "results", "Headline accuracy across configurations")
    rows = [
        ["Configuration", "n", "Mean", "Std", "Best"],
        ["Voxtral + Scribe  |  full pipeline",      "2", "26.5", "2.5",  "29"],
        ["Voxtral  |  +targeted, no Whisper",        "1", "28.0", "-",    "28"],
        ["Voxtral  |  +targeted, +Whisper",          "2", "26.5", "0.5",  "27"],
        ["Voxtral  |  no targeted (NEW)",            "1", "26.0", "-",    "26"],
        ["Scribe  |  no audio LLM, no targeted (NEW)","1","26.0", "-",    "26"],
        ["Scribe  |  +targeted",                     "1", "26.0", "-",    "26"],
        ["Scribe  |  no targeted",                   "2", "25.0", "0.0",  "25"],
    ]
    table_shape = s.shapes.add_table(
        rows=len(rows), cols=5,
        left=Inches(0.6), top=Inches(1.7),
        width=Inches(12.1), height=Inches(4.6),
    )
    tbl = table_shape.table
    widths = [Inches(6.6), Inches(0.8), Inches(1.4), Inches(1.4), Inches(1.4)]
    for i, w in enumerate(widths):
        tbl.columns[i].width = w
    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = ""
            tf = cell.text_frame
            tf.margin_left = Inches(0.1)
            tf.margin_right = Inches(0.1)
            par = tf.paragraphs[0]
            run = par.add_run()
            run.text = cell_text
            run.font.size = Pt(15)
            run.font.name = "Calibri"
            if r_idx == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid()
                cell.fill.fore_color.rgb = PRIMARY
            else:
                run.font.color.rgb = TEXT_DARK
                if r_idx == 1:
                    run.font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = (
                    RGBColor(0xF1, 0xF4, 0xFA) if r_idx % 2 == 1
                    else RGBColor(0xFF, 0xFF, 0xFF)
                )
    add_textbox(s, Inches(0.6), Inches(6.6), Inches(12), Inches(0.4),
                "Best run = 29 / 30 (dual STT + audio LLM + targeted "
                "+ Whisper)",
                size=14, bold=True, color=ACCENT)
    add_footer(s, 13, total_slides)

    # --- Slide 14: full match chart ---
    s = add_blank(p); add_bg(s)
    add_header(s, "results", "Full-record matches per configuration")
    img = FIG / "full_match_by_config.png"
    if img.exists():
        s.shapes.add_picture(str(img), Inches(1.3), Inches(1.6),
                             width=Inches(10.7))
    add_footer(s, 14, total_slides)

    # --- Slide 15: per-key chart ---
    s = add_blank(p); add_bg(s)
    add_header(s, "results", "Per-key accuracy: most errors live in last name + email")
    img = FIG / "per_key_accuracy.png"
    if img.exists():
        s.shapes.add_picture(str(img), Inches(1.3), Inches(1.6),
                             width=Inches(10.7))
    add_footer(s, 15, total_slides)

    # --- Slide 16: latency chart ---
    s = add_blank(p); add_bg(s)
    add_header(s, "performance", "Per-stage wall-clock latency")
    img = FIG / "timing_breakdown.png"
    if img.exists():
        s.shapes.add_picture(str(img), Inches(1.5), Inches(1.6),
                             width=Inches(10.3))
    add_textbox(s, Inches(0.6), Inches(6.95), Inches(12), Inches(0.4),
                "Targeted re-listen is conditional - on most records the layer "
                "returns immediately.",
                size=12, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_footer(s, 16, total_slides)

    # --- Slide 17: agreement + targeted modes ---
    s = add_blank(p); add_bg(s)
    add_header(s, "behaviour", "Agreement drives re-listen activity")
    img1 = FIG / "agreement_distribution.png"
    img2 = FIG / "targeted_mode_distribution.png"
    if img1.exists():
        s.shapes.add_picture(str(img1), Inches(0.4), Inches(1.7),
                             width=Inches(6.3))
    if img2.exists():
        s.shapes.add_picture(str(img2), Inches(6.7), Inches(1.7),
                             width=Inches(6.3))
    add_textbox(s, Inches(0.6), Inches(6.7), Inches(12), Inches(0.5),
                "Most records are unanimous on email and phone -> the targeted "
                "re-listen is skipped on the majority of the dataset.",
                size=13, color=TEXT_LIGHT, align=PP_ALIGN.CENTER)
    add_footer(s, 17, total_slides)

    # --- Slide 18: Ablation takeaways ---
    s = add_blank(p); add_bg(s)
    add_header(s, "ablation takeaways", "What each component is actually buying")
    add_bullets(s, Inches(0.6), Inches(1.7), Inches(12), Inches(5),
                [
                    "STT choice: dual-STT peaks at 29/30; single-Voxtral peaks at "
                    "28; single-Scribe plateaus at 26",
                    "SpeechLM (audio LLM): not strictly additive on its own - only "
                    "pays off when the targeted re-listen is enabled",
                    "Targeted re-listen: clearest gain on Voxtral-only (26 -> 28); "
                    "partly redundant once strict-majority kicks in",
                    "Whisper timestamper: needed by Voxtral path; modest cost; "
                    "small effect once candidates already agree",
                    "Bottom line: the gain over a transcript-only baseline "
                    "(26 -> 29) comes from THREE things working together, "
                    "not one component",
                ], size=18)
    add_footer(s, 18, total_slides)

    # --- Slide 19: Error analysis + conclusion ---
    s = add_blank(p); add_bg(s)
    add_header(s, "error analysis & conclusion", "The one residual error mode")
    add_bullets(s, Inches(0.6), Inches(1.7), Inches(12), Inches(2.5),
                [
                    "call_23: emma.andersson@gmail.com - both STTs hear single 'n' "
                    "(Anderson)",
                    "Country detector identifies Swedish, but email piece is also "
                    "'anderson' so Net 4 cannot help",
                    "Fix: Swedish surname suffix prior, or sub-second re-listen on the "
                    "ambiguous consonant",
                ], size=17)
    add_textbox(s, Inches(0.6), Inches(4.7), Inches(12), Inches(0.5),
                "Conclusion", size=22, bold=True, color=PRIMARY)
    add_bullets(s, Inches(0.6), Inches(5.2), Inches(12), Inches(2),
                [
                    "Best run reaches 29/30 full-record matches (96.7%)",
                    "Per-key accuracy: 100% phone, 100% first name, 96.7% email "
                    "+ last name on the best run",
                    "Code, paper, ablation tooling, and reports all reproducible "
                    "from the study/ folder",
                ], size=17)
    add_footer(s, 19, total_slides)

    p.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
