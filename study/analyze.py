"""Aggregate every report under results/ + study/results/ into a single
ablation table, then render the figures used by the paper.

Outputs:
    study/data/runs.csv               one row per report (config + accuracy + timings)
    study/data/per_key.csv            per-key accuracy, one row per (run, key)
    study/data/mismatches.csv         per-mismatch detail
    study/figures/full_match_by_config.png
    study/figures/per_key_accuracy.png
    study/figures/timing_breakdown.png
    study/figures/agreement_distribution.png
    study/figures/targeted_mode_distribution.png
    study/figures/error_taxonomy.png
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [REPO / "results", REPO / "study" / "results"]
DATA_DIR = REPO / "study" / "data"
FIG_DIR = REPO / "study" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _config_label(report: dict) -> str:
    """Produce a short readable label for a report's configuration."""
    run = report.get("run", {})
    models = run.get("models", {})
    settings = run.get("settings", {})

    trans = models.get("transcription")
    if isinstance(trans, str):
        trans = [trans]
    trans = trans or []

    short_trans: list[str] = []
    for t in trans:
        if "voxtral" in t.lower():
            short_trans.append("Voxtral")
        elif "scribe" in t.lower() or "elevenlabs" in t.lower():
            short_trans.append("Scribe")
        elif "whisper" in t.lower():
            short_trans.append("Whisper-STT")
    stt_label = "+".join(short_trans) if short_trans else "?"

    instruct = (models.get("instruct") or "").split(" ")[0]
    has_audio_llm = bool(instruct) and "voxtral" in instruct.lower()

    timestamp = (models.get("timestamp") or "")
    has_whisper = "whisper" in timestamp.lower()

    # Older reports omit the key entirely - those runs predate Layer 4.5,
    # so an absent key means the layer wasn't part of the pipeline yet.
    use_targeted = bool(settings.get("use_targeted_extraction", False))
    targeted_str = "T+" if use_targeted else "T-"

    audio_llm_str = "AL+" if has_audio_llm else "AL-"
    whisper_str = "W+" if has_whisper else "W-"
    return f"{stt_label} | {audio_llm_str} | {targeted_str} | {whisper_str}"


def _config_key(report: dict) -> str:
    """A stable canonical key per (STT-set, audio-LLM, targeted, whisper) cell."""
    run = report.get("run", {})
    models = run.get("models", {})
    settings = run.get("settings", {})
    trans = models.get("transcription")
    if isinstance(trans, str):
        trans = [trans]
    trans = sorted(trans or [])
    instruct = (models.get("instruct") or "").split(" ")[0]
    use_audio_llm = bool(instruct)
    timestamp = (models.get("timestamp") or "")
    has_whisper = "whisper" in timestamp.lower()
    targeted = settings.get("use_targeted_extraction", False)
    return json.dumps({
        "stt": trans,
        "audio_llm": use_audio_llm,
        "whisper": has_whisper,
        "targeted": bool(targeted),
    }, sort_keys=True)


def _avg_timings(report: dict) -> dict[str, float]:
    keys = ["transcription_s", "country_s", "candidate_1_s",
            "candidate_2_s", "targeted_s", "reconcile_s", "timestamp_s"]
    sums: dict[str, float] = {k: 0.0 for k in keys}
    counts: dict[str, int] = {k: 0 for k in keys}
    for rec in report.get("recordings", []):
        for k, v in (rec.get("timings") or {}).items():
            if k in sums and v is not None:
                sums[k] += float(v)
                counts[k] += 1
    return {k: (sums[k] / counts[k]) if counts[k] else 0.0 for k in keys}


def load_reports() -> list[dict]:
    out = []
    for rdir in RESULTS_DIRS:
        if not rdir.exists():
            continue
        for path in sorted(rdir.glob("report_*.json")):
            try:
                with open(path) as f:
                    report = json.load(f)
            except Exception:
                continue
            if not isinstance(report, dict):
                continue
            evalblk = report.get("evaluation")
            if not evalblk:
                # Skip incomplete runs
                continue
            report["_path"] = str(path.relative_to(REPO))
            out.append(report)
    return out


def write_runs_csv(reports: list[dict]) -> None:
    cols = [
        "report", "label", "config_key", "stt", "audio_llm",
        "whisper_ts", "targeted",
        "num_llm_votes", "num_instruct_votes", "num_targeted_votes",
        "n_records", "full_match", "first_name", "last_name", "email",
        "phone_number", "transcription_s", "country_s", "candidate_1_s",
        "candidate_2_s", "targeted_s", "reconcile_s", "timestamp_s",
    ]
    rows = []
    for r in reports:
        run = r.get("run", {})
        models = run.get("models", {})
        settings = run.get("settings", {})
        trans = models.get("transcription")
        if isinstance(trans, str):
            trans = [trans]
        trans = trans or []
        evalblk = r.get("evaluation", {}) or {}
        per_key = evalblk.get("per_key_correct", {}) or {}
        timings = _avg_timings(r)
        instruct = (models.get("instruct") or "").split(" ")[0]
        rows.append({
            "report": r["_path"],
            "label": _config_label(r),
            "config_key": _config_key(r),
            "stt": "+".join(["V" if "voxtral" in t.lower() else
                              "S" if "scribe" in t.lower() else "?"
                              for t in trans]),
            "audio_llm": int(bool(instruct)),
            "whisper_ts": int("whisper" in (models.get("timestamp") or "").lower()),
            "targeted": int(bool(settings.get("use_targeted_extraction", False))),
            "num_llm_votes": settings.get("num_llm_votes", ""),
            "num_instruct_votes": settings.get("num_instruct_votes", ""),
            "num_targeted_votes": settings.get("num_targeted_votes", ""),
            "n_records": evalblk.get("recordings_evaluated", 0),
            "full_match": evalblk.get("full_match", 0),
            "first_name": per_key.get("first_name", 0),
            "last_name": per_key.get("last_name", 0),
            "email": per_key.get("email", 0),
            "phone_number": per_key.get("phone_number", 0),
            **{k: round(v, 3) for k, v in timings.items()},
        })
    out_path = DATA_DIR / "runs.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out_path}")


def write_per_key_csv(reports: list[dict]) -> None:
    out_path = DATA_DIR / "per_key.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report", "label", "key", "correct", "total", "accuracy"])
        for r in reports:
            evalblk = r.get("evaluation", {}) or {}
            per_key_correct = evalblk.get("per_key_correct", {}) or {}
            per_key_total = evalblk.get("per_key_total", {}) or {}
            label = _config_label(r)
            for k in ["first_name", "last_name", "email", "phone_number"]:
                c = per_key_correct.get(k, 0)
                t = per_key_total.get(k, 0)
                acc = (c / t) if t else 0.0
                w.writerow([r["_path"], label, k, c, t, f"{acc:.4f}"])
    print(f"wrote {out_path}")


def write_mismatches_csv(reports: list[dict]) -> None:
    out_path = DATA_DIR / "mismatches.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["report", "label", "id", "key", "expected", "got"])
        for r in reports:
            evalblk = r.get("evaluation", {}) or {}
            label = _config_label(r)
            for m in evalblk.get("mismatches", []) or []:
                exp = m.get("expected")
                if isinstance(exp, list):
                    exp = " | ".join(exp)
                w.writerow([r["_path"], label, m.get("id"), m.get("key"),
                             exp, m.get("got")])
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot helpers (no emojis, no fancy unicode, ASCII only)
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _aggregate_by_config(reports: list[dict]) -> dict[str, dict]:
    """Average runs that share the same config_key. Returns dict keyed on
    config_key, with values: list of full-match counts, per-key, label, n."""
    bucket: dict[str, dict] = defaultdict(lambda: {
        "label": "",
        "n": 0,
        "full_match": [],
        "per_key": defaultdict(list),
        "n_records": [],
    })
    for r in reports:
        ck = _config_key(r)
        evalblk = r.get("evaluation", {}) or {}
        bucket[ck]["label"] = _config_label(r)
        bucket[ck]["n"] += 1
        bucket[ck]["full_match"].append(evalblk.get("full_match", 0))
        bucket[ck]["n_records"].append(evalblk.get("recordings_evaluated", 0))
        for k, v in (evalblk.get("per_key_correct") or {}).items():
            bucket[ck]["per_key"][k].append(v)
    return bucket


def plot_full_match_by_config(reports: list[dict]) -> None:
    bucket = _aggregate_by_config(reports)
    items = []
    for ck, info in bucket.items():
        n_records = max(info["n_records"]) if info["n_records"] else 30
        full = info["full_match"]
        mean = float(np.mean(full)) if full else 0.0
        std = float(np.std(full)) if len(full) > 1 else 0.0
        items.append({
            "label": info["label"],
            "mean": mean,
            "std": std,
            "n_runs": info["n"],
            "n_records": n_records,
        })
    items.sort(key=lambda x: x["mean"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ys = np.arange(len(items))
    means = [it["mean"] for it in items]
    stds = [it["std"] for it in items]
    labels = [f"{it['label']}  (n={it['n_runs']})" for it in items]

    bars = ax.barh(ys, means, xerr=stds, color="#4c72b0", edgecolor="black",
                    error_kw={"ecolor": "black", "capsize": 3})
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Full-record matches (out of 30)")
    ax.set_xlim(0, 33)
    n_records_max = max(it["n_records"] for it in items)
    ax.set_title(f"Full-record accuracy across configurations (n={n_records_max} recordings)")
    for bar, it in zip(bars, items):
        pct = 100.0 * it["mean"] / max(it["n_records"], 1)
        x = bar.get_width() + max(0.4, it["std"]) + 0.4
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{it['mean']:.1f} ({pct:.1f}%)",
                va="center", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "full_match_by_config.png")


def plot_per_key_accuracy(reports: list[dict]) -> None:
    bucket = _aggregate_by_config(reports)
    keys = ["first_name", "last_name", "email", "phone_number"]
    pretty = {"first_name": "first name", "last_name": "last name",
              "email": "email", "phone_number": "phone number"}
    items = []
    for ck, info in bucket.items():
        per_key_means = {}
        n_records = max(info["n_records"]) if info["n_records"] else 30
        for k in keys:
            vals = info["per_key"].get(k, [])
            per_key_means[k] = float(np.mean(vals)) if vals else 0.0
        items.append({"label": info["label"], "vals": per_key_means,
                       "n": info["n"], "n_records": n_records})
    items.sort(key=lambda x: -sum(x["vals"].values()))

    fig, ax = plt.subplots(figsize=(11, 6))
    n_cfg = len(items)
    width = 0.18
    xs = np.arange(n_cfg)
    colors = ["#4c72b0", "#dd8452", "#55a467", "#c44e52"]
    for i, k in enumerate(keys):
        ys = [100.0 * it["vals"].get(k, 0) / max(it["n_records"], 1) for it in items]
        ax.bar(xs + (i - 1.5) * width, ys, width, label=pretty[k],
                color=colors[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([it["label"] for it in items], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylabel("per-key accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-key accuracy across configurations")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "per_key_accuracy.png")


def plot_timing_breakdown(reports: list[dict]) -> None:
    label_to_avg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    label_to_n: dict[str, int] = defaultdict(int)
    for r in reports:
        label = _config_label(r)
        timings = _avg_timings(r)
        for k, v in timings.items():
            if v > 0:
                label_to_avg[label][k].append(v)
        label_to_n[label] += 1

    rows = []
    for label, d in label_to_avg.items():
        rows.append({
            "label": label,
            "n": label_to_n[label],
            "transcription": float(np.mean(d.get("transcription_s", [0]))) if d.get("transcription_s") else 0.0,
            "timestamp": float(np.mean(d.get("timestamp_s", [0]))) if d.get("timestamp_s") else 0.0,
            "country": float(np.mean(d.get("country_s", [0]))) if d.get("country_s") else 0.0,
            "candidate_1": float(np.mean(d.get("candidate_1_s", [0]))) if d.get("candidate_1_s") else 0.0,
            "candidate_2": float(np.mean(d.get("candidate_2_s", [0]))) if d.get("candidate_2_s") else 0.0,
            "targeted": float(np.mean(d.get("targeted_s", [0]))) if d.get("targeted_s") else 0.0,
            "reconcile": float(np.mean(d.get("reconcile_s", [0]))) if d.get("reconcile_s") else 0.0,
        })
    if not rows:
        return
    rows.sort(key=lambda x: sum([x["transcription"], x["timestamp"],
                                  x["country"], x["candidate_1"],
                                  x["candidate_2"], x["targeted"],
                                  x["reconcile"]]))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    components = [
        ("transcription", "#4c72b0"),
        ("timestamp",      "#88aacc"),
        ("country",         "#dd8452"),
        ("candidate_1",     "#55a467"),
        ("candidate_2",     "#c44e52"),
        ("targeted",        "#8172b2"),
        ("reconcile",       "#937860"),
    ]
    n_rows = len(rows)
    ys = np.arange(n_rows)
    left = np.zeros(n_rows)
    for comp, color in components:
        widths = np.array([r[comp] for r in rows])
        ax.barh(ys, widths, left=left, label=comp, color=color,
                edgecolor="black", linewidth=0.4)
        left += widths
    totals = left
    for i, (y, t, r) in enumerate(zip(ys, totals, rows)):
        ax.text(t + 0.3, y, f"{t:.1f}s", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} (n={r['n']})" for r in rows], fontsize=9)
    ax.set_xlabel("avg seconds per recording")
    ax.set_title("Per-stage latency by configuration")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "timing_breakdown.png")


def plot_agreement_distribution(reports: list[dict]) -> None:
    """Across the BEST run (most full matches), what fraction of records were
    unanimous on each key vs. had some disagreement?"""
    if not reports:
        return
    best = max(reports, key=lambda r: (r.get("evaluation") or {}).get("full_match", 0))
    keys = ["first_name", "last_name", "email", "phone_number"]
    pretty = {"first_name": "first name", "last_name": "last name",
              "email": "email", "phone_number": "phone number"}
    counts = {k: {"unanimous": 0, "disagreement": 0, "single": 0}
              for k in keys}
    for rec in best.get("recordings", []):
        ag = rec.get("agreement") or {}
        for k in keys:
            entry = ag.get(k) or {}
            n_cands = entry.get("n_candidates", 0)
            n_unique = entry.get("n_unique", 0)
            if n_cands < 2:
                counts[k]["single"] += 1
            elif n_unique == 1:
                counts[k]["unanimous"] += 1
            else:
                counts[k]["disagreement"] += 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.55
    xs = np.arange(len(keys))
    una = [counts[k]["unanimous"] for k in keys]
    dis = [counts[k]["disagreement"] for k in keys]
    sng = [counts[k]["single"] for k in keys]
    ax.bar(xs, una, width, label="unanimous", color="#55a467",
            edgecolor="black")
    ax.bar(xs, dis, width, bottom=una, label="some disagreement",
            color="#c44e52", edgecolor="black")
    ax.bar(xs, sng, width, bottom=np.array(una) + np.array(dis),
            label="single candidate", color="#888888", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty[k] for k in keys])
    ax.set_ylabel("recordings")
    n_records = len(best.get("recordings", []))
    ax.set_title(f"Inter-candidate agreement on the best run "
                 f"(n={n_records} recordings)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "agreement_distribution.png")


def plot_targeted_mode_distribution(reports: list[dict]) -> None:
    """For the BEST run with targeted enabled, count how often Layer 4.5
    actually fired per field (and in what mode)."""
    candidate = None
    best_score = -1
    for r in reports:
        if not r.get("run", {}).get("settings", {}).get("use_targeted_extraction"):
            continue
        score = r.get("evaluation", {}).get("full_match", 0)
        if score > best_score:
            best_score = score
            candidate = r
    if candidate is None:
        return

    counts: dict[str, Counter] = {"email": Counter(), "phone_number": Counter()}
    for rec in candidate.get("recordings", []):
        targeted = rec.get("targeted") or {}
        em = (targeted.get("email_meta") or {}).get("mode", "skipped")
        ph = (targeted.get("phone_meta") or {}).get("mode", "skipped")
        counts["email"][em] += 1
        counts["phone_number"][ph] += 1

    modes = ["cropped", "full_audio", "skipped_unanimous", "skipped"]
    colors = {"cropped": "#c44e52", "full_audio": "#dd8452",
              "skipped_unanimous": "#55a467", "skipped": "#888888"}
    pretty = {"email": "email", "phone_number": "phone number"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(2)
    width = 0.65
    bottom = np.zeros(2)
    for m in modes:
        ys = np.array([counts["email"].get(m, 0),
                        counts["phone_number"].get(m, 0)])
        ax.bar(xs, ys, width, bottom=bottom, label=m,
                color=colors[m], edgecolor="black", linewidth=0.5)
        bottom += ys
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty["email"], pretty["phone_number"]])
    ax.set_ylabel("recordings")
    n_records = len(candidate.get("recordings", []))
    ax.set_title(f"Layer 4.5 (targeted re-listen) firing modes "
                 f"(n={n_records}, best run)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "targeted_mode_distribution.png")


def plot_error_taxonomy(reports: list[dict]) -> None:
    """Across all runs, count error types per key."""
    err_counts = defaultdict(int)
    for r in reports:
        for m in (r.get("evaluation") or {}).get("mismatches", []) or []:
            err_counts[m.get("key", "?")] += 1

    keys = ["first_name", "last_name", "email", "phone_number"]
    pretty = {"first_name": "first name", "last_name": "last name",
              "email": "email", "phone_number": "phone number"}
    counts = [err_counts.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(keys))
    bars = ax.bar(xs, counts, width=0.55, color="#c44e52", edgecolor="black")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(c),
                ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty[k] for k in keys])
    ax.set_ylabel("total errors across all runs")
    ax.set_title("Error count by field (aggregated over every recorded run)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "error_taxonomy.png")


def main() -> None:
    reports = load_reports()
    print(f"loaded {len(reports)} reports")
    if not reports:
        sys.exit("no reports found")
    write_runs_csv(reports)
    write_per_key_csv(reports)
    write_mismatches_csv(reports)
    plot_full_match_by_config(reports)
    plot_per_key_accuracy(reports)
    plot_timing_breakdown(reports)
    plot_agreement_distribution(reports)
    plot_targeted_mode_distribution(reports)
    plot_error_taxonomy(reports)


if __name__ == "__main__":
    main()
