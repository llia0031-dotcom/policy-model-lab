"""Collect the score tables and training logs into results/RESULTS.md (no model, no network).

Run: python -m lab.report
"""
from __future__ import annotations

import json
import sys

from lab.common import RESULTS


def main() -> int:
    parts = ["# All results", "", "Built by `python -m lab.report` from the files in `results/`. "
             "Method and predictions: `eval/PREREGISTRATION.md`. Reading guide: `eval/EVALUATION.md`.", ""]
    for name in ("video_summary.md", "visual_evidence.md", "text_summary.md", "text_paired.md", "kuaimod_summary.md"):
        p = RESULTS / name
        if p.exists():
            body = p.read_text(encoding="utf-8").splitlines()
            parts += ["#" + body[0]] + body[1:] + [""]
    logs = sorted((RESULTS / "train").glob("*.json")) if (RESULTS / "train").exists() else []
    if logs:
        parts += ["## Training runs", "", "| run | examples | trainable / total parameters | epochs (train loss, validation loss) | minutes | peak GB |", "|---|---|---|---|---|---|"]
        for p in logs:
            d = json.loads(p.read_text(encoding="utf-8"))
            ep = "; ".join(f"{e['epoch']}: {e['train_loss']}, {e['val_loss']}" for e in d["epochs"])
            tp = f"{d['trainable_params']:,} / {d['total_params']:,}" if "trainable_params" in d else "all"
            parts.append(f"| {p.stem} | {d['train_examples']} | {tp} | {ep} | {d.get('minutes')} | {d.get('peak_gb', '–')} |")
        parts.append("")
    costs = RESULTS / "claude_costs.json"
    if costs.exists():
        c = json.loads(costs.read_text(encoding="utf-8"))
        parts += ["## Claude API cost (batch prices)", "", "| batch | input tokens | output tokens | USD |", "|---|---|---|---|"]
        parts += [f"| {r['tag']} | {r['input_tokens']:,} | {r['output_tokens']:,} | {r['usd']:.4f} |" for r in c["runs"]]
        parts += [f"| **total** | | | **{c['total_usd']:.4f}** |", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {RESULTS / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
