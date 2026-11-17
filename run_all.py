"""Rerun the whole lab with one command.

    python run_all.py --offline          re-score everything from the stored answers in results/ (no model, no
                                         network, a few seconds) and run the offline tests
    python run_all.py                    everything: render videos, OCR, local models, Claude batches, synthetic
                                         data, fine-tuning, prediction, KuaiMod check, scoring
    python run_all.py --stages ocr,video-local    only some stages, in the order listed below

Stages (in order): render, ocr, video-local, video-siglip, video-haiku, synth, train, predict, kuaimod, score.
Local models need an NVIDIA GPU with 8 GB; Claude stages need POLICY_LAB_ENV_FILE (a file with ANTHROPIC_API_KEY);
the KuaiMod stage needs the dataset under $POLICY_LAB_DATA/kuaimod (see README). Stages that find their results
already present skip the finished work, so an interrupted run can be started again.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES = {
    "render": [["testset/render_video.py"]],
    "ocr": [["-m", "lab.frames"]],
    "video-local": [["-m", "lab.local_vlm", "--model", m] for m in ("qwen35-4b", "qwen3vl-2b")],
    "video-siglip": [["-m", "lab.siglip_frames", "--plan", p] for p in ("frames8", "frames16")],
    "video-haiku": [["-m", "lab.claude_video"]],
    "synth": [["finetune/make_testsets.py"], ["finetune/synth.py", "generate"], ["finetune/synth.py", "build"]],
    "train": [["finetune/train_lora.py", "--base", "Qwen/Qwen3-1.7B", "--out", "qwen3-1.7b-lora"],
              ["finetune/train_lora.py", "--base", "Qwen/Qwen3-0.6B", "--out", "qwen3-0.6b-lora"],
              ["finetune/train_encoder.py"]],
    "predict": [["finetune/predict.py", "--method", m] for m in
                ("qwen3-0.6b-zs", "qwen3-0.6b-lora", "qwen3-1.7b-zs", "qwen3-1.7b-lora", "mmbert", "haiku", "haiku-400")],
    "kuaimod": [["-m", "lab.kuaimod", "sample"], ["-m", "lab.kuaimod", "local"], ["-m", "lab.kuaimod", "haiku"]],
    "score": [["-m", "lab.score"], ["finetune/score_text.py"], ["finetune/compare.py"], ["-m", "lab.visual_evidence"], ["-m", "lab.kuaimod", "score"], ["-m", "lab.report"]],
}


def run(args: list[str]) -> None:
    print(">>", " ".join(args), flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"})
    if r.returncode:
        raise SystemExit(f"stage step failed: {' '.join(args)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--stages", default="")
    a = ap.parse_args(argv)
    if a.offline:
        for step in [["-m", "lab.score"], ["-m", "lab.visual_evidence"], ["finetune/score_text.py"], ["finetune/compare.py"],
                     ["-m", "lab.report"],
                     ["-m", "unittest", "discover", "-s", "tests", "-v"]]:
            run(step)
        return 0
    names = a.stages.split(",") if a.stages else list(STAGES)
    for n in names:
        for step in STAGES[n]:
            run(step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
