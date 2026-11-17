"""Claude API calls through the Message Batches API (half price; results usually within an hour).

The key is read from the file named by POLICY_LAB_ENV_FILE (a line ANTHROPIC_API_KEY=...). Environment
variables named ANTHROPIC_* are removed from this process first, so credentials that belong to other tools are
never picked up. The key is never printed or written anywhere.

Every batch is logged to results/claude_batches/<tag>.jsonl (custom id, answer text, token usage) and its cost
is added to results/claude_costs.json, using the batch price of claude-haiku-5-5 (USD 0.05 per million input
tokens, 0.25 per million output tokens).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from lab.common import RESULTS, ROOT, read_jsonl, write_jsonl

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.05 / 1e6, 0.25 / 1e6  # batch prices, USD per token
COSTS = RESULTS / "claude_costs.json"


def _read_key() -> str:
    env_file = Path(os.environ.get("POLICY_LAB_ENV_FILE", ROOT / ".env")).resolve()
    if not env_file.exists():
        raise SystemExit(f"API key file not found: {env_file} (set POLICY_LAB_ENV_FILE)")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ANTHROPIC_API_KEY missing in the key file")


@lru_cache(maxsize=1)
def client():
    key = _read_key()
    for k in [k for k in os.environ if k.startswith("ANTHROPIC_")]:
        del os.environ[k]
    import anthropic

    return anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com", max_retries=4)


def jpeg_block(img_array, quality: int = 85) -> dict:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=quality)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}


def params(content: list[dict], max_tokens: int = 400, system: str | None = None) -> dict:
    p = {"model": MODEL, "max_tokens": max_tokens, "thinking": {"type": "disabled"},
         "messages": [{"role": "user", "content": content}]}
    if system:
        p["system"] = system
    return p


def add_cost(tag: str, inp: int, out: int) -> float:
    c = inp * PRICE_IN + out * PRICE_OUT
    data = json.loads(COSTS.read_text(encoding="utf-8")) if COSTS.exists() else {"runs": [], "total_usd": 0.0}
    data["runs"].append({"tag": tag, "input_tokens": inp, "output_tokens": out, "usd": round(c, 4),
                         "time": time.strftime("%Y-%m-%d %H:%M")})
    data["total_usd"] = round(sum(r["usd"] for r in data["runs"]), 4)
    COSTS.parent.mkdir(parents=True, exist_ok=True)
    COSTS.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return c


def run_batch(tag: str, requests: dict[str, dict], poll_s: int = 30) -> dict[str, dict]:
    """requests: custom_id -> params. Returns custom_id -> {"text", "input_tokens", "output_tokens", "stop_reason"}.
    Already-logged custom ids under the same tag are not sent again."""
    log = RESULTS / "claude_batches" / f"{tag}.jsonl"
    have = {r["custom_id"]: r for r in read_jsonl(log) if r.get("type") == "succeeded"}
    todo = {k: v for k, v in requests.items() if k not in have}
    # one upload stays well under the 256 MB batch limit (image requests are large)
    chunks, cur, size = [], {}, 0
    for k, v in todo.items():
        n = len(json.dumps(v))
        if cur and size + n > 40_000_000:
            chunks.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += n
    if cur:
        chunks.append(cur)
    for part in chunks:
        have.update(_one_batch(tag, part, log, poll_s))
    return have


def _one_batch(tag: str, todo: dict[str, dict], log: Path, poll_s: int) -> dict[str, dict]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    have = {}
    if todo:
        batch = client().messages.batches.create(requests=[
            Request(custom_id=k, params=MessageCreateParamsNonStreaming(**v)) for k, v in todo.items()])
        print(f"[{tag}] batch {batch.id}: {len(todo)} requests", flush=True)
        t0 = time.time()
        while True:
            b = client().messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(poll_s)
        print(f"[{tag}] ended after {time.time() - t0:.0f} s: {b.request_counts}", flush=True)
        rows, tin, tout = [], 0, 0
        for res in client().messages.batches.results(batch.id):
            row = {"custom_id": res.custom_id, "type": res.result.type, "batch_id": batch.id}
            if res.result.type == "succeeded":
                msg = res.result.message
                row.update(text="".join(b.text for b in msg.content if b.type == "text"),
                           input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                           stop_reason=msg.stop_reason)
                tin += msg.usage.input_tokens
                tout += msg.usage.output_tokens
            else:
                row["error"] = str(getattr(res.result, "error", ""))
            rows.append(row)
        write_jsonl(log, rows)
        usd = add_cost(tag, tin, tout)
        print(f"[{tag}] tokens in {tin}, out {tout}, USD {usd:.4f}", flush=True)
        have.update({r["custom_id"]: r for r in rows if r["type"] == "succeeded"})
    return have
