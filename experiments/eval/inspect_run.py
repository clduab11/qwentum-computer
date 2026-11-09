"""Read-only summary of a Core-4 run directory. Prints; writes nothing."""

import json
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
verbose = "--answers" in sys.argv


def jl(name):
    p = run / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


chats, guards, events = jl("chat-records.jsonl"), jl("guard-checks.jsonl"), jl("events.jsonl")
summary = json.loads((run / "run-summary.json").read_text(encoding="utf-8"))
print("files:", sorted(p.name for p in run.iterdir()))
print("complete:", summary["complete"], "elapsed:", summary["elapsed_s"], "manifest:", summary["manifest"]["manifest_sha256"][:16])
print("guard checks:", len(guards), "problems:", [g["problems"] for g in guards if g["problems"]], "states:",
      Counter((g["state"]["inference"]["active_model"], g["state"]["inference"]["gguf_variant"], g["state"]["train"]["is_training_running"]) for g in guards if g["state"]))
print("events:", Counter(e["event"] for e in events))
print("chat attempts:", len(chats), "seqs:", len({c["seq"] for c in chats}), "retries:", sum(c["attempt"] > 1 for c in chats))
print("http:", Counter(c["http_status"] for c in chats), "errors:", [c["transport_error"] for c in chats if c["transport_error"]])
print("response models:", Counter(c.get("response_model") for c in chats))
keys = Counter(tuple(sorted(c["request"])) for c in chats)
print("request key sets:", keys)
samp = Counter(tuple((k, c["request"][k]) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed")) for c in chats)
print("samplers:", samp)
print("thinking:", Counter(json.dumps(c["request"]["chat_template_kwargs"]) for c in chats), "header names:", Counter(tuple(c["request_header_names"]) for c in chats))
by = {}
for c in chats:
    by.setdefault(c["task"], []).append(c)
for t, cs in by.items():
    print(f"\n== {t}: n={len(cs)} max_tokens={Counter(c['request']['max_tokens'] for c in cs)} stop={Counter(json.dumps(c['request'].get('stop')) for c in cs)}")
    print("   finish:", Counter(c.get("finish_reason") for c in cs), "reasoning_present:", sum(bool(c.get("reasoning_content_present")) for c in cs),
          "reasoning_tokens:", Counter(c.get("reasoning_tokens") for c in cs))
    print("   prompt_tokens:", [c.get("prompt_tokens") for c in cs], "completion_tokens:", [c.get("completion_tokens") for c in cs])
    print("   elapsed_s:", [c["elapsed_s"] for c in cs], "dropped stops:", Counter(json.dumps(c["dropped_stop_sequences"]) for c in cs))
    print("   turns:", Counter((sum(m['role'] == 'user' for m in c['request']['messages']), sum(m['role'] == 'assistant' for m in c['request']['messages'])) for c in cs))
for t, info in summary["tasks"].items():
    print(f"\n## harness results {t}: requests={info['requests']} truncated={info['truncated']}")
    for ht, metrics in (info["results"] or {}).items():
        print("  ", ht, {k: v for k, v in metrics.items() if not k.endswith("stderr,none") and "," in k and "stderr" not in k})
if verbose:
    for p in sorted(run.glob("harness-samples-*.jsonl")):
        for row in jl(p.name):
            resp = row.get("resps") or row.get("filtered_resps")
            text = resp[0][0] if isinstance(resp, list) and resp and isinstance(resp[0], list) else resp
            tgt = row.get("target")
            mets = {k: row[k] for k in row if k in ("math_verify", "exact_match", "prompt_level_strict_acc", "grader_timeout")}
            print(f"\n--- {row['harness_task']} doc {row.get('doc_id')} target={str(tgt)[:60]!r} filtered={row.get('filtered_resps')!r:.80} {mets}")
            print("   ", (str(text)[-600:]).replace("\n", " | "))
