"""Capture llama-server sampler/slot lines for a run into a new audit record (read-only on the log).

Usage: capture_backend_audit.py RUN_DIR LOG_PATH
Writes RUN_DIR/backend-audit.json (exclusive create). Only model-serving lines are kept.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

run, log = Path(sys.argv[1]), Path(sys.argv[2])
lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
summary = json.loads((run / "run-summary.json").read_text(encoding="utf-8"))
n_chats = sum(1 for _ in (run / "chat-records.jsonl").open(encoding="utf-8"))

params = []
for i, l in enumerate(lines):
    if "sampler params:" in l:
        params.append(" ".join(x.strip() for x in lines[i + 1 : i + 4]))
last = params[-n_chats:]
fields = Counter()
for p in last:
    for k in ("repeat_penalty", "frequency_penalty", "presence_penalty", "top_k", "top_p", "min_p", "temp", "dry_multiplier"):
        m = re.search(rf"\b{k} = ([-0-9.]+)", p)
        fields[(k, m.group(1) if m else None)] += 1
slots = [l for l in lines if "new prompt, n_ctx_slot" in l][-n_chats:]
releases = [l for l in lines if "stop processing:" in l][-n_chats:]
audit = {
    "utc": datetime.now(UTC).isoformat(),
    "run_id": summary["run_id"],
    "source_log": str(log),
    "source_log_sha256_at_capture": hashlib.sha256(log.read_bytes()).hexdigest(),
    "chat_requests_in_run": n_chats,
    "sampler_param_blocks_in_log": len(params),
    "last_n_blocks_effective_values": {f"{k}={v}": c for (k, v), c in sorted(fields.items())},
    "last_n_sampler_blocks": last,
    "n_ctx_slot_values": dict(Counter(re.search(r"n_ctx_slot = (\d+)", l).group(1) for l in slots)),
    "truncated_flags": dict(Counter(re.search(r"truncated = (\d+)", l).group(1) for l in releases if "truncated" in l)),
    "caveat": "Matched by position (last N blocks), not by request ID. The server had no other clients documented in this window, but that is not proven by the log.",
}
with open(run / "backend-audit.json", "x", encoding="utf-8", newline="\n") as fh:
    json.dump(audit, fh, indent=1)
print(json.dumps({k: audit[k] for k in ("chat_requests_in_run", "sampler_param_blocks_in_log", "last_n_blocks_effective_values", "n_ctx_slot_values", "truncated_flags")}, indent=1))
