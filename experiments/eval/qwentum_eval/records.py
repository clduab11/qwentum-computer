"""Append-only request/response records.

Each run directory is created exclusively and each log file is opened in
exclusive-create mode, so an existing record can never be overwritten or
resumed into silently. Every line is flushed and fsynced. Request headers are
recorded by name only; no credential is ever sent or stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class JsonlLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "x", encoding="utf-8", newline="\n")  # noqa: SIM115
        self.count = 0

    def write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class RunRecorder:
    """Owns one run directory: chat attempts, guard checks and run events."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.chat = JsonlLog(self.run_dir / "chat-records.jsonl")
        self.guard = JsonlLog(self.run_dir / "guard-checks.jsonl")
        self.events = JsonlLog(self.run_dir / "events.jsonl")
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def chat_attempt(
        self,
        *,
        seq: int,
        attempt: int,
        url: str,
        header_names: list[str],
        payload: dict,
        dropped_stops: list[str],
        started: float,
        utc_start: str,
        status: int | None,
        body: bytes | None,
        error: str | None,
        derived: dict,
    ) -> dict:
        req_bytes = canonical_json(payload)
        rec = {
            "seq": seq,
            "attempt": attempt,
            "utc_start": utc_start,
            "utc_end": utc_now(),
            "elapsed_s": round(time.perf_counter() - started, 4),
            "method": "POST",
            "url": url,
            "request_header_names": sorted(header_names),
            "request": payload,
            "request_sha256": sha256_bytes(req_bytes),
            "dropped_stop_sequences": dropped_stops,
            "http_status": status,
            "response_text": body.decode("utf-8", errors="replace") if body is not None else None,
            "response_sha256": sha256_bytes(body) if body is not None else None,
            "transport_error": error,
            **derived,
        }
        self.chat.write(rec)
        return rec

    def guard_check(self, *, reason: str, state: dict | None, problems: list[str], error: str | None) -> None:
        self.guard.write(
            {"utc": utc_now(), "reason": reason, "state": state, "problems": problems, "error": error}
        )

    def event(self, kind: str, **data) -> None:
        self.events.write({"utc": utc_now(), "event": kind, **data})

    def close(self) -> None:
        for log in (self.chat, self.guard, self.events):
            log.close()
