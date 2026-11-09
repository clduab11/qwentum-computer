"""Loopback fake of the Unsloth endpoints used by the Core-4 evaluation tests.

Not a test module itself (no test functions). Records every request (method, path, headers, body). Unknown paths return 404
and are recorded, so a test can prove the guard never touched them.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "experiments" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

GOOD_STATUS = {
    "active_model": "unsloth/Qwen3.5-0.8B-GGUF",
    "gguf_variant": "Q8_0",
    "loading": [],
    "context_length": 262144,
}
IDLE_TRAIN = {"is_training_running": False, "phase": "idle", "job_id": ""}


def chat_body(content: str, *, model="unsloth/Qwen3.5-0.8B-GGUF", finish="stop", completion_tokens=7) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "reasoning_content": None},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": completion_tokens, "total_tokens": 11 + completion_tokens,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
    }


class FakeState:
    def __init__(self):
        self.status = dict(GOOD_STATUS)
        self.train = dict(IDLE_TRAIN)
        self.requests: list[dict] = []
        self.chat_codes: list[int] = []  # popped per chat request; empty means 200
        self.responder = lambda payload: chat_body("ok")
        self.after_chat = None  # optional hook(state, n_chats)
        self.lock = threading.Lock()

    @property
    def chats(self) -> list[dict]:
        return [r for r in self.requests if r["path"] == "/v1/chat/completions"]


def _handler(state: FakeState):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _record(self, body: bytes | None) -> None:
            with state.lock:
                state.requests.append({
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                    "t": time.time(),
                })

        def do_GET(self):
            self._record(None)
            if self.path == "/api/inference/status":
                self._send(200, state.status)
            elif self.path == "/api/train/status":
                self._send(200, state.train)
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._record(body)
            if self.path != "/v1/chat/completions":
                self._send(404, {"detail": "not found"})
                return
            with state.lock:
                code = state.chat_codes.pop(0) if state.chat_codes else 200
                n = len(state.chats)
            if code != 200:
                self._send(code, {"error": "fake failure"})
            else:
                self._send(200, state.responder(json.loads(body)))
            if state.after_chat:
                state.after_chat(state, n)

    return H


class FakeServer:
    def __init__(self):
        self.state = FakeState()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self.state))
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def last_user_text(payload: dict) -> str:
    users = [m["content"] for m in payload["messages"] if m["role"] == "user"]
    return users[-1] if users else ""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
