"""EXP-E-002 vLLM transport, offline against a loopback fake vLLM server. Skips without lm_eval."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

lm_eval = pytest.importorskip("lm_eval")

import test_eval_fakes  # noqa: E402,F401  (puts experiments/eval on sys.path)
from test_eval_fakes import chat_body, read_jsonl  # noqa: E402

from qwentum_eval.guard import GuardError  # noqa: E402
from qwentum_eval.kaggle_transport import VllmGuardedChatCompletion, vllm_url  # noqa: E402
from qwentum_eval.records import RunRecorder  # noqa: E402
from qwentum_eval.transport import RunStopped  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"


class FakeVllm:
    def __init__(self, served=(MODEL,), delay=0.0):
        self.served = list(served)
        self.health = 200
        self.delay = delay
        self.requests = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.lock = threading.Lock()
        self.chat_codes = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                with outer.lock:
                    outer.requests.append(("GET", self.path, None, dict(self.headers)))
                if self.path == "/health":
                    self._send(outer.health, {})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": m} for m in outer.served]})
                else:
                    self._send(404, {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with outer.lock:
                    outer.requests.append(("POST", self.path, body, dict(self.headers)))
                    outer.in_flight += 1
                    outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                    code = outer.chat_codes.pop(0) if outer.chat_codes else 200
                try:
                    if self.path != "/v1/chat/completions":
                        self._send(404, {})
                        return
                    time.sleep(outer.delay)
                    p = json.loads(body)
                    user = [m["content"] for m in p["messages"] if m["role"] == "user"][-1]
                    self._send(code, chat_body(f"echo:{user}", model=MODEL) if code == 200 else {"error": "x"})
                finally:
                    with outer.lock:
                        outer.in_flight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    @property
    def chats(self):
        return [r for r in self.requests if r[0] == "POST"]

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *e):
        self.httpd.shutdown()
        self.httpd.server_close()


class Req:
    """Minimal stand-in for a harness Instance: args = (JsonChatStr-like, gen_kwargs)."""

    class _Ctx:
        def __init__(self, msgs):
            self.prompt = json.dumps(msgs)

    def __init__(self, text, until=None):
        self.args = (self._Ctx([{"role": "user", "content": text}]), {"until": until or ["Q:"], "temperature": 0.9, "max_gen_toks": 99999})


def make(srv, rec, **kw):
    kw.setdefault("retry_sleep_s", 0)
    return VllmGuardedChatCompletion(server_base=srv.base, recorder=rec, task_label="t", gen_cap=128,
                                     expected_model=MODEL, expected_variant="n/a", **kw)


@pytest.fixture
def rec(tmp_path):
    r = RunRecorder(tmp_path / "run")
    yield r
    r.close()


def test_allowlist():
    assert vllm_url("http://127.0.0.1:8000", "POST", "/v1/chat/completions").endswith("/v1/chat/completions")
    for m, p in [("POST", "/v1/completions"), ("POST", "/api/inference/load"), ("GET", "/metrics"), ("POST", "/v1/models")]:
        with pytest.raises(GuardError):
            vllm_url("http://127.0.0.1:8000", m, p)
    for base in ["http://localhost:8000", "http://10.0.0.1:8000", "http://127.0.0.1:8000/v1"]:
        with pytest.raises(GuardError):
            vllm_url(base, "GET", "/health")


def test_guard_requires_exactly_the_pinned_model(rec):
    with FakeVllm(served=(MODEL, "other/model")) as srv:
        with pytest.raises(GuardError, match="expected exactly"):
            make(srv, rec)
        assert srv.chats == []


def test_guard_requires_health(tmp_path):
    with FakeVllm() as srv:
        srv.health = 503
        r = RunRecorder(tmp_path / "h")
        with pytest.raises(GuardError, match="/health"):
            make(srv, r)
        r.close()


def test_concurrent_generate_preserves_order_and_records_everything(rec):
    with FakeVllm(delay=0.2) as srv:
        lm = make(srv, rec, concurrency=16, guard_every=1000)
        reqs = [Req(f"q{i}") for i in range(40)]
        t0 = time.time()
        out = lm.generate_until(reqs)
        wall = time.time() - t0
        assert out == [f"echo:q{i}" for i in range(40)]
        assert srv.max_in_flight > 1, "requests were not sent concurrently"
        assert wall < 40 * 0.2 * 0.6
        recs = read_jsonl(rec.run_dir / "chat-records.jsonl")
        assert len(recs) == 40 and sorted(r["seq"] for r in recs) == list(range(1, 41))
        for raw in srv.chats:
            p = json.loads(raw[2])
            assert p["max_tokens"] == 128 and p["temperature"] == 0 and p["top_k"] == 1
            assert p["presence_penalty"] == 0 and p["chat_template_kwargs"] == {"enable_thinking": False}
            assert p["model"] == MODEL and "authorization" not in {k.lower() for k in raw[3]}


def test_periodic_guard_runs_under_concurrency(rec):
    with FakeVllm() as srv:
        lm = make(srv, rec, concurrency=8, guard_every=10)
        lm.generate_until([Req(f"q{i}") for i in range(35)])
        checks = read_jsonl(rec.run_dir / "guard-checks.jsonl")
        assert sum(c["reason"].startswith("periodic@") for c in checks) == 3


def test_failed_request_stops_the_run(rec):
    with FakeVllm() as srv:
        lm = make(srv, rec, concurrency=4, guard_every=1000)
        srv.chat_codes = [500, 500]
        with pytest.raises(RunStopped):
            lm.generate_until([Req("only")])


def test_likelihood_refused(rec):
    with FakeVllm() as srv:
        lm = make(srv, rec)
        with pytest.raises(GuardError):
            lm.loglikelihood([])
