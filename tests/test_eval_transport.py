"""Offline tests of the guarded chat transport against a loopback fake server.

These tests need the evaluation environment (.venv-eval with lm-eval 0.4.13).
In the Stim environment they skip; they never contact the real Unsloth service.
"""

import json
import time

import pytest

lm_eval = pytest.importorskip("lm_eval")
requests = pytest.importorskip("requests")

from test_eval_fakes import FakeServer, chat_body, read_jsonl  # noqa: E402

from qwentum_eval import pins  # noqa: E402
from qwentum_eval.guard import GuardError, authorization_problems, checked_url  # noqa: E402
from qwentum_eval.records import RunRecorder  # noqa: E402
from qwentum_eval.transport import (  # noqa: E402
    ALLOWED_PAYLOAD_KEYS,
    GuardedChatCompletion,
    RunStopped,
    build_payload,
    validate_payload,
)

MSGS = [{"role": "user", "content": "Say ok."}]


@pytest.fixture
def server():
    with FakeServer() as s:
        yield s


@pytest.fixture
def recorder(tmp_path):
    r = RunRecorder(tmp_path / "run")
    yield r
    r.close()


def make_lm(server, recorder, **kw):
    kw.setdefault("retry_sleep_s", 0.0)
    return GuardedChatCompletion(server_base=server.base, recorder=recorder, task_label="t", gen_cap=64, **kw)


# ---- guard ---------------------------------------------------------------


def test_endpoint_allowlist_blocks_load_unload_and_other_hosts():
    assert checked_url("http://127.0.0.1:8888", "POST", "/v1/chat/completions").endswith("/v1/chat/completions")
    for method, path in [
        ("POST", "/api/inference/load"),
        ("POST", "/api/inference/unload"),
        ("GET", "/api/inference/load"),
        ("POST", "/v1/completions"),
        ("GET", "/v1/models"),
        ("POST", "/api/train/start"),
    ]:
        with pytest.raises(GuardError):
            checked_url("http://127.0.0.1:8888", method, path)
    for base in ["http://localhost:8888", "https://127.0.0.1:8888", "http://10.0.0.5:8888", "http://127.0.0.1:8888/v1"]:
        with pytest.raises(GuardError):
            checked_url(base, "GET", "/api/inference/status")


@pytest.mark.parametrize(
    "inference,train,expect",
    [
        ({"active_model": "unsloth/Qwen3.5-0.8B-GGUF", "gguf_variant": "Q8_0", "loading": []}, {"is_training_running": False}, []),
        ({"active_model": "HauhauCS/Qwen3.5-4B-Uncensored", "gguf_variant": "Q4_K_M", "loading": []}, {"is_training_running": False}, ["active_model", "gguf_variant"]),
        ({"active_model": "unsloth/Qwen3.5-0.8B-GGUF", "gguf_variant": "Q8_0", "loading": ["x"]}, {"is_training_running": False}, ["load is in progress"]),
        ({"active_model": "unsloth/Qwen3.5-0.8B-GGUF", "gguf_variant": "Q8_0", "loading": []}, {"is_training_running": True}, ["is_training_running"]),
        ({"active_model": None, "gguf_variant": None, "loading": []}, {}, ["active_model", "gguf_variant", "is_training_running"]),
    ],
)
def test_authorization_problems(inference, train, expect):
    problems = authorization_problems({"inference": inference, "train": train}, pins.EXPECTED_MODEL, pins.EXPECTED_VARIANT)
    assert len(problems) == len(expect)
    for needle in expect:
        assert any(needle in p for p in problems), (needle, problems)


def test_wrong_model_refused_before_any_chat_request(server, recorder):
    server.state.status["active_model"] = "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive"
    server.state.status["gguf_variant"] = "Q4_K_M"
    with pytest.raises(GuardError, match="active_model"):
        make_lm(server, recorder)
    assert server.state.chats == []
    paths = {r["path"] for r in server.state.requests}
    assert paths <= {"/api/inference/status", "/api/train/status"}
    checks = read_jsonl(recorder.run_dir / "guard-checks.jsonl")
    assert checks and checks[-1]["problems"]


def test_training_running_refused(server, recorder):
    server.state.train["is_training_running"] = True
    with pytest.raises(GuardError, match="is_training_running"):
        make_lm(server, recorder)
    assert server.state.chats == []


def test_load_in_progress_refused(server, recorder):
    server.state.status["loading"] = [{"model": "something"}]
    with pytest.raises(GuardError, match="load is in progress"):
        make_lm(server, recorder)
    assert server.state.chats == []


def test_periodic_recheck_stops_when_model_changes(server, recorder):
    lm = make_lm(server, recorder, guard_every=3)

    def swap(state, n):
        if n == 3:
            state.status["active_model"] = "other/model"

    server.state.after_chat = swap
    for _ in range(3):
        lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    with pytest.raises(GuardError, match="periodic@3"):
        lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    assert len(server.state.chats) == 3


# ---- payload -------------------------------------------------------------


def test_payload_fields_and_samplers(server, recorder):
    lm = make_lm(server, recorder)
    lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=64, stop=["Q:", "a", "b", "c", "d"], profile=pins.PROFILE_D)[0])
    sent = json.loads(server.state.chats[0]["body"])
    assert set(sent) <= ALLOWED_PAYLOAD_KEYS
    assert sent["model"] == pins.EXPECTED_MODEL
    assert sent["temperature"] == 0 and sent["top_p"] == 1 and sent["top_k"] == 1 and sent["min_p"] == 0
    assert sent["presence_penalty"] == 0 and sent["frequency_penalty"] == 0
    assert sent["max_tokens"] == 64 and sent["seed"] == pins.PROFILE_D["seed"] and sent["stream"] is False
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["stop"] == ["Q:", "a", "b", "c"]
    for forbidden in ("enable_tools", "mcp_enabled", "session_id", "permission_mode", "external_model", "provider", "encrypted_api_key"):
        assert forbidden not in sent
    headers = server.state.chats[0]["headers"]
    assert "authorization" not in headers
    assert not any(h.startswith("x-unsloth") for h in headers)


def test_validate_payload_rejects_extension_fields_and_missing_samplers():
    good, _ = build_payload(model="m", messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)
    validate_payload(good)
    with pytest.raises(GuardError, match="non-allowlisted"):
        validate_payload({**good, "enable_tools": True})
    with pytest.raises(GuardError, match="non-allowlisted"):
        validate_payload({**good, "session_id": "abc"})
    bad = {k: v for k, v in good.items() if k != "top_k"}
    with pytest.raises(GuardError, match="missing required sampler"):
        validate_payload(bad)
    with pytest.raises(GuardError, match="enable_thinking"):
        validate_payload({**good, "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"}})


def test_harness_create_payload_path_uses_cap_and_profile(server, recorder):
    """The harness calls _create_payload with its own gen_kwargs; ours must override them."""
    lm = make_lm(server, recorder)
    payload, meta = lm._create_payload(MSGS, generate=True, gen_kwargs={"until": ["Problem:"], "temperature": 0.7, "max_gen_toks": 9999, "do_sample": True})
    assert payload["max_tokens"] == 64 and payload["temperature"] == 0 and payload["stop"] == ["Problem:"]
    assert meta["harness_requested_max_tokens"] == 9999
    assert "do_sample" not in payload
    with pytest.raises(GuardError):
        lm._create_payload(MSGS, generate=False)
    with pytest.raises(GuardError):
        lm.loglikelihood([])
    with pytest.raises(GuardError):
        lm.loglikelihood_rolling([])


# ---- records -------------------------------------------------------------


def test_every_attempt_recorded_and_write_once(server, recorder, tmp_path):
    lm = make_lm(server, recorder)
    server.state.responder = lambda p: chat_body("The answer is 42.", completion_tokens=5)
    out = lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    assert out["choices"][0]["message"]["content"] == "The answer is 42."
    recs = read_jsonl(recorder.run_dir / "chat-records.jsonl")
    assert len(recs) == 1
    r = recs[0]
    assert r["request"]["messages"] == MSGS and r["http_status"] == 200
    assert json.loads(r["response_text"])["choices"][0]["message"]["content"] == "The answer is 42."
    assert r["finish_reason"] == "stop" and r["completion_tokens"] == 5 and r["transport_error"] is None
    assert r["request_header_names"] == ["Content-Type"]
    assert len(r["request_sha256"]) == 64 and len(r["response_sha256"]) == 64
    with pytest.raises(FileExistsError):
        RunRecorder(recorder.run_dir)


def test_transport_failure_retries_once_then_stops_without_scoring(server, recorder):
    lm = make_lm(server, recorder)
    server.state.chat_codes = [500, 500]
    with pytest.raises(RunStopped, match="failed after 2 attempts"):
        lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    recs = read_jsonl(recorder.run_dir / "chat-records.jsonl")
    assert [r["attempt"] for r in recs] == [1, 2] and all(r["http_status"] == 500 for r in recs)
    assert len(server.state.chats) == 2
    events = read_jsonl(recorder.run_dir / "events.jsonl")
    assert any(e["event"] == "retry" for e in events)
    assert lm.n_responses == 0


def test_single_failure_then_success_is_recorded_as_two_attempts(server, recorder):
    lm = make_lm(server, recorder)
    server.state.chat_codes = [503]
    lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    recs = read_jsonl(recorder.run_dir / "chat-records.jsonl")
    assert [(r["seq"], r["attempt"], r["http_status"]) for r in recs] == [(1, 1, 503), (1, 2, 200)]


def test_response_from_other_model_is_refused(server, recorder):
    lm = make_lm(server, recorder)
    server.state.responder = lambda p: chat_body("x", model="someone/else")
    with pytest.raises(GuardError, match="came from"):
        lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])


# ---- budgets -------------------------------------------------------------


def test_wall_clock_budget_stops_run(server, recorder):
    lm = make_lm(server, recorder, run_started_monotonic=time.perf_counter() - 100, max_wall_s=10)
    with pytest.raises(RunStopped, match="wall-clock"):
        lm.send(build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0])
    assert server.state.chats == []


def test_truncation_rate_stops_run(server, recorder):
    lm = make_lm(server, recorder, truncation_min=4, truncation_stop_rate=0.5, guard_every=1000)
    server.state.responder = lambda p: chat_body("...", finish="length")
    p = build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0]
    for _ in range(4):
        lm.send(p)
    with pytest.raises(RunStopped, match="truncation rate"):
        lm.send(p)
    assert lm.n_truncated == 4


def test_truncation_checked_once_at_threshold_then_never_again(server, recorder):
    """v2 rule: a single check at N responses; later cap hits are counted, never stop the run."""
    lm = make_lm(server, recorder, truncation_min=4, truncation_stop_rate=0.5, guard_every=1000)
    finishes = ["length", "stop", "stop", "length"] + ["length"] * 6  # 2/4 = 0.5 at check: not > 0.5
    it = iter(finishes)
    server.state.responder = lambda p: chat_body("...", finish=next(it))
    p = build_payload(model=pins.EXPECTED_MODEL, messages=MSGS, max_tokens=8, stop=[], profile=pins.PROFILE_D)[0]
    for _ in range(10):
        lm.send(p)
    assert lm.n_responses == 10 and lm.n_truncated == 8
    checks = [e for e in read_jsonl(recorder.run_dir / "events.jsonl") if e["event"] == "truncation_check"]
    assert len(checks) == 1 and checks[0]["responses"] == 4 and checks[0]["rate"] == 0.5


def test_v2_pins():
    assert pins.TRUNCATION_STOP_RATE == 0.50 and pins.TRUNCATION_CHECK_MIN_RESPONSES == 50
    assert pins.TRUNCATION_CHECK_MODE == "once"
    assert pins.MANIFEST_FILE.endswith("eval-EXP-E-001-v2.json")
