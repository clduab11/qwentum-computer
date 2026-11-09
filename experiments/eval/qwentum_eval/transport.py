"""Guarded chat-completions model for lm-evaluation-harness v0.4.13.

Subclasses the harness's own `local-chat-completions` model so prompt rendering,
few-shot handling and response parsing stay upstream code. This class only:

* checks the runtime before the first request and every N requests
  (authorized model loaded, no load in progress, no training);
* builds an allowlisted payload with every sampler field explicit;
* forces the per-task generation cap and thinking off;
* sends no credentials and no Unsloth extension fields or headers;
* retries once at most, records every attempt, and stops the run (never
  scores) when a request still fails;
* stops the run on the wall-clock budget or a high truncation rate.
"""

from __future__ import annotations

import copy
import json
import threading
import time

import requests
from lm_eval.models.openai_completions import LocalChatCompletion

from . import pins
from .guard import (
    CHAT_PATH,
    GuardError,
    authorization_problems,
    checked_url,
    read_runtime,
)
from .records import RunRecorder, canonical_json, sha256_bytes, utc_now

ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "frequency_penalty", "seed", "stop", "stream",
        "chat_template_kwargs",
    }
)
REQUIRED_SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens")
SENT_HEADERS = {"Content-Type": "application/json"}
MAX_STOPS = 4  # the harness keeps only the first four stop strings


class RunStopped(RuntimeError):
    """Cooperative stop: the run is incomplete and must not be scored as complete."""


def build_payload(
    *, model: str, messages: list[dict], max_tokens: int, stop: list[str], profile: dict
) -> tuple[dict, list[str]]:
    stops = [s for s in (stop or []) if isinstance(s, str) and s]
    kept, dropped = stops[:MAX_STOPS], stops[MAX_STOPS:]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": int(max_tokens),
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_k": profile["top_k"],
        "min_p": profile["min_p"],
        "presence_penalty": profile["presence_penalty"],
        "frequency_penalty": profile["frequency_penalty"],
        "seed": profile["seed"],
        "chat_template_kwargs": {"enable_thinking": bool(profile["enable_thinking"])},
    }
    if kept:
        payload["stop"] = kept
    validate_payload(payload)
    return payload, dropped


def validate_payload(payload: dict) -> None:
    extra = set(payload) - ALLOWED_PAYLOAD_KEYS
    if extra:
        raise GuardError(f"payload has non-allowlisted keys: {sorted(extra)}")
    missing = [k for k in REQUIRED_SAMPLER_KEYS if k not in payload]
    if missing:
        raise GuardError(f"payload is missing required sampler fields: {missing}")
    if set(payload.get("chat_template_kwargs", {})) - {"enable_thinking"}:
        raise GuardError("chat_template_kwargs may only carry enable_thinking")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and set(m) <= {"role", "content"} for m in messages
    ):
        raise GuardError("messages must be a list of {role, content} dicts")


def summarize_response(body: bytes) -> dict:
    """Fields derived from a raw response for monitoring; the raw text is kept too."""
    try:
        data = json.loads(body)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return {
            "response_model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "content_chars": len(msg.get("content") or ""),
            "reasoning_content_present": bool(msg.get("reasoning_content")),
            "tool_calls_present": bool(msg.get("tool_calls")),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "parse_ok": True,
        }
    except Exception as e:  # noqa: BLE001
        return {"parse_ok": False, "parse_error": repr(e)[:300]}


class GuardedChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        *,
        server_base: str,
        recorder: RunRecorder,
        task_label: str,
        gen_cap: int,
        profile: dict | None = None,
        expected_model: str = pins.EXPECTED_MODEL,
        expected_variant: str = pins.EXPECTED_VARIANT,
        guard_every: int = pins.GUARD_EVERY_N_REQUESTS,
        run_started_monotonic: float | None = None,
        max_wall_s: float = pins.MAX_WALL_S_FULL_RUN,
        request_timeout_s: float = pins.REQUEST_TIMEOUT_S,
        max_attempts: int = pins.MAX_RETRIES_TOTAL_ATTEMPTS,
        truncation_min: int = pins.TRUNCATION_CHECK_MIN_RESPONSES,
        truncation_stop_rate: float = pins.TRUNCATION_STOP_RATE,
        retry_sleep_s: float = 2.0,
    ):
        self.server_base = server_base
        self._state_lock = threading.RLock()  # counters/guard shared by concurrent subclasses
        chat_url = self._chat_url()
        super().__init__(
            model=expected_model,
            base_url=chat_url,
            tokenizer_backend=None,
            tokenized_requests=False,
            max_retries=1,  # harness-level retry off; attempts are handled and logged here
            max_gen_toks=gen_cap,
            seed=(profile or pins.PROFILE_D)["seed"],
            timeout=request_timeout_s,
            num_concurrent=1,
            batch_size=1,
        )
        self.recorder = recorder
        self.task_label = task_label
        self.gen_cap = int(gen_cap)
        self.profile = dict(profile or pins.PROFILE_D)
        self.expected_model = expected_model
        self.expected_variant = expected_variant
        self.guard_every = int(guard_every)
        self.run_started = run_started_monotonic or time.perf_counter()
        self.max_wall_s = float(max_wall_s)
        self.request_timeout_s = float(request_timeout_s)
        self.max_attempts = int(max_attempts)
        self.truncation_min = int(truncation_min)
        self.truncation_stop_rate = float(truncation_stop_rate)
        self.retry_sleep_s = float(retry_sleep_s)
        self.session = requests.Session()
        self.session.trust_env = False  # no proxies or netrc credentials
        self.n_requests = 0
        self.n_responses = 0
        self.n_truncated = 0
        self._truncation_checked = False
        self.check_runtime("model_init")

    # -- harness hooks -----------------------------------------------------

    @property
    def header(self) -> dict:  # replaces the harness's "Authorization: Bearer" header
        return dict(SENT_HEADERS)

    @property
    def api_key(self) -> str:
        return ""

    @property
    def eos_string(self):  # server applies the chat template and its own EOS
        return None

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=None, eos=None, **kwargs):
        if not generate:
            raise GuardError("only generation requests are supported through chat completions")
        harness_payload = super()._create_payload(
            messages, generate=True, gen_kwargs=copy.deepcopy(gen_kwargs or {}), seed=seed, eos=None
        )
        stops = list((gen_kwargs or {}).get("until") or [])
        payload, dropped = build_payload(
            model=self.expected_model,
            messages=harness_payload["messages"],
            max_tokens=self.gen_cap,
            stop=stops,
            profile=self.profile,
        )
        payload_meta = {
            "harness_requested_max_tokens": harness_payload.get("max_tokens"),
            "harness_requested_temperature": harness_payload.get("temperature"),
            "dropped_stop_sequences": dropped,
        }
        return payload, payload_meta

    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        payload, meta = self._create_payload(
            self.create_message(messages), generate=generate, gen_kwargs=gen_kwargs
        )
        return self.send(payload, meta)

    async def amodel_call(self, *args, **kwargs):  # concurrency is disabled by design
        raise GuardError("concurrent requests are disabled for this evaluation")

    def loglikelihood(self, requests, **kwargs):
        raise GuardError("loglikelihood is not available through chat completions; not claimed")

    def loglikelihood_rolling(self, requests, **kwargs):
        raise GuardError("perplexity is not available through chat completions; not claimed")

    # -- guard, send, record -------------------------------------------------

    def check_runtime(self, reason: str) -> dict:
        try:
            state = read_runtime(self.session, self.server_base)
        except Exception as e:  # noqa: BLE001
            self.recorder.guard_check(reason=reason, state=None, problems=["status unreadable"], error=repr(e)[:300])
            raise GuardError(f"runtime status unreadable ({reason}): {e!r}") from e
        problems = authorization_problems(state, self.expected_model, self.expected_variant)
        self.recorder.guard_check(reason=reason, state=state, problems=problems, error=None)
        if problems:
            raise GuardError(f"runtime not authorized ({reason}): " + "; ".join(problems))
        return state

    def _check_budgets(self) -> None:
        elapsed = time.perf_counter() - self.run_started
        if elapsed > self.max_wall_s:
            raise RunStopped(f"wall-clock budget exhausted: {elapsed:.0f}s > {self.max_wall_s:.0f}s")
        # Checked once per task (one model instance per task), at exactly N responses.
        if self.n_responses == self.truncation_min and not self._truncation_checked:
            self._truncation_checked = True
            rate = self.n_truncated / self.n_responses
            self.recorder.event("truncation_check", task=self.task_label, responses=self.n_responses,
                                truncated=self.n_truncated, rate=round(rate, 4), threshold=self.truncation_stop_rate)
            if rate > self.truncation_stop_rate:
                raise RunStopped(
                    f"truncation rate {rate:.3f} > {self.truncation_stop_rate} after {self.n_responses} responses"
                )

    def _chat_url(self) -> str:
        """Chat endpoint after the allowlist check; subclasses for other servers override this."""
        return checked_url(self.server_base, "POST", CHAT_PATH)

    def send(self, payload: dict, meta: dict | None = None) -> dict:
        meta = meta or {}
        validate_payload(payload)
        with self._state_lock:
            self._check_budgets()
            if self.n_requests and self.n_requests % self.guard_every == 0:
                self.check_runtime(f"periodic@{self.n_requests}")
            url = self._chat_url()
            seq = self.recorder.next_seq()
            self.n_requests += 1
        derived_common = {
            "task": self.task_label,
            "messages_sha256": sha256_bytes(canonical_json(payload["messages"])),
            **{k: v for k, v in meta.items() if k != "dropped_stop_sequences"},
        }
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            started, utc_start = time.perf_counter(), utc_now()
            status = body = error = None
            try:
                resp = self.session.post(
                    url, json=payload, headers=dict(SENT_HEADERS), timeout=self.request_timeout_s
                )
                status, body = resp.status_code, resp.content
                if not resp.ok:
                    error = f"HTTP {status}"
            except requests.RequestException as e:
                error = repr(e)[:500]
            derived = dict(derived_common)
            if body is not None and error is None:
                derived.update(summarize_response(body))
                if not derived.get("parse_ok"):
                    error = "unparseable response: " + derived.get("parse_error", "")
            with self._state_lock:
                self.recorder.chat_attempt(
                    seq=seq, attempt=attempt, url=url, header_names=list(SENT_HEADERS), payload=payload,
                    dropped_stops=meta.get("dropped_stop_sequences", []), started=started,
                    utc_start=utc_start, status=status, body=body, error=error, derived=derived,
                )
            if error is None:
                if derived.get("response_model") not in (None, self.expected_model):
                    raise GuardError(f"response came from {derived.get('response_model')!r}")
                with self._state_lock:
                    self.n_responses += 1
                    if derived.get("finish_reason") == "length":
                        self.n_truncated += 1
                return json.loads(body)
            last_error = error
            if attempt < self.max_attempts:
                self.recorder.event("retry", seq=seq, attempt=attempt, error=error)
                time.sleep(self.retry_sleep_s)
                self.check_runtime(f"before_retry@seq{seq}")
        raise RunStopped(f"request seq {seq} failed after {self.max_attempts} attempts: {last_error}")
