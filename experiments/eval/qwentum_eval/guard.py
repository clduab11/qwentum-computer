"""Runtime guard for the qwentum-computer direct-access exception.

Only three endpoints may ever be touched: the two read-only status GETs and
chat completions. Load, unload and every other path raise before any network
I/O. A request is allowed only when the authorized model is active, nothing is
loading, and no training run holds the card.
"""

from __future__ import annotations

from urllib.parse import urlsplit

CHAT_PATH = "/v1/chat/completions"
INFERENCE_STATUS_PATH = "/api/inference/status"
TRAIN_STATUS_PATH = "/api/train/status"

ALLOWED_CALLS = frozenset(
    {
        ("GET", INFERENCE_STATUS_PATH),
        ("GET", TRAIN_STATUS_PATH),
        ("POST", CHAT_PATH),
    }
)
ALLOWED_HOSTS = frozenset({"127.0.0.1"})

# Keys copied from the status responses into records (not the full payload).
STATUS_KEYS = ("active_model", "gguf_variant", "loading", "is_loading", "context_length")
TRAIN_KEYS = ("is_training_running", "phase", "job_id")


class GuardError(RuntimeError):
    """The runtime is not in the authorized state; no request may be sent."""


def checked_url(base_url: str, method: str, path: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme != "http" or parts.hostname not in ALLOWED_HOSTS or parts.port is None:
        raise GuardError(f"base URL {base_url!r} is not an allowed loopback address")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise GuardError(f"base URL {base_url!r} must be a bare origin")
    if (method.upper(), path) not in ALLOWED_CALLS:
        raise GuardError(f"{method.upper()} {path} is not on the endpoint allowlist")
    return f"http://{parts.hostname}:{parts.port}{path}"


def read_runtime(session, base_url: str, timeout: float = 10.0) -> dict:
    inf = session.get(checked_url(base_url, "GET", INFERENCE_STATUS_PATH), timeout=timeout)
    inf.raise_for_status()
    trn = session.get(checked_url(base_url, "GET", TRAIN_STATUS_PATH), timeout=timeout)
    trn.raise_for_status()
    inf_json, trn_json = inf.json(), trn.json()
    return {
        "inference": {k: inf_json.get(k) for k in STATUS_KEYS},
        "train": {k: trn_json.get(k) for k in TRAIN_KEYS},
    }


def authorization_problems(state: dict, expected_model: str, expected_variant: str) -> list[str]:
    inf, trn = state.get("inference", {}), state.get("train", {})
    problems = []
    if inf.get("active_model") != expected_model:
        problems.append(f"active_model is {inf.get('active_model')!r}, expected {expected_model!r}")
    if inf.get("gguf_variant") != expected_variant:
        problems.append(f"gguf_variant is {inf.get('gguf_variant')!r}, expected {expected_variant!r}")
    if inf.get("loading") or inf.get("is_loading"):
        problems.append("a model load is in progress")
    if trn.get("is_training_running") is not False:
        problems.append(f"is_training_running is {trn.get('is_training_running')!r}, expected False")
    return problems


def require_authorized(state: dict, expected_model: str, expected_variant: str) -> None:
    problems = authorization_problems(state, expected_model, expected_variant)
    if problems:
        raise GuardError("; ".join(problems))
