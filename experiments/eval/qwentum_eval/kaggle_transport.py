"""EXP-E-002 transport: harness -> vLLM OpenAI server inside a Kaggle notebook.

Subclasses the guarded Unsloth transport so payload building, the field allowlist,
explicit Protocol D samplers, per-task caps, write-once records, retry policy
and budget stops are the same code. Only three things differ:

* the runtime guard reads vLLM's `/health` and `/v1/models` (there is no
  Unsloth status or training endpoint on Kaggle) and requires exactly the one
  pinned model id to be served;
* the endpoint allowlist is those two GETs plus chat completions;
* `generate_until` sends requests concurrently (vLLM batches them), keeping
  result order identical to the harness's request order.

No model is loaded or unloaded from here.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from lm_eval.models.api_models import LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER

from .guard import GuardError
from .transport import GuardedChatCompletion, RunStopped  # noqa: F401  (RunStopped re-exported)

VLLM_CHAT = "/v1/chat/completions"
VLLM_HEALTH = "/health"
VLLM_MODELS = "/v1/models"
VLLM_ALLOWED = frozenset({("GET", VLLM_HEALTH), ("GET", VLLM_MODELS), ("POST", VLLM_CHAT)})


def vllm_url(base_url: str, method: str, path: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.port is None:
        raise GuardError(f"base URL {base_url!r} is not a loopback vLLM address")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise GuardError(f"base URL {base_url!r} must be a bare origin")
    if (method.upper(), path) not in VLLM_ALLOWED:
        raise GuardError(f"{method.upper()} {path} is not on the vLLM endpoint allowlist")
    return f"http://127.0.0.1:{parts.port}{path}"


class VllmGuardedChatCompletion(GuardedChatCompletion):
    def __init__(self, *, concurrency: int = 64, **kwargs):
        self.concurrency = int(concurrency)
        super().__init__(**kwargs)

    # -- endpoint + guard ----------------------------------------------------

    def _chat_url(self) -> str:
        return vllm_url(self.server_base, "POST", VLLM_CHAT)

    def check_runtime(self, reason: str) -> dict:
        state, problems = None, []
        try:
            h = self.session.get(vllm_url(self.server_base, "GET", VLLM_HEALTH), timeout=10)
            m = self.session.get(vllm_url(self.server_base, "GET", VLLM_MODELS), timeout=10)
            served = [d.get("id") for d in (m.json().get("data") or [])] if m.ok else None
            state = {"health_status": h.status_code, "served_models": served}
            if h.status_code != 200:
                problems.append(f"/health returned {h.status_code}")
            if served != [self.expected_model]:
                problems.append(f"served models {served!r}, expected exactly [{self.expected_model!r}]")
        except Exception as e:  # noqa: BLE001
            self.recorder.guard_check(reason=reason, state=None, problems=["status unreadable"], error=repr(e)[:300])
            raise GuardError(f"vLLM status unreadable ({reason}): {e!r}") from e
        self.recorder.guard_check(reason=reason, state=state, problems=problems, error=None)
        if problems:
            raise GuardError(f"vLLM runtime not authorized ({reason}): " + "; ".join(problems))
        return state

    # -- concurrent generation, order preserved --------------------------------

    def generate_until(self, requests, disable_tqdm: bool = False):
        def one(req):
            context, gen_kwargs = req.args[0], req.args[1]
            messages = json.loads(context.prompt) if hasattr(context, "prompt") else context
            payload, meta = self._create_payload(messages, generate=True, gen_kwargs=copy.deepcopy(gen_kwargs))
            out = self.send(payload, meta)
            texts = self.parse_generations(out)
            text = texts[0] if texts else None
            return LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER if text is None else text

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = [ex.submit(one, r) for r in requests]
            results = []
            for f in futures:
                results.append(f.result())  # raises RunStopped/GuardError: the run stops, nothing scored
        return results
