"""Windows-safe grading for the harness `minerva_math500` task (approved option a).

The pinned harness grader is reused unchanged: the worker calls upstream
`lm_eval.tasks.minerva_math.utils.process_results`. Only its time limits are
replaced, because both of them fail on Windows:

* `is_equiv` uses `signal.SIGALRM`, which Windows lacks; the error is swallowed
  and every non-identical but equivalent answer scores 0.
* Math-Verify's Windows timeout runs each comparison in a child process that
  fails (WinError 6) and silently returns False.

In the worker process only, the upstream timeout context manager becomes a
no-op and Math-Verify is called with its timeouts disabled. The worker is a
separate process with a hard per-item wall limit; on timeout it is killed and
restarted, and the item is scored 0 with `grader_timeout = 1` (never hidden).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import threading

from . import pins

_SPAWN = mp.get_context("spawn")


class _NoTimeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patched_upstream():
    import functools

    import math_verify
    from lm_eval.tasks.minerva_math import utils as upstream

    upstream.timeout = _NoTimeout
    upstream.parse = functools.partial(math_verify.parse, parsing_timeout=None)
    upstream.verify = functools.partial(math_verify.verify, timeout_seconds=None)
    return upstream


def _worker_main(conn) -> None:
    import logging

    logging.getLogger("math_verify").setLevel(logging.ERROR)
    upstream = _patched_upstream()
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        doc, results = msg
        try:
            conn.send(("ok", upstream.process_results(doc, results)))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)[:500]))


class GraderWorker:
    def __init__(
        self,
        timeout_s: float = pins.MATH_GRADER_TIMEOUT_S,
        startup_timeout_s: float = 180.0,
        worker_target=None,
    ):
        self.timeout_s = float(timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self._target = worker_target or _worker_main  # tests may pass a stand-in worker
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()
        self.restarts = 0

    def _start(self) -> None:
        parent, child = _SPAWN.Pipe()
        proc = _SPAWN.Process(target=self._target, args=(child,), daemon=True)
        proc.start()
        child.close()
        if not parent.poll(self.startup_timeout_s):
            proc.kill()
            raise RuntimeError("MATH grader worker did not start")
        status, _ = parent.recv()
        if status != "ready":
            raise RuntimeError(f"MATH grader worker failed to start: {status}")
        self._proc, self._conn = proc, parent

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(10)
        self._proc = self._conn = None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                    self._proc.join(5)
                except Exception:  # noqa: BLE001, S110
                    pass
            self._kill()

    def grade(self, doc: dict, results: list[str]) -> dict:
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._start()
            self._conn.send(({k: doc[k] for k in ("problem", "solution", "answer")}, list(results)))
            if not self._conn.poll(self.timeout_s):
                self._kill()
                self.restarts += 1
                return {"exact_match": 0, "math_verify": 0, "grader_timeout": 1, "grader_error": 0}
            status, value = self._conn.recv()
        if status == "ok":
            return {
                "exact_match": int(value["exact_match"]),
                "math_verify": int(value["math_verify"]),
                "grader_timeout": 0,
                "grader_error": 0,
            }
        return {"exact_match": 0, "math_verify": 0, "grader_timeout": 0, "grader_error": 1}


_WORKER: GraderWorker | None = None


def get_worker() -> GraderWorker:
    global _WORKER
    if _WORKER is None:
        _WORKER = GraderWorker()
        atexit.register(_WORKER.close)
    return _WORKER


def process_results(doc: dict, results: list[str]) -> dict:
    """Drop-in `process_results` for the minerva_math500 task config."""
    return get_worker().grade(doc, results)


METRIC_LIST = [
    {"metric": "math_verify", "aggregation": "mean", "higher_is_better": True},
    {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
    {"metric": "grader_timeout", "aggregation": "mean", "higher_is_better": False},
    {"metric": "grader_error", "aggregation": "mean", "higher_is_better": False},
]
