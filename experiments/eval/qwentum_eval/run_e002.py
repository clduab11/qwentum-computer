"""EXP-E-002 runner: executes inside the Kaggle notebook against its own vLLM server.

The notebook entry script starts vLLM, then calls `run(mode, out_dir)`. Each
Core-4 task goes through the pinned harness with the guarded vLLM transport.
Everything is written to a fresh run directory; a stopped run is incomplete
and is never scored or resumed.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from . import pins_e002 as P
from .records import RunRecorder, utc_now

TASKS = ("math500", "gsm8k", "ifeval", "mmlu_pro_1k")


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def _write_json(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(_jsonable(obj), fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def run(mode: str, out_dir: Path, server_base: str, manifest: dict, task_names=TASKS) -> dict:
    from lm_eval.evaluator import simple_evaluate

    from . import tasks as core4
    from .kaggle_transport import VllmGuardedChatCompletion
    from .transport import RunStopped

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{mode}"
    run_dir = Path(out_dir) / run_id
    recorder = RunRecorder(run_dir)
    started = time.perf_counter()
    max_wall = P.SMOKE_WALL_S if mode == "smoke" else P.MAX_WALL_S_FULL_RUN
    summary = {
        "experiment": P.EXP_ID, "profile": P.PROFILE_ID, "mode": mode, "run_id": run_id,
        "utc_start": utc_now(), "manifest_sha256": manifest.get("_sha256"),
        "score_status": "plumbing check, not a benchmark score" if mode == "smoke" else "benchmark run",
        "precision_note": "float16 on Tesla T4; float16 vs bfloat16 agreement not measured",
        "tasks": {}, "complete": False,
    }
    recorder.event("run_start", mode=mode, run_id=run_id)
    try:
        for name in task_names:
            task_objs = core4.build_tasks(name, upstream_math_grader=True)
            limit = None
            if mode == "smoke":
                limit = P.SMOKE_LIMIT
                if name == "mmlu_pro_1k":
                    task_objs = {k: v for k, v in task_objs.items() if k.endswith("_biology")}
            lm = VllmGuardedChatCompletion(
                server_base=server_base, recorder=recorder, task_label=name,
                gen_cap=P.GEN_CAPS[name], profile=P.PROFILE_D,
                expected_model=P.EXPECTED_SERVED_MODEL, expected_variant="float16",
                guard_every=P.GUARD_EVERY_N_REQUESTS, run_started_monotonic=started,
                max_wall_s=max_wall, request_timeout_s=P.REQUEST_TIMEOUT_S,
                max_attempts=P.MAX_ATTEMPTS, truncation_min=P.TRUNCATION_CHECK_MIN_RESPONSES,
                truncation_stop_rate=P.TRUNCATION_STOP_RATE, concurrency=P.CONCURRENCY,
            )
            recorder.event("task_start", task=name, harness_tasks=sorted(task_objs), limit=limit)
            t0 = time.perf_counter()
            results = simple_evaluate(
                model=lm, tasks=list(task_objs.values()), limit=limit,
                apply_chat_template=True, fewshot_as_multiturn=True, log_samples=True,
                bootstrap_iters=0, use_cache=None, cache_requests=False,
                random_seed=P.PROFILE_D["seed"], numpy_random_seed=P.PROFILE_D["seed"],
                torch_random_seed=P.PROFILE_D["seed"], fewshot_random_seed=P.PROFILE_D["seed"],
            )
            samples = results.pop("samples", {}) if isinstance(results, dict) else {}
            _write_json(run_dir / f"harness-results-{name}.json", results)
            with open(run_dir / f"harness-samples-{name}.jsonl", "x", encoding="utf-8", newline="\n") as fh:
                for tname, rows in samples.items():
                    for row in rows:
                        fh.write(json.dumps(_jsonable({"harness_task": tname, **row}), ensure_ascii=False) + "\n")
            summary["tasks"][name] = {
                "elapsed_s": round(time.perf_counter() - t0, 2), "requests": lm.n_requests,
                "responses": lm.n_responses, "truncated": lm.n_truncated,
                "results": results.get("results") if isinstance(results, dict) else None,
            }
            recorder.event("task_end", task=name, requests=lm.n_requests, truncated=lm.n_truncated)
        summary["complete"] = True
    except RunStopped as e:
        summary["stopped"] = str(e)
        recorder.event("run_stopped", reason=str(e))
    except Exception as e:  # noqa: BLE001
        summary["error"] = repr(e)
        summary["traceback"] = traceback.format_exc()
        recorder.event("run_error", error=repr(e))
    finally:
        summary["utc_end"] = utc_now()
        summary["elapsed_s"] = round(time.perf_counter() - started, 2)
        _write_json(run_dir / "run-summary.json", summary)
        recorder.close()
    return summary
