"""Run Qwentum Core-4 (Protocol D) against the already-loaded Unsloth model.

This script sends model requests. It requires explicit approval for each run.
It never loads or unloads a model and refuses to start unless the authorized
Qwen Q8_0 is active and training is idle.

Usage (repository root):
    .venv-eval\\Scripts\\python.exe experiments\\eval\\run_core4.py --mode smoke --i-have-approval
    .venv-eval\\Scripts\\python.exe experiments\\eval\\run_core4.py --mode full  --i-have-approval

smoke = the first 5 documents of each Core-4 task (MMLU-Pro: first 5 frozen
subset documents of biology only), a plumbing check that is never a score.
full  = all 3,360 documents, one task at a time.

Outputs go to a new directory experiments/records/EXP-E-001/<run_id>/ that must
not already exist: raw chat attempts, guard checks, events, harness per-sample
logs, harness results and a run summary. A stopped or failed run is recorded
as incomplete; nothing is resumed or regenerated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwentum_eval import REPO_ROOT, pins  # noqa: E402
from qwentum_eval.records import RunRecorder, utc_now  # noqa: E402

SMOKE_LIMIT = 5
SMOKE_WALL_S = 15 * 60


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def _write_json(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(_jsonable(obj), fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def _verify_manifest() -> dict:
    from qwentum_eval.manifest import verify_manifest

    return verify_manifest(REPO_ROOT / pins.MANIFEST_FILE)


def run(mode: str, server_base: str, run_root: Path, task_names: list[str]) -> int:
    from lm_eval.evaluator import simple_evaluate

    from qwentum_eval import tasks as core4
    from qwentum_eval.transport import GuardedChatCompletion, RunStopped

    manifest_check = _verify_manifest()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{mode}"
    run_dir = run_root / run_id
    recorder = RunRecorder(run_dir)
    started = time.perf_counter()
    max_wall = SMOKE_WALL_S if mode == "smoke" else pins.MAX_WALL_S_FULL_RUN
    summary = {
        "experiment": pins.EXP_ID,
        "profile": pins.PROFILE_ID,
        "mode": mode,
        "run_id": run_id,
        "utc_start": utc_now(),
        "manifest": manifest_check,
        "score_status": "plumbing check, not a benchmark score" if mode == "smoke" else "benchmark run",
        "tasks": {},
        "complete": False,
    }
    recorder.event("run_start", mode=mode, run_id=run_id, manifest_sha256=manifest_check["manifest_sha256"])
    exit_code = 0
    try:
        for name in task_names:
            task_objs = core4.build_tasks(name)
            if mode == "smoke":
                if name == "mmlu_pro_1k":
                    task_objs = {k: v for k, v in task_objs.items() if k.endswith("_biology")}
                limit = SMOKE_LIMIT
            else:
                limit = None
            lm = GuardedChatCompletion(
                server_base=server_base,
                recorder=recorder,
                task_label=name,
                gen_cap=pins.GEN_CAPS[name],
                run_started_monotonic=started,
                max_wall_s=max_wall,
            )
            recorder.event("task_start", task=name, harness_tasks=sorted(task_objs), limit=limit)
            t0 = time.perf_counter()
            results = simple_evaluate(
                model=lm,
                tasks=list(task_objs.values()),
                limit=limit,
                apply_chat_template=True,
                fewshot_as_multiturn=True,
                log_samples=True,
                bootstrap_iters=0,
                use_cache=None,
                cache_requests=False,
                random_seed=pins.PROFILE_D["seed"],
                numpy_random_seed=pins.PROFILE_D["seed"],
                torch_random_seed=pins.PROFILE_D["seed"],
                fewshot_random_seed=pins.PROFILE_D["seed"],
            )
            samples = results.pop("samples", {}) if isinstance(results, dict) else {}
            _write_json(run_dir / f"harness-results-{name}.json", results)
            with open(run_dir / f"harness-samples-{name}.jsonl", "x", encoding="utf-8", newline="\n") as fh:
                for tname, rows in samples.items():
                    for row in rows:
                        fh.write(json.dumps(_jsonable({"harness_task": tname, **row}), ensure_ascii=False) + "\n")
            summary["tasks"][name] = {
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "requests": lm.n_requests,
                "responses": lm.n_responses,
                "truncated": lm.n_truncated,
                "results": results.get("results") if isinstance(results, dict) else None,
            }
            recorder.event("task_end", task=name, requests=lm.n_requests, truncated=lm.n_truncated)
        summary["complete"] = True
    except RunStopped as e:
        summary["stopped"] = str(e)
        recorder.event("run_stopped", reason=str(e))
        exit_code = 2
    except Exception as e:  # noqa: BLE001
        summary["error"] = repr(e)
        summary["traceback"] = traceback.format_exc()
        recorder.event("run_error", error=repr(e))
        exit_code = 3
    finally:
        summary["utc_end"] = utc_now()
        summary["elapsed_s"] = round(time.perf_counter() - started, 2)
        _write_json(run_dir / "run-summary.json", summary)
        recorder.close()
    print(json.dumps({k: summary[k] for k in ("run_id", "complete", "elapsed_s")}, indent=1))
    print(f"records: {run_dir}")
    return exit_code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("smoke", "full"), required=True)
    ap.add_argument("--i-have-approval", action="store_true", help="Chris approved this specific run in the conversation")
    ap.add_argument("--tasks", nargs="*", default=["math500", "gsm8k", "ifeval", "mmlu_pro_1k"])
    ap.add_argument("--server-base", default=pins.SERVER_BASE)
    ap.add_argument("--records-root", default=str(REPO_ROOT / "experiments" / "records" / pins.EXP_ID))
    args = ap.parse_args(argv)
    if not args.i_have_approval:
        print("Refusing to send model requests without --i-have-approval.", file=sys.stderr)
        return 4
    unknown = set(args.tasks) - set(pins.GEN_CAPS)
    if unknown:
        print(f"unknown tasks: {sorted(unknown)}", file=sys.stderr)
        return 4
    root = Path(args.records_root)
    root.mkdir(parents=True, exist_ok=True)
    return run(args.mode, args.server_base, root, args.tasks)


if __name__ == "__main__":
    raise SystemExit(main())
