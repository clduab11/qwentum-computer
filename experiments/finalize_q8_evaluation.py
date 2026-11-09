"""Offline evidence finalization. No inference, model load, or network access."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

from qwentum.language_eval import score_report

ROOT = Path("experiments/records")
PREFIX = "q8-arm4-report-v2"
SHA = "15a07a45cef6535a81e953363e416e09cf426989b8a3a39c7509304ca0632774"


def read(name):
    return json.loads((ROOT / name).read_bytes())


def save(name, data, exclusive=True):
    with (ROOT / name).open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = (ROOT / "_arm1_raw.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    prior = read(f"{PREFIX}-prior-record.json")
    record = read("unsloth-base-language.json")
    assert record == prior, "Refuse to overwrite an already modified evaluation"
    execution = read(f"{PREFIX}-execution.json")
    response = read(f"{PREFIX}-response.json")
    request = read(f"{PREFIX}-request.json")
    metadata = read(f"{PREFIX}-metadata.json")
    preflight = read(f"{PREFIX}-preflight.json")
    assert request["messages"] == prior["pending_report_request"]["messages"]
    assert metadata["error"] is None and metadata["http_status"] == 200
    assert preflight["inference"]["active_model"] == response["model"] == request["model"]
    assert preflight["training"]["is_training_running"] is False
    assert execution["engine"]["bits"] == execution["oracle"]["bits"] == [0]
    text = response["choices"][0]["message"]["content"]
    scores = score_report(text, execution, True)
    save(f"{PREFIX}-scores.json", {"observed_utc": utc, **scores})

    # Preserve original request evidence, transport deviation, pending specification,
    # artifact attribution, original scores and historical log verbatim in prior-record.
    record["schema_version"] = 2
    record["status"] = "evaluation_complete_with_exact_report_format_failure"
    record["historical_routing_blocker"] = record["routing_blocker"]
    record["routing_blocker"] = None
    record["historical_pending_report_request"] = record.pop("pending_report_request")
    record["prior_record"] = {"file": f"{PREFIX}-prior-record.json",
                              "sha256": hashlib.sha256((ROOT / f"{PREFIX}-prior-record.json").read_bytes()).hexdigest()}
    record["direct_access_authority"] = "Chris 2026-11-07 active workspace exception: qwentum-computer only; existing loaded model, chat completions only; historical deviations unchanged"
    record["completed_utc"] = utc
    record["completion_scope"] = "Preserved-output scoring and one post-hoc report request complete; no new program generation, retry, training, model switch, or embedded execution"
    for a, b in zip(record["arm1"], record["arm4_continuation"]):
        a.update({"report_format_pass": None, "report_oracle_agreement": None,
                  "task_aware_end_to_end": None})
        b["parse"] = dict(a["parse"])
        b["task_correctness"] = dict(a["task_correctness"])
        b.update({"report_format_pass": None, "report_oracle_agreement": None,
                  "task_aware_end_to_end": None, "pipeline_success": False})
        if b["seed"] == 2000005:
            b["historical_execution"] = {key: b[key] for key in ("engine", "oracle", "route", "program", "inject", "engine_oracle_agreement")}
            b.update(execution)
            b.update(scores)
            b["engine"]["exception_class"] = None
            b["oracle"]["exception_class"] = None
            b["report_status"] = "model_report_received_exact_format_failed_missing_period"
            b["report_evidence_prefix"] = PREFIX
            b["pipeline_success"] = scores["task_aware_end_to_end"]
            b["engine_execution_count_this_evaluation"] = 1
    record["summary"].update({"model_reports": 1, "faithfulness": {"pass": 1, "assessed": 1},
                              "report_oracle_agreement": {"pass": 1, "assessed": 1},
                              "exact_report_format": {"pass": 0, "assessed": 1},
                              "end_to_end": {"pass": 0, "assessed": 1},
                              "task_aware_pipeline_success": {"pass": 0, "rows": 8},
                              "new_completion_calls": 1,
                              "continuation_engine_calls_note": "1 dispatched row of 8, not lifetime invocation count; historical execution plus one explicitly recorded rerun",
                              "not_applicable_report_rows": 7})
    record["failures"].append({"seed": 2000005, "stage": "separate_model_reporting",
                               "failure_class": "ExactReportFormatMismatch",
                               "detail": "Model omitted final period. Correct bits retained; no repair or retry.",
                               "actual": text, "expected": "Engine measurement bits: [0]."})
    record["finalization_command"] = "PYTHONPATH=src ./.venv/Scripts/python.exe experiments/finalize_q8_evaluation.py"
    record["python_version"] = sys.version
    save("unsloth-base-language.json", record, exclusive=False)

    log = Path(preflight["backend_log"])
    data = log.read_bytes()[preflight["backend_log_offset"]:].decode("utf-8", errors="replace")
    lines = data.splitlines()
    # Keep only the reporting task's log range, including multiline sampler params.
    first = next(i for i, line in enumerate(lines) if "slot get_availabl" in line)
    last = next(i for i, line in enumerate(lines) if "task 333 | stop processing" in line)
    serving_lines = lines[first:last + 1]
    save(f"{PREFIX}-sampler-audit.json", {
        "observed_utc": utc, "source_log": str(log), "source_offset": preflight["backend_log_offset"],
        "lines": serving_lines,
        "correction_to_initial_backend_capture": "Initial filtered capture omitted continuation lines after sampler params. Read-only recovery below establishes the actual values; no new generation.",
        "requested": {k: request[k] for k in ("temperature", "top_p", "top_k", "min_p", "max_tokens", "seed", "chat_template_kwargs")},
        "backend_effective_logged": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0,
                                     "presence_penalty": 1.5, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
                                     "greedy_sampler": True, "context_slot": 2048, "truncated": False},
        "seed_effective": "Requested 2000005; source forwarding verified, not independently logged at sampler",
        "max_tokens_effective": "Requested 512; source forwarding verified, cap not reached so not empirically verified",
        "reasoning": {"requested_enable_thinking": False, "reasoning_content": response["choices"][0]["message"]["reasoning_content"],
                      "reported_reasoning_tokens": response["usage"]["completion_tokens_details"]["reasoning_tokens"]},
        "caveat": "Presence penalty 1.5 is inherited from serving defaults and was not overridden by the authorized request contract. No claim of penalty-free greedy baseline."})
    artifact = Path("C:/Users/cld-main/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q8_0.gguf")
    h = hashlib.sha256()
    with artifact.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    assert artifact.stat().st_size == 811843840
    assert h.hexdigest() == prior["artifact"]["artifact_sha256"]
    source_root = Path("C:/Users/cld-main/.unsloth/studio/unsloth_studio/Lib/site-packages/studio/backend")
    excerpts = []
    for rel, ranges in [("routes/inference.py", [(30256,30272),(33103,33135)]),
                        ("core/inference/llama_cpp.py", [(37127,37169),(37265,37278)])]:
        p = source_root / rel
        source = p.read_bytes()
        split = source.decode("utf-8").splitlines()
        excerpts.append({"file": str(p), "sha256": hashlib.sha256(source).hexdigest(),
                         "excerpts": [{"first_line": lo, "last_line": hi, "text": "\n".join(split[lo-1:hi])} for lo,hi in ranges]})
    save(f"{PREFIX}-artifact-serving-audit.json", {
        "observed_utc": utc, "artifact_path": str(artifact), "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": h.hexdigest(), "artifact_repository_revision": prior["artifact"]["revision"],
        "upstream_safetensors_source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "exact_derivation_from_upstream_verified": False,
        "historical_audit_unchanged": "unsloth-q8-load-audit.json",
        "installed_source_read_only_evidence": excerpts,
        "new_completions": 1, "diagnostic_program_generation": "not run; not needed",
        "peak_model_vram": None, "peak_model_ram": None, "per_process_memory": None,
        "resource_scope": "Before/after nvidia-smi samples only, whole-card not model memory or guaranteed peak",
        "raw_sha256": SHA})
    print(json.dumps(record["summary"], indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
