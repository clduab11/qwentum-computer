"""One-shot authorized Q8_0 report continuation; never loads a model or retries.

Run from the project root with .venv/Scripts/python.exe. All generated records
use exclusive creation. Authentication is not read or supplied.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from qwentum.language_eval import score_original

ROOT = Path("experiments/records")
PREFIX = "q8-arm4-report-v2"
BASE = "http://127.0.0.1:8888"
RAW_SHA = "15a07a45cef6535a81e953363e416e09cf426989b8a3a39c7509304ca0632774"
LOG = Path("C:/Users/cld-main/.unsloth/studio/logs/llama-server/llama-1794106896-port-49590-try0.log")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(suffix, value):
    with (ROOT / f"{PREFIX}-{suffix}.json").open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return json.load(response)


def gpu_sample():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8)
        return {"utc": now(), "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "scope": "whole card, not model-specific; not a peak measurement"}
    except Exception as exc:
        return {"utc": now(), "error_class": type(exc).__name__, "detail": str(exc)}


def main():
    raw = (ROOT / "_arm1_raw.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA
    historical = (ROOT / "unsloth-base-language.json").read_bytes()
    (ROOT / f"{PREFIX}-prior-record.json").open("xb").write(historical)
    record = json.loads(historical)
    row = next(row for row in json.loads(raw) if row["seed"] == 2000005)
    assert row["text"] == "M 0"
    original, execution = score_original(row)  # actual Machine.run, then independent Stim
    assert original["parse"]["pass"] and original["task_correctness"]["pass"]
    execution.update({"observed_utc": now(), "python_version": sys.version,
                      "raw_sha256": RAW_SHA, "program_text": row["text"],
                      "label": "arm 4; post-hoc continuation of preserved arm-1 output; automatic host dispatch; not autonomous tool selection; not embedded execution"})
    assert execution["engine"]["bits"] == execution["oracle"]["bits"] == [0]
    assert execution["oracle"]["version"] == "1.16.0"
    save("execution", execution)

    pending = record["pending_report_request"]
    body = {"model": "unsloth/Qwen3.5-0.8B-GGUF", "messages": pending["messages"],
            "stream": False, "temperature": 0, "top_p": 1, "top_k": 1,
            "min_p": 0, "max_tokens": 512, "seed": row["seed"],
            "chat_template_kwargs": {"enable_thinking": False}}
    # Sampling and transport adapted; pending report messages are unchanged.
    assert len(body["messages"]) == 2
    assert "oracle" not in json.dumps(body["messages"]).lower()
    gpu_before = gpu_sample()
    status = get("/api/inference/status")
    train = get("/api/train/status")
    log_bytes = LOG.read_bytes()
    activity = log_bytes.decode("utf-8", errors="replace").splitlines()[-12:]
    log_age = time.time() - LOG.stat().st_mtime
    preflight = {"utc": now(), "inference": status, "training": train,
                 "backend_log": str(LOG), "backend_log_offset": len(log_bytes),
                 "backend_log_age_seconds": log_age, "backend_log_tail": activity,
                 "slots_probe": "Read-only /slots returned 401; no authentication read or attempted. Fresh loader activity log used instead.",
                 "resource_before": gpu_before}
    save("preflight", preflight)
    if (status.get("active_model") != body["model"] or status.get("gguf_variant") != "Q8_0"
            or train.get("is_training_running") is not False or status.get("loading")):
        raise RuntimeError("Model/training preflight mismatch; no completion sent")
    activity_events = [line for line in activity if "update_slots:" in line or "launch_slot" in line or "processing task" in line]
    if log_age > 30 or not activity_events or "all slots are idle" not in activity_events[-1]:
        raise RuntimeError("Cannot establish recent idle backend; no completion sent")
    body["model"] = status["active_model"]
    request_bytes = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
    request_path = ROOT / f"{PREFIX}-request.json"
    with request_path.open("xb") as f:
        f.write(request_bytes)
    metadata = {"started_utc": now(), "endpoint": BASE + "/v1/chat/completions",
                "request_file": str(request_path), "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "request_headers": {"Content-Type": "application/json"}, "authentication_supplied": False,
                "http_status": None, "error": None}
    start = time.perf_counter()
    response_bytes = b""
    try:
        request = urllib.request.Request(metadata["endpoint"], data=request_bytes,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            metadata["http_status"] = response.status
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        metadata["http_status"] = exc.code
        response_bytes = exc.read()
        metadata["error"] = {"class": type(exc).__name__, "detail": str(exc)}
    except Exception as exc:
        metadata["error"] = {"class": type(exc).__name__, "detail": str(exc)}
    finally:
        metadata.update({"ended_utc": now(), "latency_seconds": time.perf_counter() - start})
        with (ROOT / f"{PREFIX}-response.json").open("xb") as f:
            f.write(response_bytes)
        metadata["response_sha256"] = hashlib.sha256(response_bytes).hexdigest()
        metadata["resource_after"] = gpu_sample()
        metadata["peak_model_vram"] = None
        metadata["peak_model_ram"] = None
        metadata["per_process_memory"] = None
        try:
            after = json.loads(response_bytes)
            metadata["usage"] = after.get("usage")
            metadata["choices"] = after.get("choices")
        except Exception:
            metadata["response_parse_error"] = True
        save("metadata", metadata)
        after_log = LOG.read_bytes()[len(log_bytes):].decode("utf-8", errors="replace")
        # Only model-serving lines; never capture headers or credentials.
        relevant = [line for line in after_log.splitlines() if any(term in line for term in
                    ("slot ", "slots:", "eval time", "prompt eval", "sampler", "truncated", "POST /v1/chat/completions",
                     "repeat_last_n =", "dry_multiplier =", "top_k =", "mirostat ="))]
        save("backend", {"utc": now(), "log": str(LOG), "offset": len(log_bytes), "lines": relevant,
                         "effective_sampling": "Retain sampler parameter continuation lines when present; distinguish them from requested values and from server defaults."})
    print(json.dumps(metadata, indent=2))
    if metadata["error"]:
        raise RuntimeError("Completion failed; retained evidence; no retry")


if __name__ == "__main__":
    main()
