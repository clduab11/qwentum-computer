"""EXP-E-002 discovery v3. No Core-4 items.

v3 changes (approved 2026-11-09):
* requests an NVIDIA L4 and keeps bfloat16; stops before install if the GPU
  cannot run bfloat16 (compute capability < 8.0), recording what Kaggle gave;
* smoke test sends each prompt as its own conversation and checks the answers;
* throughput test with 64 synthetic prompts (not benchmark items), fixed length.
Outputs land in /kaggle/working/discovery for `kaggle kernels output`.
"""

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/kaggle/working/discovery")
OUT.mkdir(parents=True, exist_ok=True)
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL = "Qwen/Qwen3.5-0.8B"
VERSION = 3


def save(name, obj):
    path = OUT / name
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def run(cmd, timeout):
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": cmd,
        "code": proc.returncode,
        "elapsed_s": round(time.time() - started, 1),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def gpu_info():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    rows = []
    for line in q.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"name": parts[0], "compute_cap": parts[1], "memory": parts[2], "driver": parts[3]})
    return {"rows": rows, "stderr": q.stderr}


def main():
    save("started.json", {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "version": VERSION})
    gpu = gpu_info()
    save("gpu.json", gpu)
    print("GPU:", gpu)
    caps = [float(r["compute_cap"]) for r in gpu["rows"] if r["compute_cap"].replace(".", "", 1).isdigit()]
    if not caps or min(caps) < 8.0:
        save("result.json", {"ok": False, "stage": "gpu_check",
                             "reason": "assigned GPU cannot run bfloat16 (compute capability < 8.0); stopped before install",
                             "gpu": gpu["rows"]})
        return

    install = run(["pip", "install", "-q", "vllm"], timeout=1500)
    save("vllm-install.json", install)
    if install["code"] != 0:
        save("result.json", {"ok": False, "stage": "install"})
        return

    import vllm
    from vllm import LLM, SamplingParams

    save("versions.json", {"vllm": getattr(vllm, "__version__", "unknown")})

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        revision=REVISION,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
    )
    save("load.json", {"elapsed_s": round(time.time() - t0, 1), "model": MODEL, "revision": REVISION, "dtype": "bfloat16"})

    greedy = dict(temperature=0.0, top_p=1.0, top_k=1, min_p=0.0,
                  presence_penalty=0.0, frequency_penalty=0.0, seed=20261108)

    # Smoke: each prompt is its own conversation (v2 sent them as one).
    smoke = [
        ("Reply with the single word pong.", lambda t: "pong" in t.lower()),
        ("What is 2+2? Reply with the number only.", lambda t: t.strip().rstrip(".") == "4"),
        ("What is the capital of France? Reply with one word.", lambda t: "paris" in t.lower()),
    ]
    convs = [[{"role": "user", "content": p}] for p, _ in smoke]
    t1 = time.time()
    outs = llm.chat(convs, sampling_params=SamplingParams(max_tokens=64, **greedy),
                    chat_template_kwargs={"enable_thinking": False})
    records = []
    for (prompt, check), out in zip(smoke, outs):
        o = out.outputs[0]
        records.append({"prompt": prompt, "text": o.text, "finish_reason": o.finish_reason,
                        "tokens": len(o.token_ids), "check_passed": bool(check(o.text))})
        print("ANSWER:", repr(prompt), "->", repr(o.text[:120]))
    save("smoke.json", {"elapsed_s": round(time.time() - t1, 1), "n_prompts": len(smoke),
                        "n_outputs": len(outs), "records": records})

    # Throughput: 64 synthetic prompts, fixed 512-token outputs (ignore_eos).
    synth = [[{"role": "user", "content": f"Write a long numbered list of facts about the number {i}."}] for i in range(64)]
    t2 = time.time()
    touts = llm.chat(synth, sampling_params=SamplingParams(max_tokens=512, ignore_eos=True, **greedy),
                     chat_template_kwargs={"enable_thinking": False})
    elapsed = time.time() - t2
    gen = sum(len(o.outputs[0].token_ids) for o in touts)
    prompt_toks = sum(len(o.prompt_token_ids or []) for o in touts)
    save("throughput.json", {
        "n_prompts": len(synth), "max_tokens": 512, "elapsed_s": round(elapsed, 2),
        "generated_tokens": gen, "prompt_tokens": prompt_toks,
        "gen_tokens_per_s": round(gen / elapsed, 1) if elapsed else None,
        "sample_tail_prompt0": touts[0].outputs[0].text[-300:],
        "note": "synthetic prompts, not benchmark items; fixed length via ignore_eos",
    })
    save("result.json", {"ok": True, "stage": "done",
                         "smoke_checks_passed": sum(r["check_passed"] for r in records), "smoke_total": len(records)})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("result.json", {"ok": False, "stage": "exception", "error": repr(exc)})
        raise
