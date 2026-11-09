"""EXP-E-002 discovery v4 (approved 2026-11-09, Option 1: float16 on Kaggle T4). No Core-4 items.

Checks, before any benchmark run:
1. Serving path of the planned full run: vLLM OpenAI-compatible server, our exact
   Protocol D payload (greedy, all sampler fields explicit, thinking off).
2. Numerical check: the same 48 development prompts answered by a float16 server
   (GPU 0) and a float32 reference server (GPU 1) of the same pinned checkpoint.
   Prompts are GSM8K *train* items (development data, never final test) plus
   8 synthetic long-output prompts.
3. Throughput with 48 concurrent requests on the float16 server.
4. Smoke prompts, each sent as its own conversation, with answer checks.

All raw requests and responses are saved. Outputs: /kaggle/working/discovery.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/kaggle/working/discovery")
VERSION = 4
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
VLLM_PIN = "vllm==0.33.0"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_FILE = "main/train-00000-of-00001.parquet"
N_GSM = 40
MAX_MODEL_LEN = 8192
PROFILE_D = {
    "temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 20261108,
}
LONG_TOPICS = [
    "how a CPU pipeline handles branch misprediction",
    "the history and mechanics of the printing press",
    "how vaccines train the adaptive immune system",
    "the economics of supply and demand with worked examples",
    "how public-key cryptography works, step by step",
    "the water cycle and its effect on regional climate",
    "how compilers turn source code into machine code",
    "the causes and consequences of the industrial revolution",
]


# ---------------------------------------------------------------- helpers

def utc():
    return datetime.now(timezone.utc).isoformat()


def save(name, obj):
    path = OUT / name
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_info():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    rows = []
    for line in q.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            rows.append({"index": p[0], "name": p[1], "compute_cap": p[2], "memory": p[3], "driver": p[4]})
    return rows


ANSWER_RE = re.compile(r"[Tt]he final answer is\s*\$?\s*(-?[\d,]*\.?\d+)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    m = ANSWER_RE.findall(text or "")
    if m:
        return m[-1].replace(",", "").rstrip(".")
    nums = NUM_RE.findall(text or "")
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def norm_num(s):
    if s is None:
        return None
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def has_loop(text, n=8, times=4):
    words = (text or "").split()
    counts = {}
    for i in range(len(words) - n + 1):
        k = tuple(words[i:i + n])
        counts[k] = counts.get(k, 0) + 1
        if counts[k] >= times:
            return True
    return False


def has_garbage(text):
    t = text or ""
    return "\ufffd" in t or any(ord(c) < 32 and c not in "\n\t\r" for c in t)


def first_divergence(a, b):
    a, b = a or "", b or ""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ---------------------------------------------------------------- servers

def server_cmd(dtype, port):
    exe = shutil.which("vllm")
    args = ["--revision", REVISION, "--dtype", dtype, "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", "0.85", "--host", "127.0.0.1", "--port", str(port),
            "--generation-config", "vllm", "--seed", "0"]
    if exe:
        return [exe, "serve", MODEL] + args
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL] + args


def launch(name, dtype, gpu, port):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_LOGGING_LEVEL="INFO")
    log = open(OUT / f"server-{name}.log", "w", encoding="utf-8")
    cmd = server_cmd(dtype, port)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"name": name, "dtype": dtype, "gpu": gpu, "port": port, "cmd": cmd,
            "proc": proc, "log": log, "t_launch": time.time()}


def wait_ready(srv, timeout=2100):
    import requests

    base = f"http://127.0.0.1:{srv['port']}"
    while time.time() - srv["t_launch"] < timeout:
        if srv["proc"].poll() is not None:
            return {"ready": False, "reason": f"server exited with code {srv['proc'].returncode}"}
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200:
                models = requests.get(base + "/v1/models", timeout=10).json()
                return {"ready": True, "startup_s": round(time.time() - srv["t_launch"], 1),
                        "served_models": [m.get("id") for m in models.get("data", [])]}
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return {"ready": False, "reason": "startup timeout"}


def stop(srv):
    if srv["proc"].poll() is None:
        srv["proc"].terminate()
        try:
            srv["proc"].wait(60)
        except subprocess.TimeoutExpired:
            srv["proc"].kill()
    srv["log"].close()


def chat(port, messages, max_tokens, stop_seqs=None):
    import requests

    payload = {"model": MODEL, "messages": messages, "stream": False, "max_tokens": max_tokens,
               **PROFILE_D, "chat_template_kwargs": {"enable_thinking": False}}
    if stop_seqs:
        payload["stop"] = stop_seqs
    t0 = time.time()
    rec = {"utc_start": utc(), "payload": payload}
    try:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=1500)
        rec.update(status=r.status_code, response_text=r.text)
        if r.ok:
            d = r.json()
            ch = d["choices"][0]
            rec.update(content=ch["message"].get("content"), finish_reason=ch.get("finish_reason"),
                       completion_tokens=(d.get("usage") or {}).get("completion_tokens"),
                       prompt_tokens=(d.get("usage") or {}).get("prompt_tokens"),
                       reasoning_present=bool(ch["message"].get("reasoning_content")))
    except Exception as e:  # noqa: BLE001
        rec.update(status=None, error=repr(e)[:500])
    rec["elapsed_s"] = round(time.time() - t0, 3)
    return rec


def run_batch(srv, items):
    """items: list of (id, messages, max_tokens). Sent concurrently; returns records + wall time."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futs = {iid: ex.submit(chat, srv["port"], msgs, mt) for iid, msgs, mt in items}
        recs = {iid: f.result() for iid, f in futs.items()}
    wall = time.time() - t0
    with open(OUT / f"records-{srv['name']}.jsonl", "a", encoding="utf-8") as fh:
        for iid, rec in recs.items():
            fh.write(json.dumps({"id": iid, **rec}, ensure_ascii=False) + "\n")
    return recs, wall


# ---------------------------------------------------------------- prompts

def build_items():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(GSM8K_REPO, GSM8K_FILE, repo_type="dataset", revision=GSM8K_REV)
    df = pd.read_parquet(p)
    gsm = []
    for i in range(N_GSM):
        q, a = df.iloc[i]["question"], df.iloc[i]["answer"]
        gold = norm_num(a.split("####")[-1].strip().replace(",", ""))
        msgs = [{"role": "user", "content": f"{q}\nSolve step by step. End with the sentence: The final answer is N."}]
        gsm.append((f"gsm8k-train-{i}", msgs, 1024, gold))
    longs = [(f"long-{j}", [{"role": "user", "content": f"Write a detailed, multi-section technical explanation of {t}."}], 2048, None)
             for j, t in enumerate(LONG_TOPICS)]
    meta = {"gsm8k_file": GSM8K_FILE, "gsm8k_revision": GSM8K_REV, "gsm8k_sha256": sha256_file(p),
            "gsm8k_split": "train (development data)", "gsm8k_rows": list(range(N_GSM)), "long_topics": LONG_TOPICS}
    return gsm + longs, meta


SMOKE = [
    ("Reply with the single word pong.", lambda t: "pong" in (t or "").lower()),
    ("What is 2+2? Reply with the number only.", lambda t: (t or "").strip().rstrip(".") == "4"),
    ("What is the capital of France? Reply with one word.", lambda t: "paris" in (t or "").lower()),
]


def summarize(recs, golds):
    out = {"n": len(recs), "http_ok": sum(r.get("status") == 200 for r in recs.values()),
           "finish": {}, "loops": [], "garbage": [], "reasoning_present": 0, "completion_tokens": 0}
    correct = 0
    for iid, r in recs.items():
        fr = r.get("finish_reason")
        out["finish"][str(fr)] = out["finish"].get(str(fr), 0) + 1
        out["completion_tokens"] += r.get("completion_tokens") or 0
        out["reasoning_present"] += int(bool(r.get("reasoning_present")))
        if has_loop(r.get("content")):
            out["loops"].append(iid)
        if has_garbage(r.get("content")):
            out["garbage"].append(iid)
        if golds.get(iid) is not None and norm_num(extract_answer(r.get("content"))) == golds[iid]:
            correct += 1
    out["gsm8k_train_correct"] = correct
    out["gsm8k_train_n"] = sum(g is not None for g in golds.values())
    return out


def compare(r16, r32, golds):
    rows, exact, ans_agree, n_gsm = [], 0, 0, 0
    for iid in r16:
        a, b = r16[iid].get("content"), (r32.get(iid) or {}).get("content")
        same = a == b
        exact += int(same)
        row = {"id": iid, "exact": same, "first_divergence_char": None if same else first_divergence(a, b),
               "len16": len(a or ""), "len32": len(b or ""),
               "finish16": r16[iid].get("finish_reason"), "finish32": (r32.get(iid) or {}).get("finish_reason")}
        if golds.get(iid) is not None:
            n_gsm += 1
            x, y = norm_num(extract_answer(a)), norm_num(extract_answer(b))
            row.update(ans16=x, ans32=y, gold=golds[iid], answers_agree=(x == y))
            ans_agree += int(x == y)
        rows.append(row)
    return {"exact_text_match": exact, "n": len(r16), "gsm8k_final_answer_agreement": ans_agree,
            "gsm8k_n": n_gsm, "rows": rows}


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("started.json", {"utc": utc(), "python": platform.python_version(), "version": VERSION})
    gpus = gpu_info()
    save("gpu.json", gpus)
    print("GPU:", gpus, flush=True)
    if not gpus:
        save("result.json", {"ok": False, "stage": "gpu_check", "reason": "no GPU visible"})
        return

    t = time.time()
    inst = subprocess.run([sys.executable, "-m", "pip", "install", "-q", VLLM_PIN], capture_output=True, text=True, timeout=1500)
    save("vllm-install.json", {"pin": VLLM_PIN, "code": inst.returncode, "elapsed_s": round(time.time() - t, 1),
                               "stderr_tail": inst.stderr[-3000:]})
    if inst.returncode != 0:
        save("result.json", {"ok": False, "stage": "install"})
        return
    import vllm  # noqa: F401
    from huggingface_hub import snapshot_download

    save("versions.json", {"vllm": vllm.__version__})

    t = time.time()
    snap = Path(snapshot_download(MODEL, revision=REVISION))
    weights = {f.name: {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
               for f in sorted(snap.glob("*.safetensors"))}
    save("model-files.json", {"model": MODEL, "revision": REVISION, "download_s": round(time.time() - t, 1),
                              "safetensors": weights,
                              "other_files": sorted(f.name for f in snap.iterdir() if f.suffix != ".safetensors")})

    items, meta = build_items()
    golds = {iid: g for iid, _, _, g in items}
    save("prompts.json", {**meta, "n_items": len(items), "ids": [i for i, *_ in items]})
    batch = [(iid, msgs, mt) for iid, msgs, mt, _ in items]

    two = len(gpus) >= 2
    specs = [("fp16", "float16", 0, 8000), ("fp32", "float32", 1 if two else 0, 8001)]
    servers, status, results = {}, {}, {}

    def bring_up(spec):
        srv = launch(*spec)
        servers[spec[0]] = srv
        return srv

    def exercise(srv):
        st = wait_ready(srv)
        status[srv["name"]] = st
        save(f"server-status-{srv['name']}.json", {k: v for k, v in srv.items() if k not in ("proc", "log")} | st)
        if not st.get("ready"):
            return
        if srv["name"] == "fp16":
            smoke = []
            for prompt, check in SMOKE:
                r = chat(srv["port"], [{"role": "user", "content": prompt}], 64)
                smoke.append({"prompt": prompt, "content": r.get("content"), "finish_reason": r.get("finish_reason"),
                              "status": r.get("status"), "check_passed": bool(check(r.get("content")))})
                with open(OUT / "records-fp16-smoke.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"prompt": prompt, **r}, ensure_ascii=False) + "\n")
            save("smoke.json", smoke)
        recs, wall = run_batch(srv, batch)
        results[srv["name"]] = recs
        summ = summarize(recs, golds)
        summ.update(wall_s=round(wall, 1),
                    gen_tokens_per_s=round(summ["completion_tokens"] / wall, 1) if wall else None,
                    concurrency=len(batch))
        save(f"summary-{srv['name']}.json", summ)

    if two:
        both = [bring_up(s) for s in specs]
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(exercise, both))
        for s in both:
            stop(s)
    else:
        for s in specs:
            srv = bring_up(s)
            exercise(srv)
            stop(srv)

    out = {"ok": "fp16" in results, "stage": "done", "two_gpus": two,
           "server_status": status}
    if "fp16" in results and "fp32" in results:
        cmp_ = compare(results["fp16"], results["fp32"], golds)
        save("compare-fp16-vs-fp32.json", cmp_)
        out.update(exact_text_match=f"{cmp_['exact_text_match']}/{cmp_['n']}",
                   gsm8k_answer_agreement=f"{cmp_['gsm8k_final_answer_agreement']}/{cmp_['gsm8k_n']}")
    save("result.json", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("result.json", {"ok": False, "stage": "exception", "error": repr(exc)})
        raise
