"""EXP-E-002 Kaggle entry (approved 2026-11-09). Mode comes from RUN_MODE below.

1. Copy the attached bundle (package + frozen files + pinned datasets) to /kaggle/working/qwentum.
2. Verify every bundled file against the bundle's MANIFEST.json hashes.
3. Install the pinned stack (vllm==0.33.0, lm-eval[api,ifeval,math]==0.4.13).
4. Start one vLLM OpenAI server (float16, pinned revision) on GPU 0 and wait for it.
5. Verify the downloaded safetensors SHA-256 against the pinned value.
6. Run Core-4 through the pinned harness and the guarded vLLM transport.
All outputs land in /kaggle/working/exp-e-002 for `kaggle kernels output`.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_MODE = "smoke"  # "smoke" or "full"; set per pushed kernel version
BUNDLE_SLUG = "qwentum-exp-e-002-bundle"
WORK = Path("/kaggle/working/qwentum")
OUT = Path("/kaggle/working/exp-e-002")
PORT = 8000


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name, obj):
    p = OUT / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str), encoding="utf-8")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def unpack_payload():
    """Unpack the embedded package/manifest payload (generated kernels) and verify every member."""
    import base64
    import io
    import tarfile

    raw = base64.b64decode(PAYLOAD_B64)  # noqa: F821  (injected by build_e002_kernel.py)
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:  # noqa: F821
        raise RuntimeError("embedded payload hash mismatch")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as tf:
        tf.extractall(WORK, filter="data")
    bad = [f for f, h in PAYLOAD_MEMBERS.items() if sha256(WORK / f) != h]  # noqa: F821
    save("bundle-verify.json", {"source": "embedded payload", "payload_sha256": PAYLOAD_SHA256,  # noqa: F821
                                "files": len(PAYLOAD_MEMBERS), "mismatches": bad})  # noqa: F821
    if bad:
        raise RuntimeError(f"payload member hash mismatch: {bad[:5]}")
    return WORK


def fetch_datasets(manifest):
    """Download each pinned dataset file from the Hub and verify it against the manifest hash."""
    from huggingface_hub import hf_hub_download

    cache = WORK / "data_cache" / "hf"
    os.environ["HF_HOME"] = str(cache)
    report = {}
    for key, spec in manifest["datasets"].items():
        for fname, meta in spec["files"].items():
            p = Path(hf_hub_download(spec["repo"], fname, repo_type="dataset", revision=spec["revision"],
                                     cache_dir=str(cache / "hub")))
            got = sha256(p)
            report[f"{spec['repo']}/{fname}"] = {"expected": meta["sha256"], "got": got, "ok": got == meta["sha256"]}
    save("datasets-verify.json", report)
    bad = [k for k, v in report.items() if not v["ok"]]
    if bad:
        raise RuntimeError(f"dataset hash mismatch: {bad}")
    log(f"datasets verified: {len(report)} files")


def find_bundle():
    """Locate the attached bundle. Kaggle may expose folders as-is or as the .tar files we uploaded."""
    import tarfile

    if "PAYLOAD_B64" in globals():
        return None
    listing = sorted(str(p) for p in Path("/kaggle/input").rglob("*"))[:400]
    save("input-listing.json", listing)
    for p in Path("/kaggle/input").rglob("MANIFEST.json"):
        root = p.parent
        if (root / "qwentum_eval").is_dir():
            return root
        tars = sorted(root.glob("*.tar"))
        if tars:
            staged = Path("/kaggle/working/bundle-staged")
            if staged.exists():
                shutil.rmtree(staged)
            staged.mkdir(parents=True)
            for f in root.iterdir():
                if f.is_file() and f.suffix != ".tar":
                    shutil.copy2(f, staged / f.name)
            for t in tars:
                with tarfile.open(t) as tf:
                    tf.extractall(staged, filter="data")
            for f in root.iterdir():
                if f.is_dir():
                    shutil.copytree(f, staged / f.name, dirs_exist_ok=True)
            return staged
    raise FileNotFoundError("bundle MANIFEST.json not found under /kaggle/input")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("entry-started.json", {"utc": datetime.now(timezone.utc).isoformat(), "mode": RUN_MODE, "python": sys.version})

    src = find_bundle()
    embedded = src is None
    if embedded:
        unpack_payload()
        log("embedded payload verified")
    else:
        if WORK.exists():
            shutil.rmtree(WORK)
        shutil.copytree(src, WORK)
        bm = json.loads((WORK / "MANIFEST.json").read_text())
        bad = [f for f, h in bm["files"].items() if sha256(WORK / f) != h]
        save("bundle-verify.json", {"bundle_source": str(src), "files": len(bm["files"]), "mismatches": bad,
                                    "bundle_manifest_sha256": sha256(WORK / "MANIFEST.json")})
        if bad:
            raise RuntimeError(f"bundle hash mismatch: {bad[:5]}")
        log(f"bundle verified: {len(bm['files'])} files")

    t = time.time()
    pins = ["vllm==0.33.0", "lm-eval[api,ifeval,math]==0.4.13"]
    inst = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], capture_output=True, text=True, timeout=2400)
    save("install.json", {"pins": pins, "code": inst.returncode, "elapsed_s": round(time.time() - t, 1), "stderr_tail": inst.stderr[-4000:]})
    if inst.returncode != 0:
        raise RuntimeError("pip install failed")
    log("installed")

    os.environ["QWENTUM_REPO_ROOT"] = str(WORK)
    os.environ["QWENTUM_DATA_CACHE"] = str(WORK / "data_cache")
    os.environ["QWENTUM_HF_OFFLINE"] = "1"
    if embedded:
        fetch_datasets(json.loads((WORK / "manifest" / "eval-EXP-E-002.json").read_text()))
    sys.path.insert(0, str(WORK))
    import qwentum_eval  # noqa: F401  (sets HF/NLTK env from the bundle)
    from qwentum_eval import pins_e002 as P

    # The model comes from the Hub at the pinned revision; datasets stay offline.
    os.environ["HF_HUB_OFFLINE"] = "0"
    model_cache = Path("/kaggle/working/hf-model-cache")
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(P.MODEL["repo"], revision=P.MODEL["revision"], cache_dir=str(model_cache)))
    st = snap / P.MODEL["safetensors"]
    got = sha256(st)
    save("model-verify.json", {"file": st.name, "bytes": st.stat().st_size, "sha256": got, "expected": P.MODEL["safetensors_sha256"]})
    if got != P.MODEL["safetensors_sha256"]:
        raise RuntimeError("model safetensors hash mismatch")
    log("model verified")

    S = P.SERVING
    cmd = [shutil.which("vllm") or "vllm", "serve", str(snap), "--served-model-name", P.MODEL["repo"],
           "--dtype", S["dtype"], "--max-model-len", str(S["max_model_len"]),
           "--gpu-memory-utilization", str(S["gpu_memory_utilization"]), "--generation-config", "vllm",
           "--seed", str(S["engine_seed"]), "--host", "127.0.0.1", "--port", str(PORT)]
    save("server-cmd.json", cmd)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1")
    slog = open(OUT / "vllm-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=slog, stderr=subprocess.STDOUT)
    import requests

    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > 2400:
            raise RuntimeError("vLLM startup timeout")
        time.sleep(5)
    save("server-ready.json", {"startup_s": round(time.time() - t0, 1)})
    log(f"vLLM ready after {round(time.time() - t0)} s")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    import importlib.metadata as md

    runtime = {"gpu": gpu.strip().splitlines(), "python": sys.version.split()[0],
               "packages": {p: md.version(p) for p in ["vllm", "lm-eval", "torch", "transformers", "datasets", "math-verify", "sympy", "nltk"]}}
    save("runtime.json", runtime)

    manifest = json.loads((WORK / "manifest" / "eval-EXP-E-002.json").read_text())
    manifest["_sha256"] = sha256(WORK / "manifest" / "eval-EXP-E-002.json")

    # The installed harness must be byte-identical to the pinned files in the manifest.
    import lm_eval

    hroot = Path(lm_eval.__file__).resolve().parent
    hdiff = {f: (sha256(hroot / f) if (hroot / f).exists() else None)
             for f, h in manifest["harness"]["file_sha256"].items()
             if not (hroot / f).exists() or sha256(hroot / f) != h}
    save("harness-verify.json", {"files": len(manifest["harness"]["file_sha256"]), "mismatches": hdiff})
    if hdiff:
        raise RuntimeError(f"installed lm-eval differs from the pinned files: {sorted(hdiff)[:5]}")
    log("harness files verified")
    from qwentum_eval import run_e002

    try:
        summary = run_e002.run(RUN_MODE, OUT / "runs", f"http://127.0.0.1:{PORT}", manifest)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        slog.close()
    log(f"run complete={summary.get('complete')} id={summary.get('run_id')}")
    save("entry-result.json", {"complete": summary.get("complete"), "run_id": summary.get("run_id"),
                               "stopped": summary.get("stopped"), "error": summary.get("error")})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("entry-result.json", {"complete": False, "error": repr(exc)})
        raise
