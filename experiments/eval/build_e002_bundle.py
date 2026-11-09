"""Assemble the EXP-E-002 Kaggle dataset bundle and write the EXP-E-002 manifest.

Bundle layout (uploaded as a private Kaggle dataset):
  bundle/qwentum_eval/*.py        evaluation package
  bundle/experiments/manifests/   frozen MMLU-Pro subset (path the package expects)
  bundle/data_cache/hf/...        pinned dataset files only (HF cache layout, offline)
  bundle/data_cache/nltk_data/    punkt_tab
  bundle/manifest/eval-EXP-E-002.json
  bundle/MANIFEST.json            sha256 of every file above

The manifest file is created once (exclusive create). Run from the repo root.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwentum_eval import DATA_CACHE, REPO_ROOT  # noqa: E402
from qwentum_eval import pins_e002 as P  # noqa: E402
from qwentum_eval.manifest import _dataset_files, _file_sha, _harness_root, _nltk_files  # noqa: E402

BUNDLE = REPO_ROOT / "experiments" / "eval" / "kaggle" / "exp-e-002-bundle" / "bundle"
PKG_FILES = ["__init__.py", "pins.py", "pins_e002.py", "guard.py", "records.py", "transport.py",
             "kaggle_transport.py", "math_grading.py", "mmlu_subset.py", "tasks.py", "manifest.py", "run_e002.py"]


def build_manifest():
    pkg = REPO_ROOT / "experiments" / "eval" / "qwentum_eval"
    entry = REPO_ROOT / "experiments" / "eval" / "kaggle" / "exp-e-002-run" / "entry.py"
    root = _harness_root()
    return {
        "experiment": P.EXP_ID, "profile": P.PROFILE_ID, "supersedes": P.SUPERSEDES,
        "purpose": "Qwentum Core-4, Protocol D, Qwen/Qwen3.5-0.8B pinned revision, vLLM float16 on Kaggle T4",
        "model": P.MODEL, "serving": P.SERVING,
        "precision_caveat": "float16, not bfloat16. Kaggle assigned Tesla T4 (cc 7.5) every time, including an L4 request. "
                            "float16 vs bfloat16/float32 agreement not measured (vLLM GDN kernel rejects float32; T4 cannot run bfloat16).",
        "harness": P.HARNESS | {"file_sha256": {f: _file_sha(root / f) for f in P.HARNESS_FILES}},
        "datasets": _dataset_files(),
        "nltk_punkt_tab_english_sha256": _nltk_files(),
        "mmlu_pro_subset": {"file": P.MMLU_SUBSET_FILE, "question_ids_sha256": P.MMLU_SUBSET_QUESTION_IDS_SHA256,
                            "file_sha256": _file_sha(REPO_ROOT / P.MMLU_SUBSET_FILE)},
        "profile_D": P.PROFILE_D, "generation_caps_max_tokens": P.GEN_CAPS,
        "rendering": {"apply_chat_template": True, "fewshot_as_multiturn": True, "system_instruction": None,
                      "chat_template": "applied by vLLM from the pinned checkpoint's tokenizer files"},
        "math_grader": P.MATH_GRADER,
        "transport_policy": {"concurrency": P.CONCURRENCY, "max_attempts": P.MAX_ATTEMPTS,
                             "request_timeout_s": P.REQUEST_TIMEOUT_S, "guard_every_n_requests": P.GUARD_EVERY_N_REQUESTS,
                             "guard": "vLLM /health 200 and /v1/models == [pinned model id]",
                             "truncation_stop": {"min_responses": P.TRUNCATION_CHECK_MIN_RESPONSES, "rate": P.TRUNCATION_STOP_RATE,
                                                 "mode": P.TRUNCATION_CHECK_MODE},
                             "max_wall_s_full": P.MAX_WALL_S_FULL_RUN, "smoke": {"limit": P.SMOKE_LIMIT, "max_wall_s": P.SMOKE_WALL_S},
                             "failed_request": "run stops; incomplete, never scored", "harness_request_cache": "disabled",
                             "batching_note": "concurrent greedy requests are batched by vLLM; outputs may differ slightly between reruns"},
        "core4_aggregate": "(MATH-500 math_verify + GSM8K strict-match + IFEval prompt-level strict + MMLU-Pro-1k exact_match) / 4, 0-100",
        "not_claimed": ["bfloat16 baseline", "equivalence to the cached Q8_0 GGUF", "loglikelihood or perplexity",
                        "WSP think35 reproduction", "official leaderboard status"],
        "own_source_sha256": {f"qwentum_eval/{f}": _file_sha(pkg / f) for f in PKG_FILES}
                             | {"kaggle/exp-e-002-run/entry.py": _file_sha(entry)},
        "kaggle_runtime": "recorded per run in runtime.json (GPU, package versions) because Kaggle assigns them",
    }


def main():
    if BUNDLE.exists():
        raise SystemExit(f"{BUNDLE} exists; bundles are built once")
    pkg = REPO_ROOT / "experiments" / "eval" / "qwentum_eval"
    (BUNDLE / "qwentum_eval").mkdir(parents=True)
    for f in PKG_FILES:
        shutil.copy2(pkg / f, BUNDLE / "qwentum_eval" / f)
    sub = BUNDLE / P.MMLU_SUBSET_FILE
    sub.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / P.MMLU_SUBSET_FILE, sub)
    # pinned dataset files only, preserving the HF cache layout so offline loading resolves revisions
    hub_src = DATA_CACHE / "hf" / "hub"
    for d in hub_src.iterdir():
        if d.name.startswith("datasets--"):
            shutil.copytree(d, BUNDLE / "data_cache" / "hf" / "hub" / d.name, symlinks=False)
    ds_src = DATA_CACHE / "hf" / "datasets"
    if ds_src.exists():
        shutil.copytree(ds_src, BUNDLE / "data_cache" / "hf" / "datasets", symlinks=False)
    shutil.copytree(DATA_CACHE / "nltk_data", BUNDLE / "data_cache" / "nltk_data", symlinks=False)
    m = build_manifest()
    mpath = REPO_ROOT / P.MANIFEST_FILE
    with open(mpath, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=1)
        fh.write("\n")
    (BUNDLE / "manifest").mkdir()
    shutil.copy2(mpath, BUNDLE / "manifest" / "eval-EXP-E-002.json")
    files = {str(p.relative_to(BUNDLE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(BUNDLE.rglob("*")) if p.is_file()}
    (BUNDLE / "MANIFEST.json").write_text(json.dumps({"experiment": P.EXP_ID, "files": files}, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    print(f"bundle files={len(files)} bytes={size} manifest={mpath.relative_to(REPO_ROOT)} sha256={_file_sha(mpath)}")


if __name__ == "__main__":
    main()
