"""Build and verify the immutable EXP-E-001 manifest.

The manifest pins every input that defines the protocol: artifact identity,
harness version and file hashes, dataset revisions and file hashes, the frozen
MMLU-Pro subset hash, sampler profile, caps, transport policy and the hashes of
this project's own evaluation source files. `verify_manifest` recomputes all of
it and refuses to proceed on any difference, so a protocol change cannot be
resumed under an old manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

from . import EVAL_DIR, REPO_ROOT, pins
from .records import canonical_json, sha256_bytes

OWN_SOURCES = [
    "experiments/eval/qwentum_eval/__init__.py",
    "experiments/eval/qwentum_eval/pins.py",
    "experiments/eval/qwentum_eval/guard.py",
    "experiments/eval/qwentum_eval/records.py",
    "experiments/eval/qwentum_eval/transport.py",
    "experiments/eval/qwentum_eval/math_grading.py",
    "experiments/eval/qwentum_eval/mmlu_subset.py",
    "experiments/eval/qwentum_eval/tasks.py",
    "experiments/eval/qwentum_eval/manifest.py",
    "experiments/eval/run_core4.py",
    "experiments/eval/freeze_mmlu_subset.py",
    "experiments/eval/write_manifest.py",
    "experiments/eval/inspect_run.py",
    "experiments/eval/capture_backend_audit.py",
]
PACKAGES = [
    "lm-eval", "datasets", "huggingface-hub", "math-verify", "latex2sympy2-extended",
    "antlr4-python3-runtime", "sympy", "nltk", "langdetect", "immutabledict",
    "requests", "aiohttp", "tenacity", "pyarrow", "numpy",
]


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _harness_root() -> Path:
    import lm_eval

    return Path(lm_eval.__file__).resolve().parent


def _dataset_files() -> dict:
    from huggingface_hub import hf_hub_download

    out = {}
    for key, spec in pins.DATASETS.items():
        files = {}
        for f in spec["files"]:
            p = Path(hf_hub_download(spec["repo"], f, repo_type="dataset", revision=spec["revision"]))
            files[f] = {"bytes": p.stat().st_size, "sha256": _file_sha(p)}
        out[key] = {k: v for k, v in spec.items() if k != "files"} | {"files": files}
    return out


def _nltk_files() -> dict:
    root = EVAL_DIR / "data_cache" / "nltk_data" / "tokenizers" / "punkt_tab" / "english"
    return {p.name: _file_sha(p) for p in sorted(root.iterdir()) if p.is_file()}


def build_manifest() -> dict:
    from .mmlu_subset import load_frozen

    subset = load_frozen()
    harness_root = _harness_root()
    return {
        "experiment": pins.EXP_ID,
        "profile": pins.PROFILE_ID,
        "purpose": "Qwentum Core-4, Protocol D (deployment, non-thinking, greedy) on the cached plain Q8_0 baseline",
        "artifact": pins.ARTIFACT,
        "serving": {
            "endpoint": pins.SERVER_BASE + "/v1/chat/completions",
            "expected_active_model": pins.EXPECTED_MODEL,
            "expected_gguf_variant": pins.EXPECTED_VARIANT,
            "chat_template": "applied by the server (llama-server) from the loaded GGUF; not re-rendered client-side",
            "note": "Context length and server build are read from status and llama-server logs at run time, not pinned here.",
        },
        "harness": pins.HARNESS
        | {"file_sha256": {f: _file_sha(harness_root / f) for f in pins.HARNESS_FILES}},
        "datasets": _dataset_files(),
        "nltk_punkt_tab_english_sha256": _nltk_files(),
        "mmlu_pro_subset": {
            "file": pins.MMLU_SUBSET_FILE,
            "file_sha256": _file_sha(REPO_ROOT / pins.MMLU_SUBSET_FILE),
            "question_ids_sha256": subset["question_ids_sha256"],
            "size": len(subset["question_ids"]),
            "seed": subset["seed"],
            "category_allocation": subset["category_allocation"],
        },
        "tasks": {
            "math500": {"harness_task": "minerva_math500 (renamed qwentum_minerva_math500_winsafe)", "num_fewshot": 4,
                        "primary_metric": "math_verify", "secondary_metric": "exact_match",
                        "substitution": "Windows-safe grading wrapper (approved option a): upstream process_results in a worker process, upstream timeouts disabled, 10 s per-item wall limit; timeouts scored 0 and counted in grader_timeout"},
            "gsm8k": {"harness_task": "gsm8k_cot_llama", "num_fewshot": 8, "primary_metric": "exact_match / strict-match",
                      "secondary_metric": "exact_match / flexible-extract",
                      "note": "the harness keeps the first 4 of 5 stop strings for chat requests; the dropped one is recorded per request"},
            "ifeval": {"harness_task": "ifeval", "num_fewshot": 0, "primary_metric": "prompt_level_strict_acc"},
            "mmlu_pro_1k": {"harness_task": "mmlu_pro_<subject> x14 restricted to frozen IDs (renamed qwentum_mmlu_pro_1k_<subject>)",
                            "num_fewshot": 5, "fewshot_source": "validation split, upstream subject filter, first_n",
                            "primary_metric": "exact_match / custom-extract, micro-averaged over 1000 items"},
        },
        "profile_D": pins.PROFILE_D,
        "generation_caps_max_tokens": pins.GEN_CAPS,
        "rendering": {"apply_chat_template": True, "fewshot_as_multiturn": True, "system_instruction": None},
        "repeat_penalty": "not sent (not a standard OpenAI field); server default 1.0 to be verified from llama-server log",
        "transport_policy": {
            "max_attempts_per_request": pins.MAX_RETRIES_TOTAL_ATTEMPTS,
            "request_timeout_s": pins.REQUEST_TIMEOUT_S,
            "guard_every_n_requests": pins.GUARD_EVERY_N_REQUESTS,
            "truncation_stop": {"min_responses": pins.TRUNCATION_CHECK_MIN_RESPONSES, "rate": pins.TRUNCATION_STOP_RATE,
                                "mode": pins.TRUNCATION_CHECK_MODE,
                                "note": "cap hits are counted and reported for every task regardless; they score as wrong, never as transport failures"},
            "supersedes": pins.SUPERSEDED_MANIFESTS,
            "max_wall_s_full_run": pins.MAX_WALL_S_FULL_RUN,
            "smoke": {"limit_per_task": 5, "max_wall_s": 900},
            "failed_request": "run stops; item recorded incomplete, never scored as wrong",
            "harness_request_cache": "disabled",
        },
        "core4_aggregate": "(MATH-500 math_verify + GSM8K strict-match + IFEval prompt-level strict + MMLU-Pro-1k exact_match) / 4, each on 0-100",
        "not_claimed": ["loglikelihood or perplexity metrics", "WSP think35 reproduction (Protocol R)",
                        "official leaderboard status", "derivation of the GGUF from the upstream pin"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: metadata.version(p) for p in PACKAGES},
        },
        "own_source_sha256": {f: _file_sha(REPO_ROOT / f) for f in OWN_SOURCES},
    }


def manifest_digest(m: dict) -> str:
    return sha256_bytes(canonical_json(m))


def write_manifest(path: Path) -> dict:
    m = build_manifest()
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return m


def verify_manifest(path: Path) -> dict:
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    current = build_manifest()
    diffs = sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))
    if diffs:
        raise RuntimeError(f"manifest mismatch in {diffs}; a protocol change needs a new manifest")
    p = Path(path).resolve()
    shown = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
    return {"manifest_file": shown, "manifest_sha256": manifest_digest(frozen)}
