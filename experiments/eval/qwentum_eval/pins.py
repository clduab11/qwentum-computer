"""Frozen identities for EXP-E-001 (Qwentum Core-4, Protocol D, cached Q8_0 baseline).

Changing any value here is a protocol change: it needs a new manifest, not an edit.
"""

EXP_ID = "EXP-E-001"
PROFILE_ID = "qwentum-core4-D-v1"

SERVER_BASE = "http://127.0.0.1:8888"
EXPECTED_MODEL = "unsloth/Qwen3.5-0.8B-GGUF"
EXPECTED_VARIANT = "Q8_0"

ARTIFACT = {
    "repo": "unsloth/Qwen3.5-0.8B-GGUF",
    "revision": "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    "file": "Qwen3.5-0.8B-Q8_0.gguf",
    "path": r"C:\Users\cld-main\.cache\huggingface\hub\models--unsloth--Qwen3.5-0.8B-GGUF\snapshots\6ab461498e2023f6e3c1baea90a8f0fe38ab64d0\Qwen3.5-0.8B-Q8_0.gguf",
    "bytes": 811843840,
    "sha256": "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c",
    "upstream_source": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
    "exact_derivation_from_upstream_verified": False,
}

HARNESS = {
    "package": "lm-eval",
    "version": "0.4.13",
    "tag": "v0.4.13",
    "commit": "ddd67220430a2470529f25fd5c05a576ca1057a0",
    "repository": "https://github.com/EleutherAI/lm-evaluation-harness",
    "license": "MIT",
}

# Files whose bytes define prompts, graders and transport. Hashes are recorded
# in the manifest from the installed wheel and were checked against the tag.
HARNESS_FILES = [
    "tasks/minerva_math/minerva_math500.yaml",
    "tasks/minerva_math/minerva_math_algebra.yaml",
    "tasks/minerva_math/utils.py",
    "tasks/gsm8k/gsm8k-cot-llama.yaml",
    "tasks/ifeval/ifeval.yaml",
    "tasks/ifeval/utils.py",
    "tasks/ifeval/instructions.py",
    "tasks/ifeval/instructions_util.py",
    "tasks/ifeval/instructions_registry.py",
    "tasks/mmlu_pro/_default_template_yaml",
    "tasks/mmlu_pro/utils.py",
    "models/openai_completions.py",
    "models/api_models.py",
    "models/utils.py",
]

MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
]

DATASETS = {
    "math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "config": None,
        "split": "test",
        "files": ["test.jsonl"],
        "expected_rows": {"test": 500},
        "license": "No license field on the HF card. Derived from openai/prm800k (MIT) and hendrycks/math (MIT); basis to be recorded before publication.",
        "access": "open",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main",
        "split": "test",
        "files": ["main/test-00000-of-00001.parquet", "main/train-00000-of-00001.parquet"],
        "expected_rows": {"test": 1319, "train": 7473},
        "license": "MIT (HF card)",
        "access": "open",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "config": None,
        "split": "train",
        "files": ["ifeval_input_data.jsonl"],
        "expected_rows": {"train": 541},
        "license": "Apache-2.0 (HF card)",
        "access": "open",
    },
    "mmlu_pro": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "config": None,
        "split": "test",
        "files": ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
        "expected_rows": {"test": 12032, "validation": 70},
        "license": "MIT (HF card); GitHub repository TIGER-AI-Lab/MMLU-Pro is Apache-2.0",
        "access": "open",
    },
}

# Protocol D: deployment comparison, non-thinking, greedy. Sent on every request.
PROFILE_D = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 20261108,
    "enable_thinking": False,
}

# Generation caps (max_tokens) per Core-4 task.
GEN_CAPS = {"math500": 4096, "gsm8k": 2048, "ifeval": 1280, "mmlu_pro_1k": 2048}

MMLU_SUBSET_SEED = 20261108
MMLU_SUBSET_SIZE = 1000
MMLU_SUBSET_FILE = "experiments/manifests/eval-mmlu-pro-1k-subset.json"
# v2 (approved 2026-11-08 after the smoke run): truncation stop changed from
# "rate > 0.15, checked continuously from 50 responses" to "rate > 0.50, checked
# once at the 50th response of each task". v1 manifest is kept, superseded.
MANIFEST_FILE = "experiments/manifests/eval-EXP-E-001-v2.json"
SUPERSEDED_MANIFESTS = {"experiments/manifests/eval-EXP-E-001.json": "c36bacac7d06cbd1d4d19039e3cb2d6d4dc9efa83e5fbb080e5eccece010ba44"}

# Transport policy
MAX_RETRIES_TOTAL_ATTEMPTS = 2  # one retry; every attempt recorded
REQUEST_TIMEOUT_S = 600
GUARD_EVERY_N_REQUESTS = 25
TRUNCATION_CHECK_MIN_RESPONSES = 50
TRUNCATION_STOP_RATE = 0.50
TRUNCATION_CHECK_MODE = "once"  # evaluated a single time, when a task reaches TRUNCATION_CHECK_MIN_RESPONSES
MAX_WALL_S_FULL_RUN = 10 * 3600
MATH_GRADER_TIMEOUT_S = 10
