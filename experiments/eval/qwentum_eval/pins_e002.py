"""Frozen identities for EXP-E-002 (approved 2026-11-09).

Qwentum Core-4, Protocol D, on Qwen/Qwen3.5-0.8B at the pinned revision, served by
vLLM in float16 on Kaggle Tesla T4 GPUs. float16 because Kaggle assigned T4s
(compute capability 7.5, no bfloat16) every time, including when an L4 was
requested. float16 agreement with a bfloat16 reference has NOT been measured:
vLLM's GDN kernel refused float32 on T4 and T4 cannot run bfloat16.

Reuses EXP-E-001's harness pin, datasets, frozen MMLU-Pro subset, Protocol D
samplers and generation caps unchanged. Changing a value here needs a new manifest.
"""

from . import pins as e001

EXP_ID = "EXP-E-002"
PROFILE_ID = "qwentum-core4-D-v1-kaggle-fp16"
SUPERSEDES = {"EXP-E-001": "deprecated 2026-11-09 by Chris; its incomplete runs are kept as records and never scored"}

MODEL = {
    "repo": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "safetensors": "model.safetensors-00001-of-00001.safetensors",
    "safetensors_bytes": 1746942600,
    "safetensors_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "license": "Apache-2.0",
    "sha256_source": "measured on Kaggle in discovery v4 (experiments/records/EXP-E-002/discovery-v4/discovery/model-files.json)",
}
SERVING = {
    "engine": "vllm",
    "vllm_pin": "vllm==0.33.0",
    "dtype": "float16",
    "dtype_reason": "Kaggle T4 (cc 7.5) cannot run bfloat16; weights cast bfloat16->float16 at load",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.85,
    "generation_config": "vllm (model generation_config.json defaults NOT applied)",
    "engine_seed": 0,
    "tensor_parallel_size": 1,
    "kaggle_machine_shape": "NvidiaTeslaT4",
    "port": 8000,
}
EXPECTED_SERVED_MODEL = MODEL["repo"]
CONCURRENCY = 64

HARNESS = e001.HARNESS
HARNESS_FILES = e001.HARNESS_FILES
DATASETS = e001.DATASETS
PROFILE_D = e001.PROFILE_D
GEN_CAPS = e001.GEN_CAPS
MMLU_SUBSET_FILE = e001.MMLU_SUBSET_FILE
MMLU_SUBSET_QUESTION_IDS_SHA256 = "4de5c852d91c8095b6569b2f7a5cd0c6b976e03002334eef9106cc8e6106e606"
MANIFEST_FILE = "experiments/manifests/eval-EXP-E-002.json"

MATH_GRADER = "upstream minerva_math500 process_results unchanged (Linux; signal.SIGALRM timeouts work)"

MAX_ATTEMPTS = 2
REQUEST_TIMEOUT_S = 1800
GUARD_EVERY_N_REQUESTS = 100
TRUNCATION_CHECK_MIN_RESPONSES = e001.TRUNCATION_CHECK_MIN_RESPONSES
TRUNCATION_STOP_RATE = e001.TRUNCATION_STOP_RATE
TRUNCATION_CHECK_MODE = e001.TRUNCATION_CHECK_MODE
MAX_WALL_S_FULL_RUN = 3 * 3600
SMOKE_LIMIT = 5
SMOKE_WALL_S = 20 * 60
