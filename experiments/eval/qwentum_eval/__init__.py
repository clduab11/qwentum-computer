"""Qwentum Core-4 evaluation glue around EleutherAI lm-evaluation-harness v0.4.13.

Importing this package pins the process environment before `datasets`, `nltk`
or `lm_eval` are imported: Hugging Face and NLTK data come only from the
project-local cache under experiments/eval/data_cache, and the Hub is offline.
Nothing here loads a model. Inference goes only through the guarded transport.
"""

import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
# On Kaggle the package and frozen files are copied into a working tree; QWENTUM_REPO_ROOT
# and QWENTUM_DATA_CACHE point at it. Locally both default to this repository.
REPO_ROOT = Path(os.environ.get("QWENTUM_REPO_ROOT", EVAL_DIR.parent.parent))
DATA_CACHE = Path(os.environ.get("QWENTUM_DATA_CACHE", EVAL_DIR / "data_cache"))


def configure_environment() -> None:
    os.environ["HF_HOME"] = str(DATA_CACHE / "hf")
    os.environ["NLTK_DATA"] = str(DATA_CACHE / "nltk_data")
    os.environ["HF_HUB_OFFLINE"] = os.environ.get("QWENTUM_HF_OFFLINE", "1")
    os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("QWENTUM_HF_OFFLINE", "1")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


configure_environment()
