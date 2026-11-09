"""Core-4 task configs built from the pinned harness YAML, with minimal overrides.

Every prompt, few-shot set, filter and grader comes from the installed
lm-eval 0.4.13 files (hash-checked against tag commit ddd67220). Overrides:

* all: `dataset_kwargs.revision` pins the dataset revision (the harness does not).
* minerva_math500: `process_results` and `metric_list` use the Windows-safe
  wrapper around the upstream grader (see math_grading.py).
* MMLU-Pro: 14 per-subject tasks restricted to the frozen 1,000 test IDs. The
  few-shot source keeps the upstream subject filter on the validation split.

Generation caps are enforced by the transport, not here, and recorded.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import lm_eval
from lm_eval.tasks._yaml_loader import load_yaml

from . import math_grading, pins
from .mmlu_subset import load_frozen

HARNESS_ROOT = Path(lm_eval.__file__).resolve().parent
TASKS_ROOT = HARNESS_ROOT / "tasks"

CORE4_TASKS = ("math500", "gsm8k", "ifeval", "mmlu_pro_1k")
HARNESS_TASK = {
    "math500": "minerva_math/minerva_math500.yaml",
    "gsm8k": "gsm8k/gsm8k-cot-llama.yaml",
    "ifeval": "ifeval/ifeval.yaml",
}
PRIMARY_METRIC = {
    "math500": ("math_verify", "none"),
    "gsm8k": ("exact_match", "strict-match"),
    "ifeval": ("prompt_level_strict_acc", "none"),
    "mmlu_pro_1k": ("exact_match", "custom-extract"),
}


def _with_revision(cfg: dict, dataset_key: str) -> dict:
    spec = pins.DATASETS[dataset_key]
    if cfg["dataset_path"] != spec["repo"]:
        raise ValueError(f"{cfg['task']}: dataset {cfg['dataset_path']} != pinned {spec['repo']}")
    cfg["dataset_kwargs"] = {**(cfg.get("dataset_kwargs") or {}), "revision": spec["revision"]}
    return cfg


def _load(rel: str) -> dict:
    return copy.deepcopy(load_yaml(TASKS_ROOT / rel, resolve_func=True))


def math500_config(*, upstream_grader: bool = False) -> dict:
    """upstream_grader=True (EXP-E-002, Linux): the pinned harness task, grading unchanged.

    False (EXP-E-001, Windows): the Windows-safe grading wrapper.
    """
    cfg = _with_revision(_load(HARNESS_TASK["math500"]), "math500")
    assert cfg["task"] == "minerva_math500" and cfg["output_type"] == "generate_until"
    if upstream_grader:
        import signal

        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("upstream MATH grader needs signal.SIGALRM (Linux); use the Windows-safe wrapper")
        cfg.pop("tag", None)
        return cfg
    cfg["task"] = "qwentum_minerva_math500_winsafe"
    cfg["process_results"] = math_grading.process_results
    cfg["metric_list"] = copy.deepcopy(math_grading.METRIC_LIST)
    cfg.pop("tag", None)
    return cfg


def gsm8k_config() -> dict:
    cfg = _with_revision(_load(HARNESS_TASK["gsm8k"]), "gsm8k")
    assert cfg["task"] == "gsm8k_cot_llama" and cfg["output_type"] == "generate_until"
    cfg.pop("tag", None)
    return cfg


def ifeval_config() -> dict:
    cfg = _with_revision(_load(HARNESS_TASK["ifeval"]), "ifeval")
    assert cfg["task"] == "ifeval" and cfg["output_type"] == "generate_until"
    return cfg


def _keep_ids(dataset, *, subject_filter, ids: frozenset):
    return subject_filter(dataset).filter(lambda x: x["question_id"] in ids)


def mmlu_pro_1k_configs(subset: dict | None = None) -> list[dict]:
    subset = subset or load_frozen()
    ids = frozenset(subset["question_ids"])
    cfgs = []
    for subject in pins.MMLU_PRO_SUBJECTS:
        cfg = _with_revision(_load(f"mmlu_pro/mmlu_pro_{subject}.yaml"), "mmlu_pro")
        assert cfg["task"] == f"mmlu_pro_{subject}" and cfg["output_type"] == "generate_until"
        subject_filter = cfg["process_docs"]
        cfg["task"] = f"qwentum_mmlu_pro_1k_{subject}"
        cfg["process_docs"] = functools.partial(_keep_ids, subject_filter=subject_filter, ids=ids)
        fewshot = dict(cfg.get("fewshot_config") or {})
        fewshot["process_docs"] = subject_filter  # upstream few-shot source, unchanged
        cfg["fewshot_config"] = fewshot
        cfg.pop("tag", None)
        cfgs.append(cfg)
    return cfgs


def task_configs(name: str, *, upstream_math_grader: bool = False) -> list[dict]:
    if name == "math500":
        return [math500_config(upstream_grader=upstream_math_grader)]
    if name == "gsm8k":
        return [gsm8k_config()]
    if name == "ifeval":
        return [ifeval_config()]
    if name == "mmlu_pro_1k":
        return mmlu_pro_1k_configs()
    raise KeyError(name)


def build_tasks(name: str, *, upstream_math_grader: bool = False) -> dict:
    """Instantiate ConfigurableTask objects keyed by harness task name."""
    from lm_eval.api.task import ConfigurableTask

    return {cfg["task"]: ConfigurableTask(config=cfg)
            for cfg in task_configs(name, upstream_math_grader=upstream_math_grader)}
