"""Frozen, subject-stratified 1,000-question MMLU-Pro test subset.

Allocation is proportional to category size by the largest-remainder method
(ties broken by category name). Within a category, question IDs are sorted and
sampled with `random.Random(seed)`. The result depends only on the pinned
dataset revision, the seed and the size. It is chosen before any inference.
"""

from __future__ import annotations

import json
import random

from . import pins
from .records import canonical_json, sha256_bytes


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    n = sum(counts.values())
    quotas = {c: total * k / n for c, k in counts.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    remaining = total - sum(alloc.values())
    order = sorted(counts, key=lambda c: (-(quotas[c] - alloc[c]), c))
    for c in order[:remaining]:
        alloc[c] += 1
    return alloc


def select(question_ids: list[int], categories: list[str], *, seed: int, size: int) -> dict:
    by_cat: dict[str, list[int]] = {}
    for qid, cat in zip(question_ids, categories, strict=True):
        by_cat.setdefault(cat, []).append(int(qid))
    counts = {c: len(v) for c, v in by_cat.items()}
    alloc = allocate(counts, size)
    rng = random.Random(seed)
    chosen: dict[str, list[int]] = {}
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        chosen[cat] = sorted(rng.sample(ids, alloc[cat]))
    all_ids = sorted(i for v in chosen.values() for i in v)
    return {
        "seed": seed,
        "size": size,
        "method": "proportional largest-remainder allocation by category; random.Random(seed).sample over sorted IDs per category, categories in sorted order",
        "category_population": dict(sorted(counts.items())),
        "category_allocation": dict(sorted(alloc.items())),
        "question_ids_by_category": chosen,
        "question_ids": all_ids,
        "question_ids_sha256": sha256_bytes(canonical_json(all_ids)),
    }


def build_from_dataset() -> dict:
    import datasets

    spec = pins.DATASETS["mmlu_pro"]
    test = datasets.load_dataset(spec["repo"], revision=spec["revision"], split="test")
    if test.num_rows != spec["expected_rows"]["test"]:
        raise ValueError(f"MMLU-Pro test has {test.num_rows} rows, expected {spec['expected_rows']['test']}")
    if len(set(test["question_id"])) != test.num_rows:
        raise ValueError("MMLU-Pro question_id is not unique")
    out = select(test["question_id"], test["category"], seed=pins.MMLU_SUBSET_SEED, size=pins.MMLU_SUBSET_SIZE)
    out = {
        "dataset": spec["repo"],
        "revision": spec["revision"],
        "split": "test",
        "frozen_before_inference": True,
        **out,
    }
    return out


def load_frozen(path=None) -> dict:
    from . import REPO_ROOT

    p = REPO_ROOT / (path or pins.MMLU_SUBSET_FILE)
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = data["question_ids"]
    if sha256_bytes(canonical_json(ids)) != data["question_ids_sha256"]:
        raise ValueError("frozen MMLU-Pro subset hash mismatch")
    if len(ids) != pins.MMLU_SUBSET_SIZE or len(set(ids)) != len(ids):
        raise ValueError("frozen MMLU-Pro subset has the wrong size or duplicates")
    if sum(data["category_allocation"].values()) != len(ids):
        raise ValueError("category allocation does not sum to the subset size")
    by_cat = data["question_ids_by_category"]
    if sorted(i for v in by_cat.values() for i in v) != ids:
        raise ValueError("per-category IDs do not match the flat ID list")
    if {c: len(v) for c, v in by_cat.items()} != data["category_allocation"]:
        raise ValueError("per-category counts do not match the allocation")
    return data
