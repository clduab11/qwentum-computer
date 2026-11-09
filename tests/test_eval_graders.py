"""Grader and subset tests for the Core-4 package. Skip outside .venv-eval."""

import json
import time

import pytest

lm_eval = pytest.importorskip("lm_eval")
pytest.importorskip("math_verify")

import test_eval_fakes  # noqa: E402,F401  (puts experiments/eval on sys.path)

from qwentum_eval import REPO_ROOT, pins  # noqa: E402
from qwentum_eval.math_grading import GraderWorker  # noqa: E402
from qwentum_eval.mmlu_subset import allocate, load_frozen, select  # noqa: E402

BS = chr(92)


def L(s: str) -> str:
    return s.replace("@", BS)


def doc(answer_latex: str) -> dict:
    sol = L(f"Therefore the answer is $@boxed{{{answer_latex}}}$.")
    from lm_eval.tasks.minerva_math.utils import last_boxed_only_string, normalize_final_answer, remove_boxed

    return {"problem": "p", "solution": sol, "answer": normalize_final_answer(remove_boxed(last_boxed_only_string(sol)))}


def resp(final_latex: str, boxed: bool = True) -> str:
    body = L(f"Work. $@boxed{{{final_latex}}}$" if boxed else "Work.")
    return body + L(f"\nFinal Answer: The final answer is ${final_latex}$. I hope it is correct.")


@pytest.fixture(scope="module")
def worker():
    w = GraderWorker(timeout_s=20)
    yield w
    w.close()


@pytest.mark.parametrize(
    "gold,pred,want_mv,want_em",
    [
        ("@frac{1}{2}", "@frac{1}{2}", 1, 1),
        ("@frac{1}{2}", "0.5", 1, 1),
        # sympy's LaTeX parser reads x(x+2) as a function call, so the upstream
        # exact_match (is_equiv) scores it 0 on every platform; math_verify is right.
        ("x^2+2x", "x(x+2)", 1, 0),
        ("x^2+2x", "2x+x^2", 1, 1),
        ("24", "25", 0, 0),
        ("2", "3", 0, 0),
    ],
)
def test_windows_safe_grader_distinguishes_equivalence(worker, gold, pred, want_mv, want_em):
    out = worker.grade(doc(gold), [resp(pred)])
    assert out["grader_timeout"] == 0 and out["grader_error"] == 0
    assert out["math_verify"] == want_mv, out
    assert out["exact_match"] == want_em, out


def test_grader_handles_missing_final_answer(worker):
    out = worker.grade(doc("7"), ["I do not know."])
    assert out == {"exact_match": 0, "math_verify": 0, "grader_timeout": 0, "grader_error": 0}


def _sleeping_worker(conn):
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        time.sleep(30)


def test_grader_timeout_is_flagged_and_worker_restarts():
    w = GraderWorker(timeout_s=1, worker_target=_sleeping_worker)
    try:
        t0 = time.perf_counter()
        out = w.grade({"problem": "p", "solution": "s", "answer": "1"}, ["x"])
        assert out == {"exact_match": 0, "math_verify": 0, "grader_timeout": 1, "grader_error": 0}
        assert time.perf_counter() - t0 < 15
        assert w.restarts == 1 and w._proc is None
    finally:
        w.close()


# ---- MMLU-Pro subset -----------------------------------------------------


def test_allocation_is_proportional_and_sums():
    counts = {"a": 1351, "b": 381, "c": 410}
    alloc = allocate(counts, 100)
    assert sum(alloc.values()) == 100 and alloc["a"] > alloc["c"] > alloc["b"]


def test_select_is_deterministic_and_stratified():
    ids = list(range(1, 301))
    cats = ["x"] * 200 + ["y"] * 100
    s1 = select(ids, cats, seed=7, size=30)
    s2 = select(ids, cats, seed=7, size=30)
    s3 = select(ids, cats, seed=8, size=30)
    assert s1["question_ids"] == s2["question_ids"] != s3["question_ids"]
    assert s1["category_allocation"] == {"x": 20, "y": 10}
    assert all(i <= 200 for i in s1["question_ids_by_category"]["x"])


def test_frozen_subset_file_matches_pins_and_hash():
    d = load_frozen()
    assert d["seed"] == pins.MMLU_SUBSET_SEED and d["size"] == pins.MMLU_SUBSET_SIZE
    assert d["revision"] == pins.DATASETS["mmlu_pro"]["revision"]
    assert d["question_ids_sha256"] == "4de5c852d91c8095b6569b2f7a5cd0c6b976e03002334eef9106cc8e6106e606"
    assert len(d["category_allocation"]) == 14


def test_manifest_verifies_and_rejects_protocol_drift(tmp_path):
    """verify_manifest must accept an identical manifest and reject any drift.

    EXP-E-001 is deprecated and its frozen v2 manifest is kept unchanged; editing
    shared source for EXP-E-002 makes it stop verifying, which is the intended
    guard. So this checks round-trip on a freshly built manifest instead.
    """
    from qwentum_eval.manifest import build_manifest, verify_manifest

    frozen_path = tmp_path / "fresh.json"
    frozen_path.write_text(json.dumps(build_manifest()), encoding="utf-8")
    ok = verify_manifest(frozen_path)
    assert len(ok["manifest_sha256"]) == 64
    m = build_manifest()
    assert m["artifact"]["sha256"] == "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"
    assert m["harness"]["commit"] == "ddd67220430a2470529f25fd5c05a576ca1057a0"
    assert m["profile_D"]["presence_penalty"] == 0 and m["profile_D"]["enable_thinking"] is False
    drifted = json.loads(frozen_path.read_text(encoding="utf-8"))
    drifted["profile_D"]["temperature"] = 0.6
    p = tmp_path / "drifted.json"
    p.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="profile_D"):
        verify_manifest(p)


def test_frozen_subset_rejects_tampering(tmp_path):
    src = json.loads((REPO_ROOT / pins.MMLU_SUBSET_FILE).read_text(encoding="utf-8"))
    src["question_ids"][0] = -1
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(src), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen(p)
