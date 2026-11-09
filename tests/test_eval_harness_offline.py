"""End-to-end offline run: pinned harness tasks -> guarded transport -> fake server.

Proves prompt rendering, per-task caps, stop handling, record capture, grading
and the run script's refusal paths without any real model. Skips outside
.venv-eval. The fake answers are chosen so each grader has one right and some
wrong items; the resulting numbers are plumbing checks, not benchmark scores.
"""

import json
import re
import sys

import pytest

lm_eval = pytest.importorskip("lm_eval")

from test_eval_fakes import FakeServer, chat_body, last_user_text, read_jsonl  # noqa: E402

from qwentum_eval import REPO_ROOT, pins  # noqa: E402
from qwentum_eval import tasks as core4  # noqa: E402
from qwentum_eval.records import RunRecorder  # noqa: E402
from qwentum_eval.transport import GuardedChatCompletion  # noqa: E402


@pytest.fixture
def server():
    with FakeServer() as s:
        yield s


def run_task(server, tmp_path, name, limit, responder, only=None):
    from lm_eval.evaluator import simple_evaluate

    server.state.responder = responder
    rec = RunRecorder(tmp_path / name)
    lm = GuardedChatCompletion(server_base=server.base, recorder=rec, task_label=name, gen_cap=pins.GEN_CAPS[name], retry_sleep_s=0)
    objs = core4.build_tasks(name)
    if only:
        objs = {k: v for k, v in objs.items() if k.endswith(only)}
    res = simple_evaluate(model=lm, tasks=list(objs.values()), limit=limit, apply_chat_template=True,
                          fewshot_as_multiturn=True, log_samples=True, bootstrap_iters=0,
                          random_seed=1, numpy_random_seed=1, torch_random_seed=1, fewshot_random_seed=1)
    rec.close()
    return res, read_jsonl(rec.run_dir / "chat-records.jsonl"), lm


def test_math500_offline(server, tmp_path):
    ds = core4.build_tasks("math500")["qwentum_minerva_math500_winsafe"]
    gold = [d["answer"] for d in list(ds.eval_docs)[:3]]
    seen = []

    def responder(p):
        q = last_user_text(p)
        seen.append(q)
        ans = gold[len(seen) - 1] if len(seen) <= 2 else "999999"
        return chat_body(f"Solution: $\\boxed{{{ans}}}$\nFinal Answer: The final answer is ${ans}$. I hope it is correct.")

    res, recs, lm = run_task(server, tmp_path, "math500", 3, responder)
    r = res["results"]["qwentum_minerva_math500_winsafe"]
    assert r["math_verify,none"] == pytest.approx(2 / 3)
    assert r["exact_match,none"] == pytest.approx(2 / 3)
    assert r["grader_timeout,none"] == 0 and r["grader_error,none"] == 0
    assert len(recs) == 3 and all(x["request"]["max_tokens"] == 4096 for x in recs)
    msgs = recs[0]["request"]["messages"]
    assert sum(m["role"] == "user" for m in msgs) == 5 and sum(m["role"] == "assistant" for m in msgs) == 4
    assert msgs[0]["content"].startswith("Problem:") and "Final Answer:" in msgs[1]["content"]
    assert recs[0]["request"]["stop"] == ["Problem:"]


def test_gsm8k_offline(server, tmp_path):
    ds = core4.build_tasks("gsm8k")["gsm8k_cot_llama"]
    docs = list(ds.eval_docs)[:4]
    golds = [d["answer"].split("####")[-1].strip() for d in docs]
    n = [0]

    def responder(p):
        i = n[0]
        n[0] += 1
        ans = golds[i] if i < 3 else "-1"
        return chat_body(f"Reasoning. The final answer is {ans}")

    res, recs, lm = run_task(server, tmp_path, "gsm8k", 4, responder)
    r = res["results"]["gsm8k_cot_llama"]
    assert r["exact_match,strict-match"] == pytest.approx(3 / 4)
    assert r["exact_match,flexible-extract"] == pytest.approx(3 / 4)
    assert all(x["request"]["max_tokens"] == 2048 for x in recs)
    assert len(recs[0]["request"]["stop"]) == 4 and recs[0]["dropped_stop_sequences"] == ["<|im_end|>"]
    assert sum(m["role"] == "assistant" for m in recs[0]["request"]["messages"]) == 8


def test_ifeval_offline(server, tmp_path):
    def responder(p):
        return chat_body("my answer is yes.")

    res, recs, lm = run_task(server, tmp_path, "ifeval", 6, responder)
    r = res["results"]["ifeval"]
    for k in ("prompt_level_strict_acc,none", "inst_level_strict_acc,none", "prompt_level_loose_acc,none"):
        assert 0.0 <= r[k] <= 1.0
    assert len(recs) == 6 and all(x["request"]["max_tokens"] == 1280 for x in recs)
    assert "stop" not in recs[0]["request"]
    assert len(recs[0]["request"]["messages"]) == 1 and recs[0]["request"]["messages"][0]["role"] == "user"


def test_mmlu_pro_1k_offline(server, tmp_path):
    objs = core4.build_tasks("mmlu_pro_1k")
    bio = objs["qwentum_mmlu_pro_1k_biology"]
    docs = list(bio.eval_docs)
    frozen_bio = set(json.loads((REPO_ROOT / pins.MMLU_SUBSET_FILE).read_text(encoding="utf-8"))["question_ids_by_category"]["biology"])
    assert {d["question_id"] for d in docs} == frozen_bio and len(docs) == 60
    golds = [d["answer"] for d in docs[:5]]
    n = [0]

    def responder(p):
        i = n[0]
        n[0] += 1
        letter = golds[i] if i < 4 else ("A" if golds[i] != "A" else "B")
        return chat_body(f"Let's think. The answer is ({letter}).")

    res, recs, lm = run_task(server, tmp_path, "mmlu_pro_1k", 5, responder, only="_biology")
    r = res["results"]["qwentum_mmlu_pro_1k_biology"]
    assert r["exact_match,custom-extract"] == pytest.approx(4 / 5)
    msgs = recs[0]["request"]["messages"]
    assert sum(m["role"] == "assistant" for m in msgs) == 5
    assert msgs[0]["content"].startswith("The following are multiple choice questions (with answers) about biology.")
    assert re.search(r"Options:\nA\. ", msgs[-1]["content"]) and msgs[-1]["content"].endswith("Answer: Let's think step by step.")
    assert all(x["request"]["max_tokens"] == 2048 and x["request"]["stop"] == ["Question:"] for x in recs)


def test_run_script_refuses_without_approval_flag(capsys):
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "eval"))
    import run_core4

    assert run_core4.main(["--mode", "smoke"]) == 4
    assert "Refusing" in capsys.readouterr().err
    assert run_core4.main(["--mode", "smoke", "--i-have-approval", "--tasks", "bogus"]) == 4
