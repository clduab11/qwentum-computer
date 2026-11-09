"""EXP-E-002 end-to-end offline: pinned harness -> vLLM transport -> fake vLLM server.

Uses the Windows-safe MATH config locally (upstream grader needs SIGALRM, which
Windows lacks); the upstream-grader path is exercised for real on Kaggle.
"""

import json
import sys

import pytest

lm_eval = pytest.importorskip("lm_eval")

from test_eval_kaggle_transport import MODEL, FakeVllm  # noqa: E402

from qwentum_eval import pins_e002 as P  # noqa: E402
from qwentum_eval import tasks as core4  # noqa: E402
from qwentum_eval.kaggle_transport import VllmGuardedChatCompletion  # noqa: E402
from qwentum_eval.records import RunRecorder  # noqa: E402


def test_pins_reuse_e001_protocol_and_record_fp16():
    from qwentum_eval import pins as e001

    assert P.PROFILE_D is e001.PROFILE_D and P.GEN_CAPS is e001.GEN_CAPS and P.DATASETS is e001.DATASETS
    assert P.HARNESS["commit"] == "ddd67220430a2470529f25fd5c05a576ca1057a0"
    assert P.SERVING["dtype"] == "float16" and P.MODEL["revision"].startswith("2fc06364")
    assert P.EXPECTED_SERVED_MODEL == MODEL
    assert P.SERVING["max_model_len"] >= 4096 + 4000


def test_upstream_math_grader_refused_without_sigalrm(monkeypatch):
    import signal

    monkeypatch.delattr(signal, "SIGALRM", raising=False)
    with pytest.raises(RuntimeError, match="SIGALRM"):
        core4.math500_config(upstream_grader=True)


@pytest.mark.parametrize("name,limit,expect_turns", [("gsm8k", 3, 8), ("ifeval", 3, 0), ("mmlu_pro_1k", 3, 5), ("math500", 3, 4)])
def test_harness_through_vllm_transport(tmp_path, name, limit, expect_turns):
    from lm_eval.evaluator import simple_evaluate

    with FakeVllm() as srv:
        rec = RunRecorder(tmp_path / name)
        lm = VllmGuardedChatCompletion(
            server_base=srv.base, recorder=rec, task_label=name, gen_cap=P.GEN_CAPS[name],
            expected_model=MODEL, expected_variant="float16", concurrency=8, retry_sleep_s=0,
        )
        objs = core4.build_tasks(name)  # local: Windows-safe MATH wrapper
        if name == "mmlu_pro_1k":
            objs = {k: v for k, v in objs.items() if k.endswith("_biology")}
        res = simple_evaluate(model=lm, tasks=list(objs.values()), limit=limit, apply_chat_template=True,
                              fewshot_as_multiturn=True, log_samples=True, bootstrap_iters=0,
                              random_seed=1, numpy_random_seed=1, torch_random_seed=1, fewshot_random_seed=1)
        rec.close()
        chats = [json.loads(r[2]) for r in srv.chats]
        assert len(chats) == limit
        assert all(c["max_tokens"] == P.GEN_CAPS[name] and c["presence_penalty"] == 0 for c in chats)
        assert all(sum(m["role"] == "assistant" for m in c["messages"]) == expect_turns for c in chats)
        assert set(res["results"]) == set(objs)
        samples = [s for rows in res["samples"].values() for s in rows]
        assert len({s["doc_id"] for s in samples}) == limit
        # order preserved: each logged response echoes its own document's last user turn
        for s in samples:
            resp = s["resps"][0][0]
            last_user = [m for m in json.loads(s["arguments"][0][0])  if m["role"] == "user"][-1]["content"] if isinstance(s["arguments"][0][0], str) and s["arguments"][0][0].startswith("[") else None
            if last_user is not None:
                assert resp == f"echo:{last_user}"


def test_run_e002_smoke_writes_complete_records(tmp_path, monkeypatch):
    """run() end to end with the fake server; MATH uses the local wrapper via monkeypatch."""
    from qwentum_eval import run_e002

    orig = core4.build_tasks
    monkeypatch.setattr(core4, "build_tasks", lambda name, upstream_math_grader=False: orig(name))
    with FakeVllm() as srv:
        summary = run_e002.run("smoke", tmp_path, srv.base, {"_sha256": "test"})
    assert summary["complete"] is True, summary.get("error") or summary.get("stopped")
    run_dir = tmp_path / summary["run_id"]
    names = {p.name for p in run_dir.iterdir()}
    for t in run_e002.TASKS:
        assert f"harness-results-{t}.json" in names and f"harness-samples-{t}.jsonl" in names
    recs = [json.loads(x) for x in (run_dir / "chat-records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 20 and all(r["http_status"] == 200 for r in recs)
    assert summary["precision_note"].startswith("float16")
