"""Host parser and execution tests; hand-authored fixtures, not model scores."""
import itertools

import pytest

from qwentum.language_eval import ParseError, parse_response, oracle_bits, score_original, score_report, execute
from qwentum.tableau import Machine, RejectedInstruction


@pytest.mark.parametrize("text", ["REJECT", "REJECT   ", "```\nM 0\n```", "m 0", "M 0\nexplanation", "M -1", "M 0.0", "", "__import__('os')"])
def test_strict_parser_rejects_without_repair(text):
    with pytest.raises(ParseError):
        parse_response(text)


def test_rejection_and_invalid_qubit():
    assert parse_response("REJECT unsupported T") == ("rejection", "unsupported T")
    with pytest.raises(RejectedInstruction):
        parse_response("H 8")
    with pytest.raises(RejectedInstruction):
        parse_response("CX 0 1")


def test_halt_and_reset_preserve_record_semantics():
    _, program = parse_response("X 0\nM 0\nRESET\nCX 0 1\nM 1\nHALT\nM 0")
    assert Machine().run(program) == oracle_bits(program, []) == [1, 1]


@pytest.mark.parametrize("width", [2, 8])
def test_bell_ghz_both_branches(width):
    text = "H 0\n" + "\n".join(f"CNOT 0 {q}" for q in range(1, width))
    text += "\n" + "\n".join(f"M {q}" for q in range(width))
    _, program = parse_response(text)
    for bit in (0, 1):
        assert Machine().run(program, [bit]) == oracle_bits(program, [bit]) == [bit] * width


@pytest.mark.parametrize("prep,basis,expected", [
    ("", "", 0), ("X 0\n", "", 1), ("H 0\n", "H 2\n", 0),
    ("X 0\nH 0\n", "H 2\n", 1), ("H 0\nS 0\n", "S 2\nZ 2\nH 2\n", 0)])
def test_teleportation_host_fixtures(prep, basis, expected):
    core = "H 1\nCNOT 1 2\nCNOT 0 1\nH 0\nM 0\nM 1\nCX 1 2\nCZ 0 2\n"
    _, program = parse_response(prep + core + basis + "M 2")
    for branch in itertools.product((0, 1), repeat=2):
        engine = Machine().run(program, branch)
        assert engine == oracle_bits(program, list(branch))
        assert engine[-1] == expected


def test_historical_rejection_is_not_engine_rejection():
    a, b = score_original({"seed": 2000003, "kind": "illegal_t", "text": "REJECT"})
    assert a["task_correctness"]["pass"]
    assert not a["parse"]["pass"]
    assert not b["route"]["called"]
    assert b["engine"]["status"] == "not_run"
    assert b["faithfulness"] is None


def test_posthoc_dispatch_never_changes_historical_route():
    a, b = score_original({"seed": 2000005, "kind": "short", "text": "M 0"})
    assert a["parse"]["pass"] and a["task_correctness"]["pass"]
    assert not a["route"]["called"]
    assert b["route"]["called"]
    assert b["engine"]["bits"] == b["oracle"]["bits"] == [0]
    assert b["faithfulness"] is None


@pytest.mark.parametrize("text,faithful,exact", [
    ("Engine measurement bits: [0].", True, True),
    ("Engine measurement bits: [0]", True, False),
    ("Engine measurement bits: [1].", False, False),
    ("Engine measurement bits: [0]. extra", False, False),
    ("Engine measurement bits: [0].\n", False, False),
    ("Engine measurement bits: [False].", False, False),
])
def test_report_format_and_semantics_stay_separate(text, faithful, exact):
    _, continuation = score_original({"seed": 2000005, "kind": "short", "text": "M 0"})
    scores = score_report(text, continuation, True)
    assert scores["faithfulness"] is faithful
    assert scores["report_oracle_agreement"] is faithful
    assert scores["report_format_pass"] is exact
    assert scores["task_aware_end_to_end"] is exact


def test_faithful_report_of_wrong_task_is_not_end_to_end_success():
    _, continuation = score_original({"seed": 2000005, "kind": "bell", "text": "M 0"})
    scores = score_report("Engine measurement bits: [0].", continuation, False)
    assert scores["faithfulness"] and scores["report_oracle_agreement"]
    assert not scores["task_aware_end_to_end"]
    scores = score_report(None, continuation, False)
    assert scores["faithfulness"] is scores["report_oracle_agreement"] is scores["end_to_end"] is None


def test_engine_and_oracle_exceptions_are_independent(monkeypatch):
    import qwentum.language_eval as evaluator
    _, program = parse_response("M 0")
    def broken_engine(*args, **kwargs):
        raise RuntimeError("fixture engine failure")
    monkeypatch.setattr(Machine, "run", broken_engine)
    result = execute(program, 2000005)
    assert result["route"]["called"]
    assert result["engine"]["exception_class"] == "RuntimeError"
    assert result["oracle"]["bits"] == [0]
    assert result["engine_oracle_agreement"] is None
    monkeypatch.undo()
    def broken_oracle(*args, **kwargs):
        raise ValueError("fixture oracle failure")
    monkeypatch.setattr(evaluator, "oracle_bits", broken_oracle)
    result = execute(program, 2000005)
    assert result["engine"]["bits"] == [0]
    assert result["oracle"]["exception_class"] == "ValueError"
    assert result["engine_oracle_agreement"] is None
