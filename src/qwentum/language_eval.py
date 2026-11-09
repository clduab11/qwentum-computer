"""Strict text-program evaluation. No model loading or generated-code execution.

Original arm-1 responses are immutable inputs. Post-hoc host execution is a
separate arm-4 continuation, never retroactively attributed to arm 1.
"""

import hashlib
import json
import random
import re
from pathlib import Path

import stim

from qwentum.isa import Instruction
from qwentum.tableau import Machine, validate_program, RejectedInstruction, UnsupportedOperation

ARITY = {"RESET": 0, "HALT": 0, "H": 1, "S": 1, "X": 1,
         "Y": 1, "Z": 1, "M": 1, "CNOT": 2, "CX": 2, "CY": 2, "CZ": 2}


class ParseError(ValueError):
    pass


def parse_response(text: str) -> tuple[str, object]:
    """No fence stripping, case repair, eval, or implicit gate substitution."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty response")
    lines = text.strip().splitlines()
    if len(lines) == 1 and re.fullmatch(r"REJECT [^\r\n]+", lines[0]):
        reason = lines[0][7:].strip()
        if reason:
            return "rejection", reason
    if len(lines) > 256:
        raise ParseError("more than 256 instructions")
    program = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0] not in ARITY:
            raise ParseError("unknown instruction or malformed rejection")
        if len(parts) != ARITY[parts[0]] + 1:
            raise ParseError("wrong operand count")
        if any(not re.fullmatch(r"[0-9]+", arg) for arg in parts[1:]):
            raise ParseError("operands must be unsigned decimal integers")
        program.append(Instruction(parts[0], tuple(int(arg) for arg in parts[1:])))
    validate_program(program)
    return "program", program


def oracle_bits(program: list[Instruction], inject: list[int]) -> list[int]:
    """Independent Stim operations; branch bits consumed only at random M.

Unlike the older engine benchmark helper, HALT is honored here. Stim has no
native 8-qubit width restriction; callers validate the language before use.
"""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(8)
    record = []
    cursor = 0
    for ins in program:
        op, args = ins.op, ins.args
        if op == "HALT":
            break
        if op == "RESET":
            sim.reset(*range(8))
        elif op in {"H", "S", "X", "Y", "Z"}:
            getattr(sim, op.lower())(*args)
        elif op == "CNOT":
            sim.cnot(*args)
        elif op == "M":
            q = args[0]
            if sim.peek_z(q) == 0:
                bit = inject[cursor]
                cursor += 1
                sim.postselect_z(q, desired_value=bool(bit))
            else:
                bit = int(sim.measure(q))
            record.append(bit)
        elif op in {"CX", "CY", "CZ"}:
            if record[args[0]]:
                getattr(sim, op[1].lower())(args[1])
        else:
            raise ValueError(f"Unsupported oracle instruction: {op}")
    return record


def execute(program: list[Instruction], seed: int) -> dict:
    rng = random.Random(seed)
    inject = [rng.randrange(2) for _ in range(256)]
    try:
        bits = Machine().run(program, inject=inject)
        engine = {"status": "executed", "bits": bits, "exception_class": None}
    except Exception as exc:
        engine = {"status": "exception", "bits": None,
                  "exception_class": type(exc).__name__, "detail": str(exc)}
    try:
        bits = oracle_bits(program, inject)
        oracle = {"status": "executed", "version": stim.__version__, "bits": bits,
                  "exception_class": None}
    except Exception as exc:
        oracle = {"status": "exception", "version": stim.__version__, "bits": None,
                  "exception_class": type(exc).__name__, "detail": str(exc)}
    agreement = (engine["bits"] == oracle["bits"]
                 if engine["status"] == oracle["status"] == "executed" else None)
    return {"route": {"called": True, "function": "qwentum.tableau.Machine.run",
                      "mode": "automatic_host_dispatch", "autonomous_tool_selection": False},
            "program": [[i.op, *i.args] for i in program], "inject": inject,
            "engine": engine, "oracle": oracle, "engine_oracle_agreement": agreement}


def score_report(text: str | None, execution: dict, task_correct: bool | None) -> dict:
    """Keep semantic bit reporting separate from the declared exact sentence.

    A missing period fails format/strict end-to-end, even if the bits are faithful.
    No eval, substring success, punctuation repair, or synthesized model output.
    """
    engine = execution["engine"]
    oracle = execution["oracle"]
    match = (re.fullmatch(r"Engine measurement bits: (\[(?:[01](?:, [01])*)?\])\.?", text)
             if isinstance(text, str) else None)
    reported = json.loads(match[1]) if match else None
    engine_ran = execution["route"]["called"] and engine["status"] == "executed"
    faithful = (reported == engine["bits"] if engine_ran and text is not None else None)
    oracle_agreement = (reported == oracle["bits"]
                        if oracle["status"] == "executed" and text is not None else None)
    exact = (text == f"Engine measurement bits: {json.dumps(engine['bits'])}."
             if engine_ran and text is not None else None)
    e2e = (task_correct is True and engine_ran and faithful is True
           and oracle_agreement is True and exact is True) if text is not None else None
    return {"final_sentence": text, "reported_bits": reported,
            "report_format_pass": exact, "faithfulness": faithful,
            "report_oracle_agreement": oracle_agreement,
            "task_aware_end_to_end": e2e, "end_to_end": e2e,
            "end_to_end_policy": "task correct AND engine ran AND faithful AND oracle agreement AND exact declared sentence"}


def score_original(row: dict) -> tuple[dict, dict]:
    """Task intent and format are separate; no fidelity scores without reports."""
    original = dict(row)
    original.update({"arm": "1-q8_0", "route": {"called": False, "mode": "no_tools"},
                     "engine": {"status": "not_run"}, "oracle": {"status": "not_run"},
                     "faithfulness": None, "end_to_end": None,
                     "report_format_pass": None, "report_oracle_agreement": None,
                     "task_aware_end_to_end": None,
                     "report_status": "not_requested_program_only_prompt"})
    continuation = {"seed": row["seed"], "kind": row["kind"],
                    "arm": "4-q8_0-posthoc-continuation", "generation_reused_from_arm1": True,
                    "route": {"called": False}, "engine": {"status": "not_run"},
                    "oracle": {"status": "not_run"}, "faithfulness": None,
                    "end_to_end": None, "final_sentence": None,
                    "report_format_pass": None, "report_oracle_agreement": None,
                    "task_aware_end_to_end": None,
                    "report_status": "awaiting_separate_report_stage"}
    refused = row["text"].strip().startswith("REJECT")
    if refused:
        original["task_correctness"] = {
            "pass": row["kind"] in {"illegal_t", "qubit8"},
            "scope": "rejection_intent_only; does not waive required reason"}
    else:
        original["task_correctness"] = {"pass": None, "scope": "not_assessed"}
    try:
        kind, value = parse_response(row["text"])
        original["parse"] = {"pass": True, "type": kind}
        if kind == "program":
            if row["kind"] == "short":
                original["task_correctness"] = {
                    "pass": len(value) <= 4 and any(i.op == "M" for i in value),
                    "scope": "at most four instructions and a reachable measurement"}
                if any(i.op == "HALT" for i in value):
                    before_halt = value[:next(i for i, v in enumerate(value) if v.op == "HALT")]
                    original["task_correctness"]["pass"] = len(value) <= 4 and any(i.op == "M" for i in before_halt)
            try:
                continuation.update(execute(value, row["seed"]))
            except Exception as exc:
                # execute records Machine.run/oracle failures independently. This
                # catches a harness failure without inventing an engine exception.
                continuation["execution_error"] = {"failure_class": type(exc).__name__, "detail": str(exc)}
                continuation["report_status"] = "execution_error"
        else:
            continuation["report_status"] = "not_applicable_model_rejection_no_engine_call"
    except (ParseError, RejectedInstruction, UnsupportedOperation) as exc:
        # Preserve exact failure class. Never turn it into a substitute program.
        original["parse"] = {"pass": False, "failure_class": type(exc).__name__, "detail": str(exc)}
        continuation["report_status"] = "not_applicable_parse_failed_no_engine_call"
    continuation["parse"] = dict(original["parse"])
    continuation["task_correctness"] = dict(original["task_correctness"])
    if continuation["engine"]["status"] == "exception":
        continuation["report_status"] = "engine_exception_no_report_requested"
    return original, continuation


def build_record(raw_path: Path) -> dict:
    raw = raw_path.read_bytes()
    rows = json.loads(raw)
    assert len(rows) == 8
    assert [r["seed"] for r in rows] == list(range(2000000, 2000008))
    originals, continuations = zip(*(score_original(r) for r in rows))
    return {"schema_version": 2, "status": "offline_scored_awaiting_separate_report",
            "original_source": {"file": str(raw_path), "sha256": hashlib.sha256(raw).hexdigest()},
            "comparison_scope": "Q8_0 language frontend and external integer engine; not embedded; not untouched BF16",
            "routing_blocker": None,
            "current_transport_authority": "Chris 2026-11-07 workspace exception: qwentum-computer only may use existing Unsloth /v1/chat/completions directly; no load/switch/training authorized.",
            "original_decoding_requested": {"temperature": 0, "top_p": 1, "top_k": 1, "min_p": 0,
                "max_tokens": 512, "enable_thinking": False, "reasoning_effort": "none",
                "thinking": {"type": "disabled"}, "enable_tools": False},
            "original_request_evidence": "Settings reconstructed from prior executed command; raw file contains outputs and usage, not request bodies.",
            "original_transport": "http://127.0.0.1:8888/v1/chat/completions; prior run bypassed mandated LiteLLM path",
            "arm1": originals, "arm4_continuation": continuations,
            "summary": {"rows": 8, "parse_pass": sum(r["parse"]["pass"] for r in originals),
                "task_intent_pass": sum(r["task_correctness"]["pass"] is True for r in originals),
                "original_engine_calls": 0, "continuation_engine_calls": sum(r["route"]["called"] for r in continuations),
                "engine_oracle_matches": sum(r.get("engine_oracle_agreement") is True for r in continuations),
                "model_reports": 0, "faithfulness": None, "end_to_end": None},
            "failures": [{"seed": r["seed"], "parse": r["parse"], "task_correctness": r["task_correctness"]}
                         for r in originals if not r["parse"]["pass"] or r["task_correctness"]["pass"] is False]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = build_record(Path("experiments/records/_arm1_raw.json"))
    record["command"] = f"PYTHONPATH=src ./.venv/Scripts/python.exe -m qwentum.language_eval --output {args.output} (Git Bash)"
    record["python_version"] = __import__("sys").version
    with open(args.output, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record["summary"]))


if __name__ == "__main__":
    main()
