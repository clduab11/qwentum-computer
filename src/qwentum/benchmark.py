"""Held-out engine benchmark against Stim. No checkpoint is loaded."""

import json
import random
from pathlib import Path

import stim

from qwentum.isa import CNOT, CX, CY, CZ, H, M, RESET, S, T, X, Y, Z, Instruction
from qwentum.seeds import (
    DEVELOPMENT_START,
    DEVELOPMENT_STOP,
    HELDOUT,
    HELDOUT_START,
    HELDOUT_STOP,
)
from qwentum.tableau import Machine, RejectedInstruction, UnsupportedOperation

REPO_ROOT = Path(__file__).resolve().parents[2]
HELDOUT_PATH = REPO_ROOT / "experiments" / "circuits" / "heldout.json"
RECORD_PATH = REPO_ROOT / "experiments" / "records" / "engine-benchmark.json"
COMMAND = r".\.venv\Scripts\python -m pytest tests/test_engine_bench.py"

_UNARY = {"H": H, "S": S, "X": X, "Y": Y, "Z": Z}
_STIM_UNARY = {
    "H": lambda simulator, qubit: simulator.h(qubit),
    "S": lambda simulator, qubit: simulator.s(qubit),
    "X": lambda simulator, qubit: simulator.x(qubit),
    "Y": lambda simulator, qubit: simulator.y(qubit),
    "Z": lambda simulator, qubit: simulator.z(qubit),
}
_CORRECTION = {"CX": CX, "CY": CY, "CZ": CZ}


def program_to_json(program: list[Instruction]) -> list[list[object]]:
    return [[instruction.op, *instruction.args] for instruction in program]


def program_from_json(rows: list[list[object]]) -> list[Instruction]:
    return [Instruction(str(row[0]), tuple(int(arg) for arg in row[1:])) for row in rows]


def inject_bits(seed: int) -> list[int]:
    generator = random.Random(seed)
    return [generator.randrange(2) for _ in range(256)]


def _random_program(seed: int, width: int, operations: int) -> list[Instruction]:
    generator = random.Random(seed)
    program: list[Instruction] = []
    recorded = 0
    for _ in range(operations):
        choices = ["H", "S", "X", "Y", "Z", "M", "RESET"]
        if width >= 2:
            choices.append("CNOT")
        if recorded:
            choices.extend(("CX", "CY", "CZ"))
        kind = generator.choice(choices)
        if kind == "CNOT":
            control, target = generator.sample(range(width), 2)
            program.append(CNOT(control, target))
        elif kind == "M":
            program.append(M(generator.randrange(width)))
            recorded += 1
        elif kind == "RESET":
            program.append(RESET())
        elif kind in _CORRECTION:
            program.append(_CORRECTION[kind](generator.randrange(recorded), generator.randrange(width)))
        else:
            program.append(_UNARY[kind](generator.randrange(width)))
    program.extend(M(qubit) for qubit in range(8))
    return program


def _ghz_program() -> list[Instruction]:
    return [H(0), *[CNOT(0, qubit) for qubit in range(1, 8)], *[M(qubit) for qubit in range(8)]]


def _idle_program() -> list[Instruction]:
    return [H(0), CNOT(0, 1), *[M(qubit) for qubit in range(8)]]


def _rejections() -> list[tuple[str, list[Instruction], str]]:
    return [
        ("reject-t", [T(0)], "UnsupportedOperation"),
        ("reject-toffoli", [Instruction("CCX", (0, 1, 2))], "UnsupportedOperation"),
        ("reject-rx", [Instruction("RX", (0,))], "UnsupportedOperation"),
        ("reject-ry", [Instruction("RY", (0,))], "UnsupportedOperation"),
        ("reject-rz", [Instruction("RZ", (0,))], "UnsupportedOperation"),
        ("reject-qubit-8", [H(8)], "RejectedInstruction"),
        ("reject-cnot-equal", [CNOT(0, 0)], "RejectedInstruction"),
        ("reject-missing-correction", [CX(0, 0)], "RejectedInstruction"),
        ("reject-length-257", [H(0)] * 257, "RejectedInstruction"),
    ]


def build_heldout() -> dict[str, object]:
    """Build the held-out document from HELDOUT only."""
    seeds = iter(HELDOUT)
    circuits: list[dict[str, object]] = []
    for width in range(1, 9):
        for kind, operations in (("random-short", 3), ("random-deep", 16)):
            seed = next(seeds)
            circuits.append(
                {
                    "id": f"width-{width}-{kind}",
                    "seed": seed,
                    "width": width,
                    "kind": kind,
                    "expect": "agree",
                    "program": program_to_json(_random_program(seed, width, operations)),
                }
            )
    circuits.append(
        {
            "id": "ghz",
            "seed": next(seeds),
            "width": 8,
            "kind": "ghz",
            "expect": "agree",
            "program": program_to_json(_ghz_program()),
        }
    )
    circuits.append(
        {
            "id": "idle",
            "seed": next(seeds),
            "width": 8,
            "kind": "idle",
            "expect": "agree",
            "program": program_to_json(_idle_program()),
        }
    )
    for name, program, expect in _rejections():
        circuits.append(
            {
                "id": name,
                "seed": next(seeds),
                "width": 8,
                "kind": "rejection",
                "expect": expect,
                "program": program_to_json(program),
            }
        )
    return {
        "development_seeds": [DEVELOPMENT_START, DEVELOPMENT_STOP],
        "heldout_seeds": [HELDOUT_START, HELDOUT_STOP],
        "circuits": circuits,
    }


def stim_bits(program: list[Instruction], inject: list[int]) -> list[int]:
    simulator = stim.TableauSimulator()
    cursor = 0
    record: list[int] = []
    for instruction in program:
        op = instruction.op
        if op == "RESET":
            simulator.reset(*range(8))
        elif op in _STIM_UNARY:
            _STIM_UNARY[op](simulator, instruction.args[0])
        elif op == "CNOT":
            simulator.cnot(instruction.args[0], instruction.args[1])
        elif op == "M":
            qubit = instruction.args[0]
            if simulator.peek_z(qubit) == 0:
                bit = inject[cursor]
                cursor += 1
                simulator.postselect_z(qubit, desired_value=bool(bit))
                record.append(bit)
            else:
                record.append(int(simulator.measure(qubit)))
        elif op in {"CX", "CY", "CZ"} and record[instruction.args[0]] == 1:
            qubit = instruction.args[1]
            if op == "CX":
                simulator.x(qubit)
            elif op == "CY":
                simulator.y(qubit)
            else:
                simulator.z(qubit)
    return record


def evaluate_circuit(case: dict[str, object]) -> dict[str, object]:
    program = program_from_json(case["program"])  # type: ignore[arg-type]
    seed = int(case["seed"])  # type: ignore[arg-type]
    result: dict[str, object] = {
        "id": case["id"],
        "seed": seed,
        "width": case["width"],
        "kind": case["kind"],
        "prompt": None,
        "checkpoint_revision": None,
        "decoding_settings": None,
        "token_budget": None,
        "command": COMMAND,
        "failure_class": None,
        "status": None,
    }
    try:
        engine = Machine().run(program, inject=inject_bits(seed))
        actual = "agree"
        detail = None
    except UnsupportedOperation as exc:
        engine = None
        actual = "UnsupportedOperation"
        detail = str(exc)
    except RejectedInstruction as exc:
        engine = None
        actual = "RejectedInstruction"
        detail = str(exc)
    expect = case["expect"]
    if expect == "agree" and actual == "agree":
        oracle = stim_bits(program, inject_bits(seed))
        result["engine_bits"] = engine
        result["stim_bits"] = oracle
        if engine != oracle:
            result["status"] = "mismatch"
            result["failure_class"] = "mismatch"
        else:
            result["status"] = "agree"
    elif expect == actual:
        result["status"] = "rejected"
        result["detail"] = detail
    else:
        result["status"] = "error"
        result["failure_class"] = "unexpected-accept" if actual == "agree" else "wrong-rejection"
        result["detail"] = detail if detail is not None else engine
    return result


def build_record(heldout: dict[str, object]) -> dict[str, object]:
    cases = [evaluate_circuit(case) for case in heldout["circuits"]]  # type: ignore[union-attr]
    failures = [case for case in cases if case["failure_class"] is not None]
    return {
        "stage": "m3-engine-bench",
        "stim_version": stim.__version__,
        "development_seeds": heldout["development_seeds"],
        "heldout_seeds": heldout["heldout_seeds"],
        "prompt": None,
        "checkpoint_revision": None,
        "decoding_settings": None,
        "token_budget": None,
        "command": COMMAND,
        "case_count": len(cases),
        "failure_count": len(failures),
        "cases": cases,
        "failures": failures,
    }


def dump_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
