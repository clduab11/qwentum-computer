"""Instruction values for the 8-qubit tableau program."""

from dataclasses import dataclass


class UnsupportedOperation(Exception):
    """A gate outside the Clifford instruction set."""


class RejectedInstruction(Exception):
    """A program or operand the 8-qubit machine refuses before running."""


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[int, ...]


def _unary(op: str, qubit: int) -> Instruction:
    return Instruction(op, (qubit,))


def H(qubit: int) -> Instruction:
    return _unary("H", qubit)


def S(qubit: int) -> Instruction:
    return _unary("S", qubit)


def X(qubit: int) -> Instruction:
    return _unary("X", qubit)


def Y(qubit: int) -> Instruction:
    return _unary("Y", qubit)


def Z(qubit: int) -> Instruction:
    return _unary("Z", qubit)


def CNOT(control: int, target: int) -> Instruction:
    return Instruction("CNOT", (control, target))


def M(qubit: int) -> Instruction:
    return _unary("M", qubit)


def CX(record: int, qubit: int) -> Instruction:
    return Instruction("CX", (record, qubit))


def CY(record: int, qubit: int) -> Instruction:
    return Instruction("CY", (record, qubit))


def CZ(record: int, qubit: int) -> Instruction:
    return Instruction("CZ", (record, qubit))


def T(qubit: int) -> Instruction:
    return _unary("T", qubit)
