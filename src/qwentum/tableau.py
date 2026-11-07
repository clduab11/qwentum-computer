"""8-qubit Aaronson–Gottesman destabilizer tableau.

Generators use the Pauli form i^r * X^x * Z^z. Phases are exponents of i in
{0, 1, 2, 3}. Comparison with Stim is by measurement behavior.
"""

from collections.abc import Iterable, Sequence

from qwentum.isa import Instruction, RejectedInstruction, UnsupportedOperation

__all__ = ["Machine", "RejectedInstruction", "UnsupportedOperation"]

WIDTH = 8
MAX_PROGRAM = 256
_UNSUPPORTED = frozenset({"T", "TDG", "CCX", "RX", "RY", "RZ", "SWAP"})
_UNARY = frozenset({"H", "M"})
_CORRECTIONS = frozenset({"CX"})


def _is_qubit(value: object) -> bool:
    return type(value) is int and 0 <= value < WIDTH


def _is_bit(value: object) -> bool:
    return type(value) is int and value in (0, 1)


def validate_program(program: Sequence[Instruction]) -> None:
    """Reject a program before any generator is updated."""
    if len(program) > MAX_PROGRAM:
        raise RejectedInstruction(
            f"instruction {len(program)} exceeds the {MAX_PROGRAM}-instruction program"
        )
    recorded = 0
    for instruction in program:
        op = instruction.op
        args = instruction.args
        if op in _UNSUPPORTED or op not in _UNARY | _CORRECTIONS | {"CNOT"}:
            raise UnsupportedOperation(op)
        if op in _UNARY:
            if len(args) != 1 or not _is_qubit(args[0]):
                raise RejectedInstruction(f"{op} qubit {args[0] if args else ''} is outside 0..7")
            if op == "M":
                recorded += 1
        elif op == "CNOT":
            if len(args) != 2 or not _is_qubit(args[0]) or not _is_qubit(args[1]) or args[0] == args[1]:
                raise RejectedInstruction(f"CNOT {args} is not a pair of distinct qubits in 0..7")
        else:
            record = args[0] if args else None
            if (
                len(args) != 2
                or type(record) is not int
                or record < 0
                or record >= recorded
                or not _is_qubit(args[1])
            ):
                raise RejectedInstruction(
                    f"{op} names measurement {record} which has not happened"
                )


class Machine:
    """Classical stabilizer machine. `run` starts from `|0⟩^8` after validation."""

    def __init__(self) -> None:
        self.measurements: list[int] = []
        self._reset_generators()

    def run(self, program: Sequence[Instruction], inject: Iterable[int] = ()) -> list[int]:
        validate_program(program)
        self._reset_generators()
        self.measurements = []
        pending = list(inject)
        cursor = 0
        for instruction in program:
            cursor = self._apply(instruction, pending, cursor)
        return list(self.measurements)

    def _reset_generators(self) -> None:
        n = WIDTH
        self.x = [[0] * n for _ in range(2 * n)]
        self.z = [[0] * n for _ in range(2 * n)]
        self.r = [0] * (2 * n)
        for qubit in range(n):
            self.x[qubit][qubit] = 1
            self.z[n + qubit][qubit] = 1

    def _apply(self, instruction: Instruction, pending: list[int], cursor: int) -> int:
        op = instruction.op
        if op == "H":
            self._h(instruction.args[0])
        elif op == "CNOT":
            self._cnot(instruction.args[0], instruction.args[1])
        elif op == "M":
            bit, cursor = self._measure(instruction.args[0], pending, cursor)
            self.measurements.append(bit)
        elif op == "CX" and self.measurements[instruction.args[0]] == 1:
            self._x(instruction.args[1])
        return cursor

    def _h(self, qubit: int) -> None:
        for row in range(2 * WIDTH):
            if self.x[row][qubit] and self.z[row][qubit]:
                self.r[row] = (self.r[row] + 2) % 4
            self.x[row][qubit], self.z[row][qubit] = self.z[row][qubit], self.x[row][qubit]

    def _cnot(self, control: int, target: int) -> None:
        for row in range(2 * WIDTH):
            self.x[row][target] ^= self.x[row][control]
            self.z[row][control] ^= self.z[row][target]

    def _x(self, qubit: int) -> None:
        for row in range(2 * WIDTH):
            if self.z[row][qubit]:
                self.r[row] = (self.r[row] + 2) % 4

    def _rowsum(self, target: int, source: int) -> None:
        phase = self.r[target] + self.r[source]
        for qubit in range(WIDTH):
            phase += 2 * self.z[target][qubit] * self.x[source][qubit]
            self.x[target][qubit] ^= self.x[source][qubit]
            self.z[target][qubit] ^= self.z[source][qubit]
        self.r[target] = phase % 4

    def _measure(self, qubit: int, pending: list[int], cursor: int) -> tuple[int, int]:
        pivot = next((row for row in range(WIDTH, 2 * WIDTH) if self.x[row][qubit]), None)
        if pivot is None:
            return self._deterministic(qubit), cursor
        outcome, cursor = self._injected(pending, cursor)
        for row in range(2 * WIDTH):
            if row != pivot and self.x[row][qubit]:
                self._rowsum(row, pivot)
        partner = pivot - WIDTH
        self.x[partner] = self.x[pivot][:]
        self.z[partner] = self.z[pivot][:]
        self.r[partner] = self.r[pivot]
        self.x[pivot] = [0] * WIDTH
        self.z[pivot] = [0] * WIDTH
        self.z[pivot][qubit] = 1
        self.r[pivot] = 2 * outcome
        return outcome, cursor

    def _deterministic(self, qubit: int) -> int:
        phase = 0
        scratch_x = [0] * WIDTH
        scratch_z = [0] * WIDTH
        for row in range(WIDTH):
            if not self.x[row][qubit]:
                continue
            source = row + WIDTH
            extra = 0
            for q in range(WIDTH):
                extra += scratch_z[q] * self.x[source][q]
                scratch_x[q] ^= self.x[source][q]
                scratch_z[q] ^= self.z[source][q]
            phase = (phase + self.r[source] + 2 * extra) % 4
        if phase not in (0, 2):
            raise RuntimeError(f"deterministic Z measurement produced phase {phase}")
        return phase // 2

    def _injected(self, pending: list[int], cursor: int) -> tuple[int, int]:
        if cursor >= len(pending):
            raise RejectedInstruction("random measurement has no injected bit")
        bit = pending[cursor]
        if not _is_bit(bit):
            raise RejectedInstruction(f"injected bit {bit!r} is not 0 or 1")
        return bit, cursor + 1
