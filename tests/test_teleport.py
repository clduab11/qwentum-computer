"""Teleportation scored by the destination after feed-forward."""

import stim

from qwentum.isa import CNOT, CX, CY, CZ, H, M, S, X, Y, Z
from qwentum.tableau import Machine

# Destination eigenvalue after measuring back in the preparation basis.
_CASES = (
    ("zero", (), (), 0),
    ("one", (X(0),), (), 1),
    ("plus", (H(0),), (H(2),), 0),
    ("minus", (X(0), H(0)), (H(2),), 1),
    ("y-plus", (H(0), S(0)), (S(2), Z(2), H(2)), 0),
)


def _teleport(prep, basis, *, swap_corrections: bool = False):
    program = [
        *prep,
        H(1),
        CNOT(1, 2),
        CNOT(0, 1),
        H(0),
        M(0),
        M(1),
    ]
    if swap_corrections:
        program.extend((CZ(1, 2), CX(0, 2)))
    else:
        program.extend((CX(1, 2), CZ(0, 2)))
    program.extend(basis)
    program.extend(M(qubit) for qubit in range(2, 8))
    return program


def _stim_destination_column(prep, basis, *, swap_corrections: bool = False) -> set[int]:
    circuit = stim.Circuit()
    recorded = 0
    for instruction in _teleport(prep, basis, swap_corrections=swap_corrections):
        if instruction.op in {"H", "S", "X", "Y", "Z"}:
            circuit.append(instruction.op, [instruction.args[0]])
        elif instruction.op == "CNOT":
            circuit.append("CNOT", list(instruction.args))
        elif instruction.op == "M":
            circuit.append("M", [instruction.args[0]])
            recorded += 1
        elif instruction.op in {"CX", "CZ"}:
            lookback = instruction.args[0] - recorded
            circuit.append(instruction.op, [stim.target_rec(lookback), instruction.args[1]])
        else:
            raise AssertionError(instruction.op)
    rows = circuit.compile_sampler().sample(128)
    return {int(row[2]) for row in rows}


def test_teleport_destination_is_deterministic_for_every_branch():
    for name, prep, basis, expected in _CASES:
        assert _stim_destination_column(prep, basis) == {expected}
        for first in (0, 1):
            for second in (0, 1):
                got = Machine().run(_teleport(prep, basis), inject=[first, second])
                assert got[2] == expected, name
                assert got[3:] == [0, 0, 0, 0, 0], name


def test_swapped_corrections_fail_at_least_one_preparation():
    mismatches = []
    for name, prep, basis, expected in _CASES:
        for first in (0, 1):
            for second in (0, 1):
                got = Machine().run(
                    _teleport(prep, basis, swap_corrections=True),
                    inject=[first, second],
                )
                if got[2] != expected:
                    mismatches.append(name)
    assert mismatches


def test_y_on_plus_matches_stim():
    simulator = stim.TableauSimulator()
    simulator.h(0)
    simulator.y(0)
    simulator.h(0)
    stim_bit = int(simulator.measure(0))
    got = Machine().run([H(0), Y(0), H(0), M(0)])
    assert got == [stim_bit] == [1]


def test_cy_correction_matches_stim():
    program = [H(0), M(0), CY(0, 1), M(1)]
    for bit in (0, 1):
        simulator = stim.TableauSimulator()
        simulator.h(0)
        simulator.postselect_z(0, desired_value=bool(bit))
        if bit:
            simulator.y(1)
        stim_bits = [bit, int(simulator.measure(1))]
        assert Machine().run(program, inject=[bit]) == stim_bits == [bit, bit]
