"""8-qubit GHZ: one shared bit on every Z measurement."""

import stim

from qwentum.isa import CNOT, H, M
from qwentum.tableau import Machine


def _ghz_program():
    return [H(0), *[CNOT(0, qubit) for qubit in range(1, 8)], *[M(qubit) for qubit in range(8)]]


def _stim_ghz_forced(bit: int) -> list[int]:
    simulator = stim.TableauSimulator()
    simulator.h(0)
    for qubit in range(1, 8):
        simulator.cnot(0, qubit)
    simulator.postselect_z(0, desired_value=bool(bit))
    return [int(value) for value in simulator.measure_many(*range(8))]


def test_ghz_injection_is_one_shared_bit():
    program = _ghz_program()
    for bit in (0, 1):
        assert Machine().run(program, inject=[bit]) == [bit] * 8


def test_ghz_forced_branch_matches_stim():
    program = _ghz_program()
    for bit in (0, 1):
        assert Machine().run(program, inject=[bit]) == _stim_ghz_forced(bit)


def test_ghz_sampler_support_is_the_shared_bit():
    circuit = stim.Circuit()
    circuit.append("H", [0])
    for qubit in range(1, 8):
        circuit.append("CNOT", [0, qubit])
    circuit.append("M", range(8))
    rows = {tuple(int(value) for value in row) for row in circuit.compile_sampler().sample(128)}
    assert rows == {(0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 1)}
