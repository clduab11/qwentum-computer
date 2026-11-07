"""Milestone 1: 8-qubit Bell behavior against Stim, not tableau bytes."""

import stim

from qwentum.isa import CNOT, H, M
from qwentum.tableau import Machine


def _bell_program():
    return [H(0), CNOT(0, 1), *[M(q) for q in range(8)]]


def _stim_bell_forced(bit: int) -> list[int]:
    """Force the random Z branch with the documented postselect_z API."""
    simulator = stim.TableauSimulator()
    simulator.h(0)
    simulator.cnot(0, 1)
    simulator.postselect_z(0, desired_value=bool(bit))
    return [int(value) for value in simulator.measure_many(*range(8))]


def test_initial_state_matches_stim_zeros():
    simulator = stim.TableauSimulator()
    stim_bits = [int(value) for value in simulator.measure_many(*range(8))]
    assert Machine().run([M(q) for q in range(8)]) == stim_bits == [0] * 8


def test_injected_bell_correlation_and_idle_qubits():
    program = _bell_program()
    for bit in (0, 1):
        got = Machine().run(program, inject=[bit])
        assert got == [bit, bit, 0, 0, 0, 0, 0, 0]


def test_injected_bell_matches_stim_forced_branches():
    program = _bell_program()
    for bit in (0, 1):
        assert Machine().run(program, inject=[bit]) == _stim_bell_forced(bit)


def test_h4_is_accepted_and_only_qubit_4_is_random():
    program = [H(4), *[M(q) for q in range(8)]]
    for bit in (0, 1):
        got = Machine().run(program, inject=[bit])
        assert got[4] == bit
        assert got[:4] == [0, 0, 0, 0]
        assert got[5:] == [0, 0, 0]
