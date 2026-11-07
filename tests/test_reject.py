"""Milestone 1: instructions that must fail before any gate runs."""

import pytest

from qwentum.isa import CNOT, CX, H, M, T
from qwentum.tableau import Machine, RejectedInstruction, UnsupportedOperation


def test_t_is_unsupported_and_does_not_run():
    machine = Machine()
    machine.run([M(1)])
    with pytest.raises(UnsupportedOperation):
        machine.run([T(0)])
    assert machine.measurements == [0]


def test_h8_is_rejected_and_does_not_run():
    machine = Machine()
    machine.run([M(1)])
    with pytest.raises(RejectedInstruction):
        machine.run([H(8)])
    assert machine.measurements == [0]


def test_cnot_with_equal_args_is_rejected_and_does_not_run():
    machine = Machine()
    machine.run([M(1)])
    with pytest.raises(RejectedInstruction):
        machine.run([CNOT(0, 0)])
    assert machine.measurements == [0]


def test_correction_of_missing_measurement_is_rejected():
    machine = Machine()
    machine.run([M(1)])
    with pytest.raises(RejectedInstruction):
        machine.run([CX(0, 0)])
    assert machine.measurements == [0]


def test_instruction_257_is_rejected_before_execution():
    machine = Machine()
    machine.run([M(1)])
    assert machine.measurements == [0]
    with pytest.raises(RejectedInstruction):
        machine.run([H(0)] * 257)
    assert machine.measurements == [0]


def test_program_of_256_measurements_is_accepted():
    assert Machine().run([M(0)] * 256) == [0] * 256
