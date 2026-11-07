"""Full-register reset keeps the measurement record. HALT stops the program."""

import stim

from qwentum.isa import CX, HALT, M, RESET, X
from qwentum.tableau import Machine


def test_reset_restores_zero_and_keeps_earlier_measurements():
    program = [X(0), M(0), RESET(), M(0), CX(0, 1), M(1)]
    simulator = stim.TableauSimulator()
    simulator.x(0)
    first = int(simulator.measure(0))
    simulator.reset(*range(8))
    second = int(simulator.measure(0))
    if first:
        simulator.x(1)
    third = int(simulator.measure(1))
    assert Machine().run(program) == [first, second, third] == [1, 0, 1]


def test_halt_skips_later_instructions():
    assert Machine().run([HALT(), X(0), M(0)]) == []
    assert Machine().run([X(0), HALT(), M(0)]) == []
    assert Machine().run([X(0), M(0), HALT()]) == [1]
