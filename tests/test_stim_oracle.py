"""Stim 1.16.0 is the oracle. This file does not implement a tableau."""

import stim


def _rows(batch):
    return [tuple(int(value) for value in row) for row in batch]


def test_stim_version_is_pinned():
    assert stim.__version__ == "1.16.0"


def test_eight_qubit_initial_state_measures_zero():
    circuit = stim.Circuit()
    circuit.append("M", range(8))
    batch = circuit.compile_sampler().sample(32)
    rows = _rows(batch)
    assert len(rows) == 32
    assert all(len(row) == 8 for row in rows)
    assert set(rows) == {(0, 0, 0, 0, 0, 0, 0, 0)}


def test_bell_on_eight_qubit_register_has_correlated_support():
    circuit = stim.Circuit()
    circuit.append("H", [0])
    circuit.append("CNOT", [0, 1])
    circuit.append("M", range(8))
    batch = circuit.compile_sampler().sample(256)
    rows = _rows(batch)
    assert len(rows) == 256
    pairs = {row[:2] for row in rows}
    assert pairs == {(0, 0), (1, 1)}
    assert all(row[2:] == (0, 0, 0, 0, 0, 0) for row in rows)
