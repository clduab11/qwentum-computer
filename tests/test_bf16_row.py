"""Stage 4: one bf16 CNOT row update over the whole 8-qubit domain."""

import struct

from qwentum.bf16_row import (
    DTYPE,
    bf16_to_float,
    bf16_xor,
    cnot_row_bf16,
    cnot_row_integer,
    round_to_bf16,
)


def test_bf16_dtype_is_narrower_than_float32():
    assert DTYPE == "bf16"
    assert bf16_to_float(round_to_bf16(0.0)) == 0.0
    assert bf16_to_float(round_to_bf16(1.0)) == 1.0
    assert bf16_to_float(round_to_bf16(2.0)) == 2.0
    assert bf16_to_float(round_to_bf16(256.0)) == 256.0
    assert bf16_to_float(round_to_bf16(257.0)) == 256.0
    float32 = struct.unpack("<f", struct.pack("<f", 257.0))[0]
    assert float32 == 257.0
    assert bf16_to_float(round_to_bf16(257.0)) != float32


def test_bf16_xor_matches_integer_xor_and_stores_bf16_patterns():
    for left in (0, 1):
        for right in (0, 1):
            pattern = bf16_xor(left, right)
            assert pattern == round_to_bf16(float(left ^ right))
            assert bf16_to_float(pattern) == float(left ^ right)


def test_cnot_row_bf16_matches_integer_on_every_legal_input():
    integer_x = [0] * 8
    integer_z = [0] * 8
    bf16_x = [0] * 8
    bf16_z = [0] * 8
    for x_mask in range(256):
        x_bits = [(x_mask >> qubit) & 1 for qubit in range(8)]
        for z_mask in range(256):
            z_bits = [(z_mask >> qubit) & 1 for qubit in range(8)]
            for phase in range(4):
                for control in range(8):
                    for target in range(8):
                        if control == target:
                            continue
                        integer_phase = cnot_row_integer(
                            x_bits, z_bits, phase, control, target, integer_x, integer_z
                        )
                        bf16_phase = cnot_row_bf16(
                            x_bits, z_bits, phase, control, target, bf16_x, bf16_z
                        )
                        assert integer_phase == bf16_phase == phase
                        assert integer_x == bf16_x
                        assert integer_z == bf16_z
