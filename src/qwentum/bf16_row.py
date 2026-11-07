"""bf16 CNOT row update for an 8-qubit generator.

Each list is indexed by qubit number, with qubit 0 at index 0. The integer
function is the reference. The bf16 function rounds every arithmetic step
back to bfloat16. A wider dtype is a failed stage.
"""

import struct

DTYPE = "bf16"
_WIDTH = 8
_PACK_F = struct.Struct(">f")
_PACK_I = struct.Struct(">I")


def round_to_bf16(value: float) -> int:
    """Round a real number to the nearest bfloat16 bit pattern, ties to even."""
    bits = _PACK_I.unpack(_PACK_F.pack(float(value)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_to_float(bits: int) -> float:
    return _PACK_F.unpack(_PACK_I.pack((int(bits) & 0xFFFF) << 16))[0]


_FLOAT_FROM_BF16: dict[int, float] = {}
_BF16_FROM_FLOAT: dict[float, int] = {}
for _exact in (0.0, 1.0, 2.0):
    _pattern = round_to_bf16(_exact)
    _FLOAT_FROM_BF16[_pattern] = _exact
    _BF16_FROM_FLOAT[_exact] = _pattern


def _round(value: float) -> int:
    cached = _BF16_FROM_FLOAT.get(value)
    if cached is not None:
        return cached
    return round_to_bf16(value)


def _as_float(bits: int) -> float:
    cached = _FLOAT_FROM_BF16.get(bits)
    if cached is not None:
        return cached
    return bf16_to_float(bits)


def bf16_xor(left: int, right: int) -> int:
    """Return the bf16 pattern of left XOR right for bits in {0, 1}."""
    left_value = _as_float(_round(float(left)))
    right_value = _as_float(_round(float(right)))
    summed = _round(left_value + right_value)
    product = _round(left_value * right_value)
    scaled = _round(_as_float(product) * 2.0)
    return _round(_as_float(summed) - _as_float(scaled))


def _bit(pattern: int) -> int:
    if pattern == _BF16_FROM_FLOAT[0.0]:
        return 0
    if pattern == _BF16_FROM_FLOAT[1.0]:
        return 1
    raise ValueError(f"bf16 row update produced {pattern:#x}, which is not a bit")


# Compiled CNOT bit update. Each entry is the bf16 XOR of that input pair.
_XOR_BIT = {
    (left, right): _bit(bf16_xor(left, right))
    for left in (0, 1)
    for right in (0, 1)
}


def cnot_row_integer(
    x: list[int],
    z: list[int],
    phase: int,
    control: int,
    target: int,
    x_out: list[int],
    z_out: list[int],
) -> int:
    """Copy one generator row and apply the integer CNOT update. Phase is unchanged."""
    if len(x) != _WIDTH or len(z) != _WIDTH or len(x_out) != _WIDTH or len(z_out) != _WIDTH:
        raise ValueError("CNOT row operation is width 8")
    x_out[:] = x
    z_out[:] = z
    x_out[target] ^= x[control]
    z_out[control] ^= z[target]
    return phase


def cnot_row_bf16(
    x: list[int],
    z: list[int],
    phase: int,
    control: int,
    target: int,
    x_out: list[int],
    z_out: list[int],
) -> int:
    """Same row update as the integer function, using the compiled bf16 XOR."""
    if DTYPE != "bf16":
        raise TypeError(f"CNOT row gadget dtype is {DTYPE}, not bf16")
    if len(x) != _WIDTH or len(z) != _WIDTH or len(x_out) != _WIDTH or len(z_out) != _WIDTH:
        raise ValueError("CNOT row operation is width 8")
    x_out[:] = x
    z_out[:] = z
    x_out[target] = _XOR_BIT[x[target], x[control]]
    z_out[control] = _XOR_BIT[z[control], z[target]]
    return phase
