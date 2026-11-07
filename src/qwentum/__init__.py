"""Classical 8-qubit stabilizer engine. Stim is the oracle, not this package."""

from qwentum.tableau import Machine, RejectedInstruction, UnsupportedOperation

__all__ = ["Machine", "RejectedInstruction", "UnsupportedOperation"]
