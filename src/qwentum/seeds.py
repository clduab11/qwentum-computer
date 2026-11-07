"""Seed ranges frozen before the held-out circuit file exists.

Development seeds and held-out seeds are disjoint. Do not edit these bounds to
make an existing held-out file look in range.
"""

DEVELOPMENT_START = 0
DEVELOPMENT_STOP = 1000
HELDOUT_START = 1_000_000
HELDOUT_STOP = 1_000_064

DEVELOPMENT = range(DEVELOPMENT_START, DEVELOPMENT_STOP)
HELDOUT = range(HELDOUT_START, HELDOUT_STOP)
