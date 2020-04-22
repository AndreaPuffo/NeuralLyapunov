from collections import namedtuple
from z3 import sat, unsat, unknown
import os

Z3Result = namedtuple('Z3Result', ['sat', 'unsat', 'unknown'])
Z3Result = Z3Result(sat=sat, unsat=unsat, unknown=unknown)

ZERO = 1e-4
SMALLEST_TOLERANCE = 1e-9

CEGIS_MAX_ITERS = float('inf')

A_MAX_DIGITS = 2
MAX_DECIMALS = 2

WOLFRAMALPHA_APP_ID = os.getenv('WOLFRAMALPHA_APP_ID', None)
