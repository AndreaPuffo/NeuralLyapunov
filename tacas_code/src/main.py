import argparse
import sys
from random import randint
import sympy as sp

from src.cegis import Cegis
from src.common import createPoly, createFX
from src.linearize import linearize
from src.log import log
from src.z3_learner import IterativeZ3Learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize Lyapunov functions")
    parser.add_argument('-e', '--exponent')
    parser.add_argument('-n', '--dimension')

    args = parser.parse_args(sys.argv[1:])
    e = args.exponent
    n = args.dimension

    log("Running with e=%s n=%s" % (e, n))
    if e is None or n is None:
        log("e and n must be set")
        sys.exit(1)
    e = int(e)
    n = int(n)

    roots = [- randint(0, 9) for _ in range(n)]
    p = sp.Poly(createPoly(roots))
    fx = createFX(p)

    A = linearize(fx)
    c = Cegis(A)
    c.set_learners([IterativeZ3Learner])
    assert c.solve(exponent=e)
    P = c.get_P()

    log("P: %s" % P)

    print('---------------------')
