import sys
sys.path.insert(1, '..')
from src.cegis import Cegis
import sympy as sp
import numpy as np
import math
from src.common import get_random_A
from src.linearize import linearize
from src.log import log
from src.z3_learner import IterativeZ3Learner
from src.iter_learner import IterativeLearner



def test_linear(t):
    #q = mp.Queue()
    A = get_random_A(dim=10)
    c = Cegis(A)
    c.set_learners(IterativeZ3Learner)
    c.solve(exponent=t)


def test_nonlinear(t):
    dim = 6
    x = sp.Matrix(sp.symbols('x:%d' % dim, real=True))  # x=(x[0], .., x[dim-1])
    #xmat = sp.MatrixSymbol('xmat', 1, dim)
    # fx = [
    #    x[0] * x[1] - 2 * x[0],
    #    x[1] ** 2 - x[1]
    # ]
    # fx = [-x - 2 * x ** 2 + 3 * x ** 3 - 4 * x ** 4 + 5 * x ** 5]
    # fx = [-x + x ** 2 - x ** 4 + 5 * x ** 5]
    # fx = [
    #     - 2*x[0] - x[1] + x[0]*(x[1]**2) + x[1]**4 - x[0]**10,
    #     -x[1] - 1.5*x[0] + x[0]**2 + x[1]*x[0]*1.5,
    #     -x[2]*2 - 5*x[2]**2,
    #    -x[3] + 2*x[2]**3,
    #    -x[4] - 5*x[2]**6,
    #    -x[5] + 2.5*x[4]**5,
    #     #-x[6],
    #     #-x[7],
    #     #-x[8],
    #     #-x[9]
    # ]

    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf
    # fx = [
    #     -x[0]**3 - x[0]*x[2]**2,
    #     -x[1] - x[0]**2 * x[1],
    #     -x[2] + 3*x[0]**2*x[2] - (3*x[2])
    # ]

    # this series, till the end comes from
    # https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
    # srirams paper from 2013 (old-ish) but plenty of lyap fcns
    # and compare w sostool

    # ex 2.1
    # NOTA : use template w exponent = 2
    # solved with init guess [-I, zeros]
    # constraints: V_od == 0 and (p1#j#j==0 OR p2#j#j==0), min_sum = None, 1...4
    # verifier: synth_counterexamples_exp = [1]
    fx = [
        - x[0]**3 + 4*x[1]**3 - 6*x[2]*x[3],
        -x[0] - x[1] + x[4] ** 3,
        x[0]*x[3] - x[2] + x[3]*x[5],
        x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
        - 2*x[1]**3 - x[4] + x[5],
        -3*x[2]*x[3] - x[4]**3 - x[5]
    ]

    # ex 4.1
    # NOTA:
    # res == unsat, ex is not None. Exponents: [2]
    # fx = [
    #     - x[0]**3 + x[1],
    #     - x[0] - x[1]
    # ]

    # ex 4.2
    # NOTA: use exp = 2
    # problema: P = 0
    # fx = [
    #     -x[0]**3 - x[1]**2,
    #     x[0]*x[1] - x[1]**3
    # ]

    # ex 4.3
    # nota: we smash them
    # fx = [
    #     -x[0] - 1.5 * x[0]**2 * x[1]**3,
    #     -x[1]**3 + 0.5 * x[0]**2 * x[1]**2
    # ]

    # ex 4.4
    # NOTA: solved w exp = 2, nl=True, P diag, min_sum=0,
    # verifier: self._synth_counterexamples_exp = [1]
    # fx=[
    #     -x[0] + x[1]**3 - 3*x[2]*x[3],
    #     -x[0] - x[1]**3,
    #     x[0]*x[3] - x[2],
    #     x[0]*x[2] - x[3]**3
    # ]

    # ex 4.5 -- param
    # NOTA:
    # File "../src/utils.py", line 25, in Model2Matrix
    #     prefix, r, c = name.split('#')
    # ValueError: need more than 1 value to unpack
    # m = sp.Symbol('m', positive=True)
    # fx=[
    #     x[1],
    #     -(m+2)*x[0] - x[1]
    # ]

    # try Vd-diag and Vd-infame
    # fx = [
    #     -x[0]+2*x[1],
    #     -x[0] - x[1]
    # ]

    if len(fx) != dim:
        raise Exception('Dim is different than the actual length of fx')

    A = linearize(fx)

    c = Cegis(A)
    c.set_learners([IterativeZ3Learner])  # IterativeLearner / IterativeZ3Learner
    c.set_timeout(90)
    c.set_xs(x)
    assert c.solve(exponent=t, nl=True, fx=sp.Matrix(fx))
    P = c.get_P()

    log("P: %s" % P)


    print('---------------------\n')


if __name__ == '__main__':
    exp = 2

    test_nonlinear(exp)
    log("Done")

