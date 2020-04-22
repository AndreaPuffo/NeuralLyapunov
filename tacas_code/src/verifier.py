from __future__ import division

# import src.utils.linear as linear
from src.sympy_converter import sympy_converter
import numpy as np

from functools import reduce
from z3 import *
from src.log import log
from src.common import OrNotZero, model_to_vector, check_with_timeout


class Z3Verifier(object):
    def __init__(self, params):
        set_option(precision=16)  # representation only (stored exactly)

        self.A = params.A
        self.n = params.A.shape[0]

        self._iter = -1  # current pos of counterexample
        self._P_sym = params.P
        self._xs, self._xs_z3 = params.xs, [Real('x%d' % i) for i in range(self.n)]
        self._V, self._Vd, self._Vd_diag = params.V, params.Vd, params.Vd_diag

        self._solver = None

        self._last_seen_counterexample = 0

        self._synth_counterexamples_exp = [1]
        self._z3_learner = False
        self._timeout = float('inf')
        self._x_precision = True

        self._error = False

        self._max_counterexamples = 1 + len(self._synth_counterexamples_exp)
        self.counterexamples = np.empty(shape=self._max_counterexamples, dtype=object)

    def __repr__(self):
        exp = self._synth_counterexamples_exp
        exponents = ".set_exponents_counterexample(%s)" % exp if len(exp) > 0 and exp[0] > 0 else ""
        timeout = ".timeout(%s)" % self._timeout if 0 < self._timeout < float('inf') else ""
        return "z3_verifier()%s%s" % (exponents, timeout)

    def __str__(self):
        exp = self._synth_counterexamples_exp
        exponents = "^(%s) " % exp if len(exp) > 0 and any(e > 0 for e in exp) else ""
        timeout = "in t < %s s" % self._timeout if 0 < self._timeout < float('inf') else ""
        return "z3_verifier %s%s" % (exponents, timeout)

    def timeout(self, t):
        """

        :param t: time in seconds
        :return: self
        """
        try:
            self._timeout = int(t)
        except Exception as e:
            self._timeout = t
        return self

    def get_unseen_counterexamples(self):
        """

        :return: generator object, yielding all the counterexamples
        from first generated after the last call to this method until the last generated (both included and in order)
        """
        res = (c for c in self.counterexamples[self._last_seen_counterexample:self._iter + 1])
        self._last_seen_counterexample = self._iter + 1
        return res

    def set_exponents_counterexample(self, es):
        """
        Counterexamples x=(x0, x1, ..) will be generated such that their sum x0^i + x1^i + .. = 1 for i in es
        :param es: list of exponents
        :return: self
        """
        if isinstance(es, list):
            self._synth_counterexamples_exp = set(es)
        elif isinstance(es, (int, long, float)):
            self._synth_counterexamples_exp = {i for i in range(es)}
        else:
            log("Cannot accept exponents: %s" % str(es))
            self._error = True
        assert all(i >= 0 for i in self._synth_counterexamples_exp)
        self._max_counterexamples = len(self._synth_counterexamples_exp) + 1
        return self

    def z3_learner(self, b):
        """

        :param b: True to generate counterexamples for a Z3-based learner
        :return: self
        """
        self._z3_learner = b is True
        return self

    def set_precision_on_x(self, p):
        self._x_precision = And(*(_x > 10 ** (-p + 1) for _x in self._xs_z3))
        return self

    def check(self, P):
        """

        :param P: matrix
        :return: unsat if there have been found no counterexamples, sat if there have been found and unknown otherwise
        """

        self._solver = Solver()
        self._set_timeout()

        log("Verifier got P %s" % P)

        V, Vd = self._V, self._Vd
        for i in range(len(self._P_sym)):
            V = V.subs({self._P_sym[i][r, c]: P[i][r, c] for r in range(self.n) for c in range(self.n)})
            Vd = Vd.subs({self._P_sym[i][r, c]: P[i][r, c] for r in range(self.n) for c in range(self.n)})

        _, _, V = sympy_converter(V.doit()[0])
        _, _, Vd = sympy_converter(Vd.doit()[0])

        if V == 0 or Vd == 0:
            log("error: P=0")
            return sat

        fml = Implies(OrNotZero(self._xs_z3), And(V >= 0, Vd <= 0))
        res, c0 = self._prove(fml, exp=0)

        if res == sat:
            self._add_counterexample(c0)

            self._solver.push()  # keep fml
            self._generate_counterexamples(c0 is not None)
            self._solver.pop()

        return res

    def error(self):
        """

        :return: True iff the verifier has encountered an error
        """
        return self._error

    def _reallocate_counterexamples(self):
        while self._max_counterexamples <= self._iter + 1:
            self._max_counterexamples *= 2
            self.counterexamples = np.resize(self.counterexamples, self._max_counterexamples)

    def _clear_counterexamples(self):
        self.counterexamples = np.empty(shape=self._max_counterexamples, dtype=object)
        self._iter = -1
        self._last_seen_counterexample = 0

    def _add_counterexample(self, ex):
        is_new_counterexample = all(e != ex for e in self.counterexamples[:self._iter + 1]) if not self._z3_learner else \
            all(c.sexpr() != ex.sexpr() for c in self.counterexamples[:self._iter + 1])

        if is_new_counterexample:
            self._reallocate_counterexamples()
            self._iter += 1
            self.counterexamples[self._iter] = ex

    def _orthogonal_counterexamples(self, xs):
        dot = lambda xs, cx: 0 if len(xs) == 0 else xs[0] * cx[0] + dot(xs[1:], cx[1:])
        orth = (dot(xs, c) == 0 for c in self.counterexamples[:self._iter + 1])

        return And(*orth) if self._iter >= 0 else True

    def _generate_counterexamples(self, c0_exists=True):
        for exp in self._synth_counterexamples_exp:
            self._solver.push()
            self._add_counterexample_with(fml=None, c0_exists=c0_exists, exp=exp)
            self._solver.pop()

    def _add_counterexample_with(self, fml, c0_exists, exp=None):
        res, c1 = self._prove(fml=fml, exp=exp)
        if res == sat:
            self._add_counterexample(c1)
        elif res == unsat and c0_exists:
            # couldn't get a counterexample with sum(x^exp == 1), but got a counterexample to the first formula
            log("res == unsat, ex is not None. Exponents: %s" % str(self._synth_counterexamples_exp))
            self._error = True
        return res

    @staticmethod
    def _intersect_or_sym(l1, l2):
        return sorted(list(set(l1).intersection(l2)), key=lambda x: str(x))

    @staticmethod
    def _union_or_sym(l1, l2):
        return sorted(list(set(l1).union(l2)), key=lambda x: str(x))

    def _n_zero(self, v):
        return self._count_num_of(v, RealVal('0'))

    def _n_ones(self, v):
        return self._count_num_of(v, RealVal('1'))

    @staticmethod
    def _count_num_of(v, n):
        return reduce(lambda rem, item: rem + If(item == n, 1, 0), v, 0)

    @staticmethod
    def _sum(v):
        return reduce(lambda rem, item: rem + item, v, 0)

    @staticmethod
    def _sum_abs(v):
        return reduce(lambda rem, item: rem + If(item < 0, - item, item), v, 0)

    @staticmethod
    def _sum_powers(v, e):
        return reduce(lambda rem, item: rem + item ** e, v, 0)

    def _set_timeout(self):
        if 0 < self._timeout < float('inf'):
            self._solver.set("timeout", max(1, int(self._timeout) * 1000))  # s to ms
        return self

    def _prove(self, fml=None, exp=0, orth=False):
        if fml is not None:
            self._solver.add(Not(fml))
            self._solver.add(self._x_precision)

        if orth:
            self._solver.add(self._orthogonal_counterexamples(self._xs_z3))

        if exp > 0:
            self._solver.add(self._sum_powers(self._xs_z3, exp) == 1)

        log("s %s" % self._solver)
        res, self._has_timedout = check_with_timeout(self._solver, self._timeout)
        counterexample = None

        if res == unknown and not self._has_timedout:
            self._error = True
        elif res == sat:
            model = self._solver.model()
            counterexample = model_to_vector(model, self._xs_z3, N=self.n, target_z3=self._z3_learner)
            log("counterexample found #%s %s" % (str(self._iter + 1), model))
        else:
            log("proved")
        return res, counterexample
