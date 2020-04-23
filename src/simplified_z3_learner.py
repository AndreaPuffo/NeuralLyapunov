from __future__ import division
from z3 import *
import timeit
import numpy as np


class SimpleZ3Learner():
    def __init__(self, n):
        self.n = n
        self.solution = None
        # self.xs = xs
        self.p_z3vars = [Real('p#%d' % i) for i in range(n)]
        self.P_z3 = np.diag(self.p_z3vars)
        self._solver = Solver()
        self.ces, self.f_ces = None, None  # values for counterexamples and f(counterexample)
        self._timeout = float('inf')
        self._has_timedout = False
        self._error = False

    def __repr__(self):
        return "iterative_z3_learner(%s)" % self.P_z3

    # NOTA: passing the numerical evaluation may be comfortable but may lead to
    # issues when computing the gradient on the pytorch loss function side
    def learn(self, ces, f_ces):
        """
        :param ces:
        :param f_ces:
        :return: True iff a solution has been computed. Use get_solution() to retrieve it
        """
        self.ces = ces
        self.f_ces = f_ces
        self.solution = self._solve()
        print("Matrix P: \n%s" % self.solution)

        return self.solution

    def _solve(self):
        """
        :return: array of matrices P
        """
        res = None
        # NOTA: absolutely *not* efficient
        # best to instanciate ces just once and keep the constraints in memory
        for i in range(self.ces.shape[1]):
            self.add_counterexample(self.ces[:, i], self.f_ces[:, i])

        r, self._has_timedout = self.check_with_timeout(self._timeout)
        if r == sat:
            # print("Learner's s: %s" % self._solver)
            res = np.diag( self.compute_model() )
            self._error = self._has_timedout
        elif r == unsat:
            print("Learner is unsat")
            self._error = True
        elif r == unknown and not self._has_timedout:
            print("Learner is unknown")
            self._error = True

        res_matrices = res
        return res_matrices

    def add_counterexample(self, counterexample, f_counterexample):
        """
        :param counterexample: numpy array
        :return: None
        """
        # compute V and Vdot, subs the value of the ctx
        V, Vdot = self.get_poly_formula(counterexample, f_counterexample)

        self._solver.add(V > 0)
        self._solver.add(Vdot <= 0)

        # avoids a whole null matrix
        # if self.n > 1:
        #     for k in range(self.n):
        #         c = Or( * ( self.P_z3[k + l* self.n ] == 0 for l in range(len(self.P_sym))) )
        #         self._solver.add( c )

    def get_poly_formula(self, x, fx, P=None):
        """
        :param x: numpy array
        :param fx: numpy array
        :return: V, Vdot: z3 expr
        """
        # so it can be called from cegis with an external P
        if P is None:
            P = self.P_z3
        V = x.T @ P @ x
        Vdot = x.T @ P @ fx + fx.T @ P @ x
        # the output type depends on the input type...
        # todo: clean the code
        if isinstance(V, np.matrix):
            V, Vdot = V[0,0], Vdot[0,0]
        return z3.simplify(V), z3.simplify(Vdot)

    def check_with_timeout(self, timeout, args=None):
        """
        Runs the solver's check() method for at most t sec and asserts if it timed out.
        t sec is the minimum to return it has timed out
        :param args: [optional] arguments to pass to check
        :param solver: Z3 Solver instance
        :param timeout: number
        :return: <solver's result, timedout>
        """
        if timeout < 0:
            return unsat, True
        else:
            timer = timeit.default_timer()
            res = self._solver.check(*args) if args is not None else self._solver.check()
            timer = timeit.default_timer() - timer
            timedout = timer >= timeout
            return res, timedout

    def compute_model(self):
        """
        :param solver: z3 solver
        :return: tensor containing single ctx
        """

        model = self._solver.model()
        # print('Paramaters Found: \n{}'.format(model))
        solution = []
        for x in self.p_z3vars:
            try:
                solution += [float(model[x].as_fraction())]
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                solution += [float(model[x].approx(10).as_fraction())]

        return np.array(solution)

    def get_solution(self):
        """

        :return: the current solution, computed by calling learn()
        """
        return self.solution

    def has_timedout(self):
        return self._has_timedout

    def _set_timeout(self):
        if 0 < self._timeout < float('inf'):
            self._solver.set("timeout", max(1, int(self._timeout) * 1000))  # s to ms
        return self

    def set_precision(self, p):
        """
        Precision in the number represented (not stored)
        :param p: precision
        :return: self
        """
        set_option(precision=p)
        return self

    def timeout(self, t):
        """

        :param t: time as an integer in seconds
        :return: self
        """
        try:
            self._timeout = int(t)
        except Exception as e:
            self._timeout = t
        self._set_timeout()
        return self

    def error(self):
        """
        :return: True iff the learner has encountered an error
        """
        return self._error

