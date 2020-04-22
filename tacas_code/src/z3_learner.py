from __future__ import division

from itertools import izip

import sympy as sp
from z3 import *

from src.utils import Model2Matrix
from src.common import is_valid_matrix_A, diagonal_names, is_diagonal, check_with_timeout
from src import sympy_converter as convert
from functools import reduce
from src.log import log


class IterativeZ3Learner(object):
    def __init__(self, params):
        assert is_valid_matrix_A(params.A)
        self.A = params.A
        self.n = params.A.shape[0]
        self.iter = -1
        self.solution = None
        self.xs = params.xs
        self.P_sym = params.P
        self.P_z3 = []  # flat array of P Z3's Real

        self._solver = Solver()

        self.vars_map = {}

        self._V_sym = params.V.doit()[0]
        self._Vd_sym = params.Vd.doit()[0]
        self._Vd_diag_sym = params.Vd_diag.doit()[0]
        # self._Vd_list = params.Vd_list
        self._Vd_diag_list = params.Vd_diag_list

        # the following lines impose a diagonal P if diag = True
        # or a full P if diag = False
        diag = True # is_diagonal(params.A)
        for P in params.P:
            for r in range(self.n):
                for c in range(self.n):
                    name = str(P[r, c])
                    p = Real(name)
                    if diag and r != c:
                        # self.P_sym[r, c] = 0
                        self._solver.add(p == 0)
                        self._V_sym = self._V_sym.subs(P[r, c], 0)
                        self._Vd_sym = self._Vd_sym.subs(P[r, c], 0)
                        self._Vd_diag_sym = self._Vd_diag_sym.subs(P[r, c], 0)

                    if name not in self.vars_map and (not diag or r == c):
                        self.vars_map[name] = p
                        self.P_z3.append(p)

        # this prevents a whole P_i matrix to be null
        if len(self.P_sym) > 1:
            self._P_not_zero = True
            n = len(self.A[0,:])
            for l in range(len(self.P_sym)):
                c = Or( *( self.P_z3[l*n+i] != 0  for i in range(n) ) )
                self._P_not_zero = And(c, self._P_not_zero)
        else:
            self._P_not_zero = Or(*(p != 0 for p in self.P_z3))

        self._xs_sym_map = {str(_x): _x for _x in self.xs}

        _, __, self.V_z3expr = convert.sympy_converter(self._V_sym, var_map=self.vars_map)
        _, __, self.Vd_z3expr = convert.sympy_converter(self._Vd_sym, var_map=self.vars_map)
        _, __, self.Vd_diag_z3expr = convert.sympy_converter(self._Vd_diag_sym, var_map=self.vars_map)

        #self.Vd_list_z3expr = []
        self.Vd_diag_list_z3expr = []

        for l in range(len(self._Vd_diag_list)):
            # _, __, temp = convert.sympy_converter(self._Vd_list[l][0], var_map=self.vars_map)
            # self.Vd_list_z3expr += [temp]
            _, __, temp2 = convert.sympy_converter(self._Vd_diag_list[l][0], var_map=self.vars_map)
            self.Vd_diag_list_z3expr += [temp2]

        self._multiply_by_inv_min = False
        self._close_diagonal_by = 0
        self._lb, self._ub = None, None
        self._min_sum, self._max_sum = None, None

        self._timeout = float('inf')
        self._has_timedout = False
        self._error = False

        self._set_solver()

    def __repr__(self):
        return "iterative_z3_learner(%s)" % self.A

    def has_timedout(self):
        return self._has_timedout

    def learn(self):
        """

        :return: True iff a solution has been computed. Use get_solution() to retrieve it
        """
        if self.iter > -1:
            self.solution = self._solve(min_sum=self._min_sum, max_sum=self._max_sum)
            log("learner %s" % self.solution)
        else:
            # correct solution of 6-dim [sp.diag(1, 0, 3, 3, 0, 2), sp.diag(0, 2, 0, 0, 1, 0)]
            self.solution = [ -sp.Identity(self.n), sp.zeros(self.n, self.n)  ]#[+sp.Identity(self.n) for _ in range(len(self.P_sym))]
        res = self._valid_sol(self.solution)
        self.iter += 1
        return res

    def get_solution(self):
        """

        :return: the current solution, computed by calling learn()
        """
        return self.solution

    def add_counterexample(self, ex):
        """

        :param ex: counterexample (SymPy Matrix)
        :return: self
        """
        self._add_constraint(ex)

    def set_min_difference_diagonal(self, m):
        """

        :param m: minimum difference of elements in P's diagonal; 0 disables it
        :return: self
        """
        assert isinstance(m, (int, long, float))
        self._close_diagonal_by = m
        if self._close_diagonal_by > 0:
            diag = diagonal_names(self.n, prefix='P')
            for p in diag:
                for q in diag:
                    if p != q:
                        vp = self.vars_map[p]
                        vq = self.vars_map[q]
                        self._solver.add(vp - vq <= self._close_diagonal_by)
                        self._solver.add(vq - vp <= self._close_diagonal_by)
        return self

    def set_multiply_by_inv_min(self, b):
        """

        :param b: True iff the generated P will be multiplied by its min p_ij
        :return: self
        """
        self._multiply_by_inv_min = b is True
        return self

    def set_lower_bound(self, lb):
        """

        :param lb: lower bound
        :return: self
        """
        self._lb = lb
        if self._lb is not None:
            self._solver.add(*(p >= self._lb for p in self.P_z3))
        return self

    def set_upper_bound(self, ub):
        """

        :param ub: upper bound
        :return: self
        """
        self._ub = ub
        if self._ub is not None:
            self._solver.add(*(p <= self._lb for p in self.P_z3))
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

    def _add_constraint(self, counterexample):
        # compute V and Vdot, subs the value of the ctx
        V = self._subs(self.V_z3expr, self._V_sym, counterexample)
        Vd = self._subs(self.Vd_z3expr, self._Vd_sym, counterexample)
        # Vd only diag terms
        Vd_diag = self._subs(self.Vd_diag_z3expr, self._Vd_diag_sym, counterexample)
        # Vd only off diag terms
        Vd_od = self._subs(self.Vd_diag_z3expr - self.Vd_z3expr, self._Vd_diag_sym - self._Vd_sym, counterexample)

        # self._solver.add(V >= 0)
        # NOTA: Vd < 0 is substituted below
        self._solver.add( Vd_od == 0 )

        # instead of adding a general Vd-off-diag constraint,
        # add one constraint per element in Vd_list -- should be easier and faster to synthesise
        # for t in range(len(self._Vd_diag_list)):
        #     Vd_d_t = self._subs(self.Vd_diag_list_z3expr[t], self._Vd_diag_list[t][0], counterexample)
        #     self._solver.add(Vd_d_t <= 0)

        # imposing a single diag entry among many P_i
        # constraints as Or(p1#0#0 == 0, p2#0#0 == 0)
        # generally Or(p1#i#i, ... , p(l)#i#i)
        # where i in [0, dim P], l = number of P_i matrices
        # if there is more than one P_i
        # NOTA: this ONLY works if self.P_z3 is structured
        # P_z3 = [ p1#0#0, p1#1#1, ... p1#n#n, ..., pl#0#0, ... pl#n#n ]
        if len(self.P_sym) > 1 and True: #is_diagonal(self.A):
            for k in range(len(self.A[0,:])):
                c = Or( * ( self.P_z3[k + l* len(self.A[0,:]) ] == 0 for l in range(len(self.P_sym))) )
                self._solver.add( c )


    @staticmethod
    def _valid_sol(sol):
        if sol is unsat or isinstance(sol, list) and len(sol) == 0:
            return unsat
        return sat

    def _solve(self, min_sum=None, max_sum=None):
        res = None

        self._solver.push()
        if min_sum is not None:
            sum_vars = reduce(lambda rem, item: item + rem, self.P_z3, 0)
            self._solver.add(sum_vars >= min_sum)

        if max_sum is not None:
            if min_sum is None:
                sum_vars = reduce(lambda rem, item: item + rem, self.P_z3, 0)
                self._solver.add(sum_vars <= max_sum)

        self._set_timeout()
        r, self._has_timedout = check_with_timeout(self._solver, self._timeout)
        if r == sat:
            m = self._solver.model()
            log("Learner's s: %s" % self._solver)
            log("Model %s" % m)
            res = Model2Matrix(m, self.P_sym, symmetric=True, target='z3')
            self._error = self._has_timedout
        elif r == unsat:
            log("unsat")
            self._error = True
        elif r == unknown and not self._has_timedout:
            log("unknown")
            self._error = True

        self._solver.pop()

        if self._error or self._has_timedout or res is None:
            log("Could not solve the system with min/max_sum %s %s" % (min_sum, max_sum))
            if max_sum is not None and min_sum is not None:
                return self._solve(min_sum=None, max_sum=None)
            self._error = True
            return unsat

        # try:
        #     min_val = min(*(_x for _x in res if _x > 0))
        # except:
        #     min_val = 1
        #
        # inv_min = 1 / min_val
        # if self._multiply_by_inv_min:
        #     res = res.applyfunc(lambda x: x * inv_min)

        res_matrices = res
        # for P in self.P_sym:
        #     res_matrices.append(
        #         P.subs({self.P_sym[p]: res[p] for p in res})
        #     )
        return res_matrices

    def _subs(self, e, e_sym, model):
        try:
            return model.evaluate(e)
        except:
            return convert.sympy_converter(e_sym.subs({
                self._xs_sym_map[str(k)]: v
                for k, v in izip(self.xs, model)
            }))[-1]

    def _set_solver(self):
        self._solver.add(self._P_not_zero)
        self._set_timeout()
        return self

    def _set_timeout(self):
        if 0 < self._timeout < float('inf'):
            self._solver.set("timeout", max(1, int(self._timeout) * 1000))  # s to ms
        return self
