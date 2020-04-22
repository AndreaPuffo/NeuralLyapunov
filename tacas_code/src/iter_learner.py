from __future__ import division

from itertools import izip

import sympy as sp
from z3 import sat, unsat

from src.utils import diagonal_names
from src.common import is_valid_matrix_A, is_diagonal
from src import sympy_converter as convert, consts

import gurobipy as gp

from src.log import log


class IterativeLearner(object):
    def __init__(self, params):
        assert is_valid_matrix_A(params.A)
        self.A = params.A
        self.n = self.A.shape[0]
        self.iter = -1
        self.solution = None
        self.P_sym = params.P

        self.P_vars = set()
        self.P_map, self.P_sym_map = {}, {}

        self.model = gp.Model("")
        self.debug_model = False

        self._multiply_by_inv_min = False
        self._close_diagonal_by = 0
        self._min_sum = 1
        self._max_sum = None
        self._lb = - gp.GRB.INFINITY
        self._ub = gp.GRB.INFINITY
        self._timeout = 0
        self._has_timedout = False

        self._xs_sym = params.xs
        self.V_sym = params.V.doit()[0]
        self.Vd_sym = params.Vd.doit()[0]
        self.V_list_sym = map(lambda V: V.doit()[0], params.V_list)
        self.Vd_list_sym = map(lambda Vd: Vd.doit()[0], params.Vd_list)

        self._reset_model()

        self._error = False

    def __repr__(self):
        return "iterative_learner(%s)" % self.A

    def has_timedout(self):
        return self._has_timedout

    def learn(self):
        """

        :return: True iff a solution has been computed. Use get_solution() to retrieve it
        """
        if self.iter > -1:
            self.solution, self._has_timedout = self._solve(min_sum=self._min_sum, max_sum=self._max_sum)
            log("learner %s" % self.solution)
        else:
            self.solution = [-sp.Identity(self.n) for _ in range(len(self.P_sym))]
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
        return self

    def set_min_difference_diagonal(self, m):
        """

        :param m: minimum difference of elements in P's diagonal; 0 disables it
        :return: self
        """
        assert isinstance(m, (int, long, float))
        self._close_diagonal_by = m
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
        Resets the model to an empty model
        :param lb: lower bound
        :return: self
        """
        self._lb = lb
        self._reset_model()
        return self

    def set_upper_bound(self, ub):
        """
        Resets the model to an empty model
        :param ub: upper bound
        :return: self
        """
        self._ub = ub
        self._reset_model()
        return self

    def set_precision(self, _):
        return self

    def timeout(self, t):
        """

        :param t: time in seconds; 0 < t < inf
        :return: self
        """
        self._timeout = t
        self._set_timeout(self.model)
        return self

    def error(self):
        """

        :return: True iff the learner has encountered an error
        """
        return self._error

    def _set_timeout(self, model):
        if 0 < self._timeout < float('inf'):
            model.setParam("TimeLimit", max(1, int(self._timeout)))  # convert gurobi's param (double)

    # noinspection PyArgumentList
    def _add_constraint(self, counterexample):
        for V_sym, Vd_sym in izip(self.V_list_sym, self.Vd_list_sym):
            V = V_sym.subs({x: c for x, c in izip(self._xs_sym, counterexample)})
            Vd = Vd_sym.subs({x: c for x, c in izip(self._xs_sym, counterexample)})

            _, __, V = convert.sympy_converter(V, var_map=self.P_map, target=convert.TARGET_SOLVER)
            _, __, Vd = convert.sympy_converter(Vd, var_map=self.P_map, target=convert.TARGET_SOLVER)
            try:
                self.model.addConstr(V >= consts.ZERO)
                self.model.addConstr(Vd <= -consts.ZERO)
                self.model.update()
            except Exception as e:
                log('Could not add counterexample: %s. (V:%s, Vd:%s). Exception: %s' % (counterexample, V, Vd, e))

    @staticmethod
    def _valid_sol(sol):
        if sol is unsat or isinstance(sol, list) and len(sol) == 0:
            return unsat
        return sat

    def _solve(self, min_sum=None, max_sum=None):
        m0 = self.model.copy()

        res = {}

        try:
            # if min_sum is not None:
            #     sum_vars = reduce(lambda rem, item: m0.getVarByName(str(item)) + rem, self.P_vars, 0)
            #     m0.addConstr(sum_vars >= min_sum)
            #
            # if max_sum is not None:
            #     if min_sum is None:  # sum_vars already computed
            #         sum_vars = reduce(lambda rem, item: m0.getVarByName(str(item)) + rem, self.P_vars, 0)
            #     m0.addConstr(sum_vars <= max_sum)
            #
            # if self._close_diagonal_by > 0:
            #     diag = diagonal_names(self.n, prefix='P')
            #     for p in diag:
            #         for q in diag:
            #             if p != q:
            #                 vp = m0.getVarByName(p)
            #                 vq = m0.getVarByName(q)
            #                 m0.addConstr(vp - vq <= self._close_diagonal_by)
            #                 m0.addConstr(vq - vp <= self._close_diagonal_by)

            # objective = reduce(lambda rem, item: m0.getVarByName(str(item)) + rem, self.P_sym, 0)
            # m0.setObjective(objective, gp.GRB.MINIMIZE)  # optional

            self._set_timeout(m0)
            m0.update()
            self._debug_model(m0)

            m0.optimize()
            if m0.status == gp.GRB.Status.TIME_LIMIT:
                log("timed out")
                self._has_timedout = True

            else:
                # min_val = float('inf')
                for v in m0.getVars():
                    log("%s %g" % (v.varName, v.x))
                    res[v.varName] = v.x
                    # if v.x > 0:
                    #     min_val = min(min_val, v.x)

                # if self._multiply_by_inv_min:
                #     inv_min = 1 / min_val
                #     res = {k: v * inv_min for k, v in res.items()}

                log('Obj: %g' % m0.objVal)
                self._error = False

        except gp.GurobiError as e:
            log("Error code %d : %s" % (e.errno, e))
            self._error = True

        except AttributeError as e:
            log("Attribute error: %s" % e)
            self._error = True

        if self._error:
            log("Could not solve the system with min/max_sum %s %s" % (min_sum, max_sum))
            # if max_sum is not None and min_sum is not None:
            #     return self._solve(min_sum=None, max_sum=None)
            self._error = True
            return unsat, self._has_timedout

        res_matrices = []
        for P in self.P_sym:
            res_matrices.append(
                P.subs({self.P_sym_map[p]: res[p] for p in res})
            )
        return res_matrices, self._has_timedout

    def _debug_model(self, m0, filename="/tmp/gurobi.lp"):
        if self.debug_model:
            m0.write(filename)
            with open(filename) as content:
                log(content.read())

    def _reset_model(self):
        self.model = gp.Model("")
        self.model.setParam("LogToConsole", 0)

        self.P_vars = set()
        self.P_map = {}

        diag = is_diagonal(self.A) # TODO

        for P in self.P_sym:
            for p_sym in P:
                p = str(p_sym)
                if p not in self.P_vars:
                    self.P_sym_map[p] = p_sym
                    self.P_map[p] = self.model.addVar(vtype=gp.GRB.CONTINUOUS, name=p, lb=self._lb, ub=self._ub)
                    self.model.update()
                    self.P_vars.add(p)
            prefix = p[:p.find('#')]
            if diag:
                for p in self.P_vars - set(diagonal_names(self.n, prefix=prefix)):
                    self.model.addConstr(self.P_map[p] == 0)

        self.timeout(self._timeout)
