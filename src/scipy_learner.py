from scipy.optimize import linprog
import timeit
import numpy as np
from z3 import simplify


class ScipyLearner():
    def __init__(self, n):
        self.n = n
        self.solution = None
        self.ces, self.f_ces = None, None  # values for counterexamples and f(counterexample)

    def __repr__(self):
        return "iterative_z3_learner(%s)" % self.solution

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
        """
        linprog solves constrints
        min c^T x , under 
        Ax <= b
        
        we have
        V = xPx > 0, with diagonal P we can write
          = x.^2 * p, where x.^2 is element-wise power and p is a vector with P elem
          in this case x is the counterexamples
          so to compute matrix A_V is enough to power2 the matrix of ces, transpose it, change sign
        
        similarly for Vdot, with P diag
        Vdot = fx*Px + xP*fx <= 0
             = 2* fx * x * p
             to compute matrix A_Vdot just need to multiply elem-wise ces and f_ces and transpose it
        
        Finally, stack A_V and A_Vdot one on top of the other
        """

        # NOTA: absolutely *not* efficient
        # best to instanciate ces just once and keep the constraints in memory
        A_V = -(self.ces * self.ces).T
        A_Vdot = 2*(self.ces * self.f_ces).T

        # set optim to zero for the moment, can think of what to use instead
        c = np.zeros((1, self.n))
        A = np.vstack((A_V, A_Vdot))
        b = np.zeros((1, 2*self.ces.shape[1]))  # there are a total of 2*number_of_ces constraints
        # NOTA: linprog automatically assumes positive vars, not necessarily the case with P *non* diag
        sol = linprog(c, A, b)
        p = sol.x

        # if no solution, linprog returns array of all zeros...
        if (p == np.zeros((1, self.n))).all():
            print("Learner is unsat")
            self._error = True
        else:
            # print("Learner's s: %s" % self._solver)
            res = np.diag(p)

        res_matrices = res
        return res_matrices

    def get_poly_formula(self, x, fx, P=None):
        """
        :param x: numpy array
        :param fx: numpy array
        :return: V, Vdot: z3 expr
        """
        # so it can be called from cegis with an external P
        if P is None:
            P = self.solution
        V = x.T @ P @ x
        Vdot = x.T @ P @ fx + fx.T @ P @ x
        # the output type depends on the input type...
        # todo: clean the code
        if isinstance(V, np.matrix):
            V, Vdot = V[0,0], Vdot[0,0]
        return simplify(V), simplify(Vdot)

    def get_solution(self):
        """
        :return: the current solution, computed by calling learn()
        """
        return self.solution

    def error(self):
        """
        :return: True iff the learner has encountered an error
        """
        return self._error

