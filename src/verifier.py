import timeit
import torch
from src.utils import *
from z3 import *


class Z3Verifier():
    def __init__(self, n_vars, inner, outer, margin, z3_vars):
        self.iter = -1
        self.n = n_vars
        self.inner = inner
        self.outer = outer
        self.counterexample_n = 10
        self.margin = margin
        self._last_cex = []
        self._n_cex_to_keep = self.counterexample_n * 1
        self.x_z3s = z3_vars
        self.z3_timeout = 60


        assert (self.counterexample_n > 0)



    def verify(self, V, Vdot):
        """
        :param V: z3 expr
        :param Vdot: z3 expr
        :return:
                found_lyap: True if V is valid
                C: a list of ctx
        """
        found_lyap = False
        s = Solver()
        self.add_domain_constraints(s, V, Vdot)

        # if sat, found counterexample; if unsat, V is lyap
        res, timedout = self.solve_with_timeout(s)
        if timedout:
            print(":/ timed out")
        C = []

        if res == unsat:
            print('No counterexamples found!')
            found_lyap = True
        elif res == sat:
            original_point = self.compute_model(s)
            C = self.randomise_counterex(original_point)

        return found_lyap, C

    def add_domain_constraints(self, solver, V, Vdot):
        """
        :param solver:
        :param V:
        :param Vdot:
        :return:
        """
        circle = np.sum([x * x for x in self.x_z3s])
        V_constr = V - self.margin * circle <= 0
        Vdot_constr = Vdot + self.margin * circle > 0
        solver.add(Or(V_constr, Vdot_constr))
        # add domain constraints
        solver.add(circle > self.inner**2)
        solver.add(circle < self.outer**2)

    def solve_with_timeout(self, solver):
        """
        :param solver: z3 solver
        :return:
                res: sat if found ctx
                timedout: true if verification timed out
        """
        solver.set("timeout", max(1, self.z3_timeout * 1000))
        timer = timeit.default_timer()
        res = solver.check()
        timer = timeit.default_timer() - timer
        timedout = timer >= self.z3_timeout
        return res, timedout

    def compute_model(self, solver):
        """
        :param solver:
        :return:
        """

        model = solver.model()
        print(model)
        temp = []
        for x in self.x_z3s: # todo: x is a matrix, works w/ x[0,0] --> make it work w/ x
            try:
                temp += [float(model[x[0,0]].as_fraction())]
            except:
                temp += [float(model[x[0,0]].approx(10).as_fraction())]

        original_point = torch.tensor(temp)
        return original_point

    def randomise_counterex(self, point):
        C = []
        for i in range(20):
            random_point = point + 0.05 * torch.randn(len(point))
            if self.inner < torch.norm(random_point) < self.outer:
                C.append(random_point)
        C.append(point)
        return C



