import torch
from src.net import NN
from src.verifier import *
import numpy as np
from src.utils import get_symbolic_formula, print_section
from z3 import *
import timeit


class Cegis():
    def __init__(self, n_vars, f, inner_radius, outer_radius, margin, n_hidden_neurons):
        self.n = n_vars
        self.f = f
        self.inner = inner_radius
        self.outer = outer_radius
        self.margin = margin
        self.h = n_hidden_neurons
        self.max_cegis_iter = 50

        # batch init
        self.batch_size = 500
        self.learning_rate = .1

        self.x = [Real('x%d' % i) for i in range(n_vars)]
        # issues w/ dimensionality, maybe could be solved better
        if self.n > 1:
            self.xdot = f(self.x)
        else:
            self.xdot = f(self.x[0])
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        self.learner = NN(n_vars, *n_hidden_neurons, bias=False)
        self.verifier = Z3Verifier(self.n, self.inner, self.outer, self.margin, self.x)

    # todo: fix return, fix map(f, S)
    def solve(self):
        S = []
        for idx in range(self.batch_size):
            s = torch.normal(0, self.outer / 3, size=torch.Size([self.n]))
            # if inner_radius < torch.norm(s) < outer_radius:
            S.append(s)
        Sdot = list(map(torch.tensor, map(self.f, S)))

        S, Sdot = torch.stack(S), torch.stack(Sdot)

        self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop = False
        start = timeit.default_timer()
        #
        while not stop:

            print_section('Learning', iters)
            learned = self.learner.learn(self.optimizer, S, Sdot, self.margin)

            print_section('Candidate', iters)
            V, Vdot = get_symbolic_formula(self.learner, self.x, self.xdot)
            V, Vdot = z3.simplify(V), z3.simplify(Vdot)

            print(f'V: {V}')
            print(f'Vdot: {Vdot}')

            print_section('Verification', iters)
            found, ces = self.verifier.verify(V, Vdot)

            if self.max_cegis_iter == iters:
                print('Out of Cegis loops')
                return None, True

            if found:
                print('Found a Lyapunov function, baby!')
                stop = True
            else:
                iters += 1
                S, Sdot = self.add_ces_to_data(S, Sdot, ces)

        return self.learner, False, iters


    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S:
        :param Sdot:
        :param ces:
        :return:
        """
        S = torch.cat([S, torch.stack(ces)], dim=0)
        Sdot = torch.cat([Sdot, torch.stack(list(map(torch.tensor, map(self.f, ces))))], dim=0)
        return S, Sdot