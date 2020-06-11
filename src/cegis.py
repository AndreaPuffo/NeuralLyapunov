import torch
import numpy as np
import sympy as sp
from z3 import *
import timeit
from src.consts import LearnerType, VerifierType
from src.z3verifier import Z3Verifier
from src.drealverifier import DRealVerifier
from src.verifier import *
from src.utils import get_symbolic_formula, print_section, compute_trajectory
from src.simplified_z3_learner import SimpleZ3Learner
from src.net import NN
from src.scipy_learner import ScipyLearner
from src.sympy_converter import sympy_converter


class Cegis():
    # todo: set params for NN and avoid useless definitions
    def __init__(self, n_vars, f, learner_type, verifier_type, inner_radius, outer_radius, \
                 margin, n_hidden_neurons, activations):
        self.n = n_vars
        self.f = f
        self.learner_type = learner_type
        self.inner = inner_radius
        self.outer = outer_radius
        self.margin = margin
        self.h = n_hidden_neurons
        self.max_cegis_iter = 5

        # batch init
        self.batch_size = 500
        self.learning_rate = .1

        self.x = [Real('x%d' % i) for i in range(n_vars)]
        self.x_map = {str(x): x for x in self.x}
        # issues w/ dimensionality, maybe could be solved better
        if self.n > 1:
            self.xdot = f(self.x)
        else:
            self.xdot = f(self.x[0])
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T
        self.eq = np.zeros((self.n, 1))

        if learner_type == LearnerType.NN:
            self.learner = NN(n_vars, *n_hidden_neurons, bias=True, activate=activations)
        elif learner_type == LearnerType.Z3:
            self.learner = SimpleZ3Learner(self.n)
        elif learner_type == LearnerType.SCIPY:
            self.learner = ScipyLearner(self.n)
        else:
            print('M8 I aint got this learner')

        if verifier_type == VerifierType.Z3:
            verifier = Z3Verifier
        elif verifier_type == VerifierType.DREAL:
            verifier = DRealVerifier
        else:
            raise ValueError('No verifier of type {}'.format(verifier_type))

        self.verifier = verifier(self.n, self.eq, self.inner, self.outer, self.margin, self.x)

    # the cegis loop
    # todo: fix return, fix map(f, S)
    def solve(self):
        S = []
        for idx in range(self.batch_size):
            s = torch.normal(0, self.outer / 3, size=torch.Size([self.n])).double()
            # if inner_radius < torch.norm(s) < outer_radius:
            S.append(s)
        Sdot = list(map(torch.tensor, map(self.f, S)))

        S, Sdot = torch.stack(S), torch.stack(Sdot)

        if self.learner_type == LearnerType.NN:
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop, found = False, False
        start = timeit.default_timer()
        #
        while not stop:

            print_section('Learning', iters)
            if self.learner_type == LearnerType.NN:
                learned = self.learner.learn(self.optimizer, S, Sdot, self.margin)

                # to disable rounded numbers, set rounding=-1
                V, Vdot = get_symbolic_formula(self.learner, self.x, self.xdot, equilibrium=self.eq, rounding=3)
                V_z3 = sympy_converter(sp.simplify(V), var_map=self.x_map)
                Vdot_z3 = sympy_converter(sp.simplify(Vdot), var_map=self.x_map)
                V, Vdot = z3.simplify(V_z3), z3.simplify(Vdot_z3)
            else:
                P = self.learner.learn(S.numpy().T, Sdot.numpy().T)
                # might modify get_symbolic_formula to work with x*P*x Lyapunov candidate...
                V, Vdot = self.learner.get_poly_formula(self.x, self.xdot, P)

            print_section('Candidate', iters)
            print(f'V: {V}')
            print(f'Vdot: {Vdot}')

            print_section('Verification', iters)
            found, ces = self.verifier.verify(V, Vdot)

            if self.max_cegis_iter == iters:
                print('Out of Cegis loops')
                stop = True

            if found:
                print('Found a Lyapunov function, baby!')
                stop = True
            else:
                iters += 1
                S, Sdot = self.add_ces_to_data(S, Sdot, ces)
                # the original ctx is in the last row of ces
                trajectory = self.trajectoriser(ces[-1])
                S, Sdot = self.add_ces_to_data(S, Sdot, trajectory)

        print('Learner times: {}'.format(self.learner.get_timer()))
        print('Verifier times: {}'.format(self.verifier.get_timer()))
        return self.learner, found, iters

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        S = torch.cat([S, ces], dim=0)
        Sdot = torch.cat([Sdot, torch.stack(list(map(torch.tensor, map(self.f, ces))))], dim=0)
        return S, Sdot

    def trajectoriser(self, point):
        point.requires_grad = True
        trajectory = compute_trajectory(self.learner, point, self.f)

        return torch.stack(trajectory)
