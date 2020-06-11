import torch
import timeit
import numpy as np
import sympy as sp
from src.cegis import Cegis
from src.utils import compute_bounds, compute_equilibria, check_real_solutions, dict_to_array
from src.benchmarks import benchmark_0, benchmark_3
from src.activations import ActivationType
from src.consts import VerifierType, LearnerType


torch.manual_seed(0)

n_vars = 2
f = benchmark_0
x = [sp.Symbol('x%d' % i) for i in range(n_vars)]

# compute equilibria and pass them as an array to cegis
equilibria = compute_equilibria(f(x))
real_eq = check_real_solutions(equilibria, x)
real_eq = dict_to_array(real_eq, n_vars)

# define NN parameters
activations = [ActivationType.RELU]
n_hidden_neurons = [3] * len(activations)

# define domain constraints
outer_radius = 10
inner_radius = 0

learner_type = LearnerType.NN
verifier_type = VerifierType.Z3

"""
the candidate Lyap is now (x-eq0) * ... * (x-eqN) * NN(x)
"""


start = timeit.default_timer()
c = Cegis(n_vars, f, learner_type, verifier_type, inner_radius, outer_radius, real_eq, n_hidden_neurons, activations)
c.solve()
stop = timeit.default_timer()
print('Elapsed Time: {}'.format(stop-start))

equilibrium = np.zeros((1, n_vars))
min_bound = compute_bounds(n_vars, f, equilibrium)
print('The validity bound for this system is: {}'.format(min_bound))

# todo: add main(), add time stats
