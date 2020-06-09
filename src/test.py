import torch
import timeit
import numpy as np
from src.cegis import Cegis
from src.utils import compute_bounds
from src.benchmarks import benchmark_0, benchmark_3
from src.activations import ActivationType
from src.consts import VerifierType, LearnerType


torch.manual_seed(0)

n_vars = 2
f = benchmark_3
activations = [ActivationType.SQUARE]
n_hidden_neurons = [10] * len(activations)

outer_radius = 10
inner_radius = 0
margin = 0

learner_type = LearnerType.NN
verifier_type = VerifierType.Z3

start = timeit.default_timer()
c = Cegis(n_vars, f, learner_type, verifier_type, inner_radius, outer_radius, margin, n_hidden_neurons, activations)
c.solve()
stop = timeit.default_timer()
print('Elapsed Time: {}'.format(stop-start))

equilibrium = np.zeros((1, n_vars))
min_bound = compute_bounds(n_vars, f, equilibrium)
print('The validity bound for this system is: {}'.format(min_bound))

# todo: add main(), add time stats
