import torch
import numpy as np
from z3 import *
import timeit
import sympy as sp

from src.utils import get_symbolic_formula, sympy_replacements
from src.net import *
from src.verifier import *

torch.manual_seed(0)

def benchmark_3(v):
    x,y = v
    return  [
            - x**3 + y,
            - x - y
            ]


# ########################################
n_vars = 2
f = benchmark_3
n_hidden_neurons = [10]
# stability params
outer_radius = 10
inner_radius = 0

# sets learner performance
margin = 0.01
# sets the threshold for the verifier V > margin_verif, Vdot < margin_verif
# useful to set it at a smaller value than margin (wrong if margin_verif > margin...)
margin_verification = 1e-3*inner_radius*margin
# batch init
batch_size = 1000
learning_rate = 0.1


########################################
# global config
########################################


x = [Real('x%d' % i) for i in range(n_vars)]
if n_vars > 1:
    xdot = np.matrix(f(x)).T
else:
    xdot = np.matrix(f(x[0])).T
x = np.matrix(x).T

print(xdot)

# model
network = NN(n_vars, *n_hidden_neurons, bias=False)
verifier = Z3Verifier(n_vars, inner_radius, outer_radius, margin, x)

S = []

for k in range(batch_size):
    s = torch.rand(n_vars).double() * 2*outer_radius - outer_radius
    S.append(s)

Sdot = list(map(torch.tensor, map(f, S)))
# tensorised version
S = torch.stack(S).type(torch.DoubleTensor)
Sdot = torch.stack(Sdot).type(torch.DoubleTensor)

optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

# the CEGIS loop
max_cegis_iters = 10
lyapunov = False
start = timeit.default_timer()

for k in range(max_cegis_iters):
    print("=" * 80)
    print(" Learning", k)
    print("=" * 80)

    network.learn(optimizer, S, Sdot, margin)

    print("=" * 80)
    print(' Candidate', k)
    print("=" * 80)

    V, Vdot = get_symbolic_formula(network, x, xdot)
    V, Vdot = z3.simplify(V), z3.simplify(Vdot)

    print(f'V: {V}')
    print(f'Vdot: {Vdot}')

    print("=" * 80)
    print(' Verification', k)
    print("=" * 80)

    lyapunov, C = verifier.verify(V, Vdot)

    if C == [] and lyapunov:
        break

    S = torch.cat([S, torch.stack(C).type(torch.DoubleTensor)], dim=0)
    Sdot = torch.cat([Sdot, torch.stack(list(map(torch.tensor, map(f, C)))).type(torch.DoubleTensor)], dim=0)

print('Elapsed Time: {}'.format(timeit.default_timer() - start))

if lyapunov:
    # print(f'V: {V}')
    # print(f'Vdot: {Vdot}')
    print("Hurray! That's Lyapunov, baby")
else:
    print("Sorry, didn't make it")
