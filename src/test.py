from src.cegis import Cegis
import torch
import timeit


def benchmark_3(v):
    x,y = v
    return  [
            - x**3 + y,
            - x - y
            ]

# ########################################


torch.manual_seed(0)

n_vars = 2
f = benchmark_3
n_hidden_neurons = [10]
outer_radius = 10
inner_radius = 0
margin = 0

start = timeit.default_timer()
c = Cegis(n_vars, f, inner_radius, outer_radius, margin, n_hidden_neurons)
c.solve()
stop = timeit.default_timer()
print('Elapsed Time: {}'.format(stop-start))

# todo: add main(), add time stats