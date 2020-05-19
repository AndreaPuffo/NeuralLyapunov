import numpy as np
import sympy as sp
from z3 import *
import torch


def activation(x):
    """
    :param x: tensor, one dimensional
    :return: tensor,
    """
    h = int(x.shape[0]/2)
    x1, x2 = x[:h], x[h:]
    return torch.cat([x1, torch.pow(x2, 2)]) # torch.pow(x, 2)
    # return torch.pow(x, 2)
    # return x*torch.relu(x)


def activation_z3(x):
    h = int(x.shape[0] / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack([x1, np.power(x2, 2)])  # torch.pow(x, 2)
    # return np.power(x, 2)
    # original_x = x
    # for idx in range(len(x)):
    #     x[idx, 0] = relu_z3(x[idx, 0])
    # return np.multiply(original_x, x)


def activation_der(x):
    h = int(x.shape[0] / 2)
    x1, x2 = x[:h], x[h:]
    return torch.cat((torch.ones(x1.shape).double(), 2*x2))
    # return 2 * x
    # return 2*torch.relu(x)


def activation_der_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((np.ones((h, 1)), 2*x2)) # NOTA: the first h elements DO NOT have variables in them
    # return 2 * x
    # return 2*np.maximum(x,0)



def relu_z3(x):
    return If(x > 0, x, 0)


def extract_val_from_z3(model, vars, useSympy):
    """
    :param model: a z3 model
    :param vars: set of vars the model is composed of
    :return: a numpy matrix containing values of the model
    """
    values = []
    for var in vars:
        val = model[var]
        if useSympy:
            values += [to_numpy(val)]
        else:
            values += [RealVal(val)]

    if useSympy:
        return np.matrix(values).T
    else:
        return values


def to_rational(x):
    """
    :param x: a string or numerical representation of a number
    :return: sympy's rational representation
    """
    return sp.Rational(x)


def to_numpy(x):
    """
       :param x: a Z3 numerical representation of a number
       :return: numpy's rational representation
       """
    x = str(x).replace('?', '0')
    return np.float(sp.Rational(x))


def get_symbolic_formula(net, x, xdot, equilibrium=None):
    """
    :param net:
    :param x:
    :param xdot:
    :return:
    """
    rounding = 3
    z, jacobian = network_until_last_layer(net, x, rounding)

    if equilibrium is None:
        equilibrium = np.zeros((net.n_inp, 1))

    projected_last_layer = weights_projection(net, equilibrium, rounding, z)
    z = np.dot(projected_last_layer, z)
    # this now contains the gradient \nabla V
    jacobian = np.dot(projected_last_layer, jacobian)

    Vdot = np.dot(jacobian, xdot)
    assert z.shape == (1, 1) and Vdot.shape == (1, 1)
    V = z[0, 0]
    Vdot = Vdot[0, 0]
    # val_in_zero, _ = z3_replacements(z3.simplify(V), V, x, equilibrium)
    # assert z3.simplify(val_in_zero) == 0
    return V, Vdot


def network_until_last_layer(net, x, rounding):
    """
    :param net:
    :param x:
    :param equilibrium:
    :return:
    """
    z = x
    jacobian = np.eye(net.n_inp, net.n_inp)

    for layer in net.layers[:-1]:
        w = np.round(layer.weight.data.numpy(), rounding)
        if layer.bias is not None:
            b = np.round(layer.bias.data.numpy(), rounding)[:, None]
        else:
            b = 0
        zhat = np.dot(w, z) + b
        z = activation_z3(zhat)
        # Vdot
        jacobian = np.dot(w, jacobian)
        jacobian = np.dot(np.diagflat(activation_der_z3(zhat)), jacobian)

    return z, jacobian


def weights_projection(net, equilibrium, rounding, z):
    """
    :param net:
    :param equilibrium:
    :return:
    """
    tol = 10 ** (-rounding)
    # constraints matrix
    c_mat = network_until_last_layer(net, equilibrium, rounding)[0]
    c_mat = sp.Matrix(sp.nsimplify(sp.Matrix(c_mat), rational=True).T)
    # compute projection matrix
    if c_mat == sp.zeros(c_mat.shape[0], c_mat.shape[1]):
        projection_mat = sp.eye(net.layers[-1].weight.shape[1])
    else:
        projection_mat = sp.eye(net.layers[-1].weight.shape[1]) \
                         - c_mat.T * (c_mat @ c_mat.T)**(-1) @ c_mat
    # make the projection w/o gradient operations with torch.no_grad
    last_layer = np.round(net.layers[-1].weight.data.numpy(), rounding)
    last_layer = sp.nsimplify(sp.Matrix(last_layer), rational=True, tolerance=tol)
    new_last_layer = sp.Matrix(last_layer @ projection_mat)

    return new_last_layer


def sympy_replacements(expr, xs, S):
    """
    :param expr: sympy expr
    :param xs: sympy vars
    :param S: list of tensors, batch numerical values
    :return: sum of expr.subs for all elements of S
    """
    total = []
    for idx in range(len(S)):
        numerical_val = S[idx].data.numpy()
        replacements = []
        for i in range(len(xs)):
            replacements += [(xs[i, 0], numerical_val[i])]
        total += [expr.subs(replacements)]
    return torch.tensor(total, dtype=torch.double, requires_grad=True)


def z3_replacements(V, Vdot, z3_vars, ctx):
    """
    :param V: z3 expr
    :param Vdot: z3 expr
    :param z3_vars: z3 vars
    :param ctx: list of numerical values
    :return: value of V, Vdot in ctx
    """
    replacements = []
    for i in range(len(z3_vars)):
        replacements += [(z3_vars[i, 0], z3.RealVal(ctx[i, 0]))]
    V_replace = z3.substitute(V, replacements)
    Vdot_replace = z3.substitute(Vdot, replacements)

    return V_replace, Vdot_replace


def print_section(word, k):
    print("=" * 80)
    print(' ', word, ' ', k)
    print("=" * 80)


def compute_equilibria(fx):
    """
    :param fx: list of sympy equations
    :return: list of equilibrium points
    """
    sol = sp.solve(fx)
    return sol


# removes imaginary solutions
def check_real_solutions(sols, x):
    """
    :param sols: list of dictories
    :param x: list of variables
    :return: list of dict w real solutions
    """
    good_sols = []
    for sol in sols:
        is_good_sol = True
        for index in range(len(sol)):
            if sp.im(sol[x[index]]) != 0:
                is_good_sol = False
                break
        if is_good_sol:
            good_sols.append(sol)
    return good_sols


def compute_distance(point, equilibrium):
    """
    :param point: np.array
    :param equilibrium: np.array
    :return: int = squared distance, r^2
    """
    return np.sum(np.power(point - equilibrium, 2))


def compute_bounds(n_vars, f, equilibrium):
    """
    :param n_vars: int, number of variables
    :param f: function
    :param equilibrium: np array
    :return: int, minimum distance from equilibrium to solution points of f
    """
    x0 = equilibrium
    # real=True should consider only real sols
    x_sp = [sp.Symbol('x%d' % i, real=True) for i in range(n_vars)]
    sols = compute_equilibria(f(x_sp))
    # sols = check_real_solutions(sols, x_sp) # removes imaginary solutions
    min_dist = np.inf
    for index in range(len(sols)):
        point = np.array(list(sols[index].values()))  # extract values from dict
        if not (point == x0).all():
            dist = compute_distance(point, x0)
            if dist < min_dist:
                min_dist = dist
    return min_dist


# computes the gradient of V, Vdot in point
# computes a 20-step trajectory (20 is arbitrary) starting from point
# towards increase: + gamma*grad
# towards decrease: - gamma*grad
def compute_trajectory(net, point, f):
    """
    :param net: NN object
    :param point: tensor
    :return: list of tensors
    """
    # set some parameters
    gamma = 0.01  # step-size factor
    max_iters = 20
    # fixing possible dimensionality issues
    trajectory = [point]
    num_vdot_value_old = 0
    # gradient computation
    for gradient_loop in range(max_iters):
        # compute gradient of Vdot
        gradient, num_vdot_value = compute_Vdot_grad(net, point, f)
        # set break conditions
        if abs(num_vdot_value_old - num_vdot_value) < 1e-3 or num_vdot_value > 1e6 or (gradient > 1e6).any():
            break
        else:
            num_vdot_value_old = num_vdot_value
        # "detach" and "requires_grad" make the new point "forget" about previous operations
        point = point.clone().detach() + gamma * gradient.clone().detach()
        point.requires_grad = True
        trajectory.append(point)
    # just checking if gradient is numerically unstable
    assert not torch.isnan(torch.stack(trajectory)).any()
    return trajectory


def compute_V_grad(net, point):
    """
    :param net:
    :param point:
    :return:
    """
    num_v = forward_V(net, point)[0]
    num_v.backward()
    grad_v = point.grad
    return grad_v, num_v


def compute_Vdot_grad(net, point, f):
    """
    :param net:
    :param point:
    :return:
    """
    num_v_dot = forward_Vdot(net, point, f)
    num_v_dot.backward()
    grad_v_dot = point.grad
    assert grad_v_dot is not None
    return grad_v_dot, num_v_dot


def forward_V(net, x):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            V: tensor, evaluation of x in net
    """
    y = x.double()
    for layer in net.layers[:-1]:
        z = layer(y)
        y = activation(z)
    y = torch.matmul(y, net.layers[-1].weight.T)
    return y


def forward_Vdot(net, x, f):
    """
    :param x: tensor of data points
    :param xdot: tensor of data points
    :return:
            Vdot: tensor, evaluation of x in derivative net
    """
    y = x.double()
    xdot = torch.stack(f(x))
    jacobian = torch.diag_embed(torch.ones(x.shape[0], net.n_inp)).double()

    for layer in net.layers[:-1]:
        z = layer(y)
        y = activation(z)
        jacobian = torch.matmul(layer.weight, jacobian)
        jacobian = torch.matmul(torch.diag_embed(activation_der(z)), jacobian)

    jacobian = torch.matmul(net.layers[-1].weight, jacobian)

    return torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1).double()[0]
