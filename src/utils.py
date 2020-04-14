import numpy as np
import sympy as sp
from z3 import *
import torch


def activation(x):
    # h = int(len(x) / 2)
    # x1, x2 = x[:h], x[h:]
    # return np.vstack((x1, np.power(x2, 2)))
    return np.power(x, 2)
    # return x*np.maximum(x,0)


def activation_z3(x):
    # h = int(len(x) / 2)
    # x1, x2 = x[:h], x[h:]
    # return np.vstack((x1, np.power(x2, 2)))
    return np.power(x, 2)
    # original_x = x
    # for idx in range(len(x)):
    #     x[idx, 0] = relu_z3(x[idx, 0])
    # return np.multiply(original_x, x)


def activation_der(x):
    # h = int(len(x) / 2)
    # x1, x2 = x[:h], x[h:]
    # return np.vstack((np.ones((h, 1)), 2*x2)) # NOTA: the first h elements DO NOT have variables in them
    return 2 * x
    # return 2*np.maximum(x,0)


def activation_der_z3(x):
    for idx in range(len(x)):
        x[idx, 0] = relu_z3(x[idx, 0])
    return 2 * x


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



def get_symbolic_formula(net, x, xdot):
    """
    :param net:
    :param x:
    :param xdot:
    :return:
    """
    z = x
    jacobian = np.eye(net.n_inp, net.n_inp)

    for layer in net.layers[:-1]:
        w = layer.weight.data.numpy()
        if layer.bias is not None:
            b = layer.bias.data.numpy()[:, None]
        else:
            b = 0
        zhat = np.dot(w, z) + b
        z = activation_z3(zhat)
        # Vdot
        jacobian = np.dot(w, jacobian)
        jacobian = np.dot(np.diagflat(activation_der(zhat)), jacobian)


    z = np.dot(net.layers[-1].weight.data.numpy(), z)
    jacobian = np.matmul(net.layers[-1].weight.data.numpy(),
                           jacobian)  # this now contains the gradient \nabla V

    Vdot = np.dot(jacobian, xdot)
    assert z.shape == (1, 1) and Vdot.shape == (1, 1)
    V = z[0, 0]
    Vdot = Vdot[0, 0]
    return V, Vdot



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
        replacements += [(z3_vars[i], z3.RealVal(ctx[i]))]
    V_replace = z3.substitute(V, replacements)
    Vdot_replace = z3.substitute(Vdot, replacements)

    return V_replace, Vdot_replace


def print_section(word, k):
    print("=" * 80)
    print(' ', word, ' ', k)
    print("=" * 80)