# pylint: disable=no-member
# definition of various activation fcns
import numpy as np
import sympy as sp
import z3
import logging
try:
    import dreal as dr
except Exception as e:
    logging.exception('Exception while importing dReal')
from src.activations import ActivationType


def activation_z3(select, p):
    if select == ActivationType.IDENTITY:
        return p
    elif select == ActivationType.RELU:
        return relu_sp(p)
    elif select == ActivationType.LINEAR:
        return p
    elif select == ActivationType.SQUARE:
        return square_z3(p)
    elif select == ActivationType.LIN_SQUARE:
        return lin_square_z3(p)
    elif select == ActivationType.RELU_SQUARE:
        return relu_square_z3(p)
    elif select == ActivationType.REQU:
        return requ_z3(p)
    elif select == ActivationType.TANH:
        return hyper_tan_dr(p)
    elif select == ActivationType.SIGMOID:
        return sigm_dr(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC:
        return lqc_z3(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC_QUARTIC:
        return lqcq_z3(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC_QUARTIC_PENTA:
        return lqcqp_z3(p)
    elif select == ActivationType.LIN_ETC_EXA:
        return l_e_z3(p)
    elif select == ActivationType.LIN_OTT:
        return l_o_z3(p)


def activation_der_z3(select, p):
    if select == ActivationType.IDENTITY:
        return np.ones((p.shape))
    elif select == ActivationType.RELU:
        return step_z3(p)
    elif select == ActivationType.LINEAR:
        return np.ones((p.shape))
    elif select == ActivationType.SQUARE:
        return 2*p
    elif select == ActivationType.LIN_SQUARE:
        return lin_square_der_z3(p)
    elif select == ActivationType.RELU_SQUARE:
        return relu_square_der_z3(p)
    elif select == ActivationType.REQU:
        return requ_der_z3(p)
    elif select == ActivationType.TANH:
        return hyper_tan_der_dr(p)
    elif select == ActivationType.SIGMOID:
        return sigm_der_dr(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC:
        return lqc_der_z3(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC_QUARTIC:
        return lqcq_der_z3(p)
    elif select == ActivationType.LIN_SQUARE_CUBIC_QUARTIC_PENTA:
        return lqcqp_der_z3(p)
    elif select == ActivationType.LIN_ETC_EXA:
        return l_e_der_z3(p)
    elif select == ActivationType.LIN_OTT:
        return l_o_der_z3(p)


def relu_sp(x):
    y = x.copy()
    for idx in range(len(y)):
        y[idx, 0] = sp.Max(y[idx, 0], 0.0)
        # y[idx, 0] = z3.If(y[idx, 0] > 0, y[idx, 0], 0.0)
    return y


def square_z3(x):
    return np.power(x, 2)
    # assert(len(p[0]) == 1)
    # return [[elem[0] ** 2] for elem in p]


def lin_square_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((x1, np.power(x2, 2)))


def relu_square_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((relu_z3(x1), np.power(x2, 2)))


def requ_z3(x):
    return np.multiply(x, relu_z3(x))


def hyper_tan_dr(x):
    y = x.copy()
    # original_shape = y.shape
    # y = y.reshape(max(y.shape[0], y.shape[1]), 1)
    for idx in range(len(y)):
        y[idx, 0] = dr.tanh(y[idx, 0])
    return y # .reshape(original_shape)


def sigm_dr(x):
    # sigmoid is f(x) = 1/(1+e^-x)
    y = x.copy()
    for idx in range(len(y)):
        y[idx, 0] = 1/(1+dr.exp(-y[idx, 0]))
    return y


def lqc_z3(x):
    # linear - quadratic - cubic activation
    h = int(x.shape[0] / 3)
    x1, x2, x3 = x[:h, :], x[h:2 * h, :], x[2 * h:, :]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3)])


def lqcq_z3(x):
    # # linear - quadratic - cubic - quartic activation
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3 * h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4)])


def lqcqp_z3(x):
    # # linear - quadratic - cubic - quartic -penta activation
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3 * h:4*h], x[4*h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4), np.power(x5, 5)])


def l_e_z3(x):
    # # linear - quadratic - cubic - quartic -penta activation
    h = int(x.shape[0] / 6)
    x1, x2, x3, x4, x5, x6 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3 * h:4*h], x[4*h:5*h], x[5*h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4),
                      np.power(x5, 5), np.power(x6,6 )])


def l_o_z3(x):
    # # linear - quadratic - cubic - quartic -penta activation
    h = int(x.shape[0] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3 * h:4*h], \
                                     x[4*h:5*h], x[5*h:6*h], x[6*h:7*h], x[7*h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4),
                      np.power(x5, 5), np.power(x6, 6), np.power(x7, 7), np.power(x8, 8)])


##############################
# DERIVATIVE
##############################


def step_z3(x):
    y = x.copy()
    original_shape = y.shape
    y = y.reshape(max(y.shape[0], y.shape[1]), 1)
    for idx in range(y.shape[0]):
        y[idx, 0] = sp.Heaviside(y[idx, 0])
        # y[idx, 0] = z3.If(y[idx, 0] > 0.0, 1.0, 0.0)  # using 0.0 and 1.0 avoids int/float issues
    return y # .reshape(original_shape)


def lin_square_der_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((np.ones(x1.shape), 2*x2))


def relu_square_der_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((step_z3(x1), 2*x2))


def requ_der_z3(x):
    return 2*relu_z3(x)


def hyper_tan_der_dr(x):
    return np.ones((x.shape)) - np.power(x, 2)


def sigm_der_dr(x):
    # elem-wise multiplication
    return np.multiply(x, (np.ones(x.shape) - x))


def lqc_der_z3(x):
    # linear - quadratic - cubic activation
    h = int(x.shape[0] / 3)
    x1, x2, x3 = x[:h, :], x[h:2 * h, :], x[2 * h:, :]
    return np.vstack([np.ones((h, 1)), 2 * x2, 3 * np.power(x3, 2)])


def lqcq_der_z3(x):
    # # linear - quadratic - cubic - quartic activation
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3 * h:]
    return np.vstack([np.ones((h, 1)), 2*x2, 3*np.power(x3, 2), 4*np.power(x4, 3)])  # torch.pow(x, 2)


def lqcqp_der_z3(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3*h:4*h], x[4*h:]
    return np.vstack([np.ones((h, 1)), 2*x2, 3*np.power(x3, 2), 4*np.power(x4, 3), 5*np.power(x5, 4)])


def l_e_der_z3(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 6)
    x1, x2, x3, x4, x5, x6 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3*h:4*h], x[4*h:5*h], x[5*h:]
    return np.vstack([np.ones((h, 1)), 2*x2, 3*np.power(x3, 2), 4*np.power(x4, 3),
                      5*np.power(x5, 4), 6*np.power(x6, 5)])


def l_o_der_z3(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = x[:h], x[h:2 * h], x[2 * h:3 * h], x[3*h:4*h], \
                                     x[4*h:5*h], x[5*h:6*h], x[6*h:7*h], x[7*h:]
    return np.vstack([np.ones((h, 1)), 2*x2, 3*np.power(x3, 2), 4*np.power(x4, 3),
                      5*np.power(x5, 4), 6*np.power(x6, 5), 7*np.power(x7, 6), 8*np.power(x8, 7)])
