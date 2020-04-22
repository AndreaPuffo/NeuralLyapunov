from sympy import Matrix
from src.nonlinear import *


def linearize(x):
    """

    :param x: Matrix or list
    :return: Jacobian of x
    """
    m = Matrix(x)
    J = Jacobian(m)
    print('Jacobian {}'.format(J))
    A = eval_matrix(J, e=0)
    return A
