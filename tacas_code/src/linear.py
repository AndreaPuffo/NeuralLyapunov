import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def f0(x, xT, P):
    """
    :param x: matrix
    :param xT: transpose of x
    :param P: matrix
    :return: x^T P x
    """
    return sp.expand((xT * P * x)[0])


def f1(x, xT, P, A):
    """
    :param x: matrix
    :param xT: transpose of x
    :param P: matrix
    :param A: matrix
    :return: x^T (A^T P + P A) x
    """
    AT = sp.Transpose(A)
    e0 = sp.Mul(AT, P, evaluate=False)
    e1 = sp.Mul(P, A, evaluate=False)
    e2 = sp.Add(e0, e1, evaluate=False)
    e = sp.Mul(xT, e2, evaluate=False)
    e = sp.Mul(e, x, evaluate=False)
    m = {
        str(P[r, c]): P[r, c] for r in range(P.shape[0]) for c in range(P.shape[0])
    }
    m.update({
        str(x): x for x in x
    })
    e = parse_expr(str(e), local_dict=m)
    # res = sp.expand((xT * (AT * P + P * A) * x)[0])
    return sp.expand(e[0])
