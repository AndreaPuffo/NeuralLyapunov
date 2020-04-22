import sympy as sp
from src import linear as linear


def Jacobian(M):
    """

    :param M: SymPy matrix
    :return: Jacobian with respect to all symbols in M (ordered by their name)
    """
    sym = sorted(list(M.atoms(sp.Symbol)), key=str)

    # keeps only x as variables
    for s in sym:
        if str(s).startswith('x'):
            pass
        else:
            sym.remove(s)

    rows = M.rows
    J = sp.zeros(rows, len(sym))
    for r in range(rows):
        row = M.row(r).jacobian(sym)
        J.row_op(r, lambda _, c: row[0, c])
    return J


def eval_matrix(M, e=0):

    sym = sorted( list( M.atoms(sp.Symbol) ), key=str )
    # keeps only variables starting w 'x'
    for s in sym:
        if str(s).startswith('x'):
            pass
        else:
            sym.remove(s)

    return M.subs({x: e for x in sym})


def _get_transpose(x):
    if isinstance(x, (list, tuple)):
        xT = x[1]
        x = x[0]
    else:
        xT = sp.Transpose(x)
    return x, xT


def f0(*args, **kwargs):
    return linear.f0(*args, **kwargs)


def f1(x, dx, P):
    x, xT = _get_transpose(x)
    dx, dxT = _get_transpose(dx)
    return sp.expand((dx * P * xT + x * P * dxT)[0])
