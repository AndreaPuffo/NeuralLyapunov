from __future__ import division
import sympy as sp


def gen_P(name, dim, symmetric=True):
    assert symmetric
    P = sp.Matrix([[0 for _ in range(dim)] for _ in range(dim)])
    for r in range(dim):
        for c in range(dim):
            P[r, c] = sp.Symbol("%s#%d#%d" % (name, r, c), real=True)
            P[c, r] = P[r, c]
    return P


def diagonal_names(n, prefix):
    return ["%s#%d#%d" % (prefix, i, i) for i in range(n)]


def Model2Matrix(model, Ps, symmetric=True, target=''):
    assert symmetric
    n = Ps[0].shape[0]
    res = [sp.zeros(n, n) for _ in range(len(Ps))]
    for x in model.decls():
        name = x.name()
        prefix, r, c = name.split('#')
        idx = int(prefix[1:]) - 1
        r = int(r)
        c = int(c)
        v = str(model[x])
        v = v if v[-1] != '?' else v[:-1]
        res[idx][r, c] = sp.Rational(v)  # TODO model[x] if target=='z3'
    return res


if __name__ == '__main__':
    print(gen_P('p0', 3))
