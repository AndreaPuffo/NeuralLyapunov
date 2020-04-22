from __future__ import division

import itertools
import random
import time
import timeit

import z3
from z3 import *
import sympy as sp
import numpy as np
import re
from itertools import izip  # Python2: less memory and may be faster (iterator, no intermediate list)

from src.consts import A_MAX_DIGITS
from src.log import log


def OrNotZero_s(xs):
    """
    :param xs: list or generator
    :return: Z3 'xs[0] != 0 or ... or xs[n] != 0 with n = len(xs)'; 'True' if n == 0
    """
    if isinstance(xs, list) and len(xs) < 1:
        return 'True'
    if isinstance(xs, list) and len(xs) == 1:
        return '%s != 0' % xs[0]
    return 'Or(%s)' % ','.join('%s != 0' % x for x in xs)


def OrNotZero(xs):
    """
    :param xs: list or generator
    :return: Z3 xs[0] != 0 or ... or xs[n] != 0 with n = len(xs); True if n == 0
    """
    if isinstance(xs, list) and len(xs) < 1:
        return True
    if isinstance(xs, list) and len(xs) == 1:
        return xs[0] != 0
    r = Or(*(x != 0 for x in xs))
    assert r.num_args() > 0
    return r


def AllZero(xs):
    """
    :param xs: list or generator
    :return: Z3 xs[0] == 0 and ... and xs[n] == 0 with n = len(xs); True if n == 0
    """
    if isinstance(xs, list) and len(xs) < 1:
        return True
    r = And(*(x == 0 for x in xs))
    assert r.num_args() > 0
    return r


def AllNotZero(xs):
    """
    :param xs: list or generator
    :return: Z3 xs[0] != 0 and ... and xs[n] != 0 with n = len(xs); True if no items exist
    """
    r = And(*(x != 0 for x in xs))
    assert r.num_args() > 0
    return r


def AndNot(xs, ns):
    """
    :raises AssertionError iff len(xs) != len(ns)
    :param xs: list
    :param ns: list
    :return: Z3 xs[0] != ns[0] and ... and xs[n] != ns[n] with n = len(xs); True if no items exist
    """
    r = And(*(x != n for x, n in izip(xs, ns)))
    assert r.num_args() > 0
    return r


def OrNot(P, M):
    """
    :param P: list
    :param M: model indexed on P's items (M(P_i) = v)
    :return: OR_i { P_i != M(P_i) }
    """
    N = len(M)

    return Or(*(P[i] != M[P[i]] for i in range(N)))


def P_symbols(N, symmetric=False, name='P'):
    """
    :param name: matrix and elements name
    :param N: dimension
    :param symmetric: [optional] True iff the result will be a symmetric matrix; False otherwise. Defaults to False
    :return: SymPy matrix of SymPy Symbol objects
    """
    if symmetric:
        M = np.empty(shape=(N, N), dtype=object)
        for c in range(N):
            for r in range(c, N):
                idx = c * N + r
                M[r, c] = sp.Symbol(name + str(idx), real=True)
                M[c, r] = M[r, c]
        return sp.Matrix(M)
    else:
        return sp.Matrix(
            [[sp.Symbol(name + str(r * N + c), real=True) for c in range(N)] for r in range(N)])


def M_to_matrix(M, N, symmetric=False):
    """
    Z3 model [Pi = vi, ..] -> [[vi, ..], ..]
    :param symmetric: if True, the result is a symmetric matrix (from the lower triangle)
    :param M: model
    :param N: dimension
    :return: SymPy matrix
    """
    P = np.empty((N, N), dtype=object)
    end = N ** 2 if not symmetric else N * (N + 1) // 2
    i = 0
    while i < end and i < len(M):
        p = M[i]
        k = int(str(p)[1:])
        c = int(k / N)
        r = k % N
        if P[r, c] is None:
            P[r, c] = M[p]
            if symmetric and r != c:
                P[c, r] = P[r, c]
        else:
            end += 1
        i += 1
    return sp.Matrix(P).applyfunc(lambda x: 0 if not x else x)


def is_diagonal(M):
    """

    :param M: a SymPy/NumPy matrix
    :return: True iff M is diagonal
    """
    N = M.shape[0]
    return len(M.shape) == 2 and M.shape[1] == N and all(M[i, j] == 0 for i in range(N) for j in range(N) if i != j)


def diagonalize(M, real=True):
    """
    M = U D U^T
    :param M: a matrix-like object (list, NumPy/SymPy matrix)
    :param real: True iff D, U need to have only real elements
    :return: None iff M is not diagonalizable; (D, U) otherwise
    """
    M = sp.Matrix(M)
    if is_diagonal(M):
        return M, sp.eye(M.shape[0])
    try:
        # O = U * D * U.T
        D, U = M.diagonalize()[::-1]  # D, U
        if real and (len(D.atoms(sp.I)) > 0 or len(U.atoms(sp.I)) > 0):  # imaginary unit
            log("Could not diagonalize: %s. M: %s" % ("D, U have complex numbers", M))
            return None
        return D, U
    except Exception as e:
        log("Could not diagonalize: %s. M: %s" % (e, M))
        return None


def dict_to_matrix(D, N, symmetric=False):
    """
    {P0: v0, P1: v1, ..} -> [[v0, v1, ..], ..] \in Mat^(NxN)
    :param symmetric:
    :param D: dictionary
    :param N: dimension
    :return: SymPy matrix
    """
    if isinstance(D, dict):
        P_sym = P_symbols(N)
        P = sp.Matrix(P_sym)
        for _k, v in D.items():
            # eg _k, v = P0, 1
            try:
                k = int(str(_k)[1:])
            except ValueError:
                k = int(_k)
            r = int(k / N)
            c = k % N
            P[r, c] = v
        if symmetric:
            # loop on triangular sub-matrix
            for r in range(N):
                for c in range(r, N):
                    P[c, r] = P[r, c]
        return P
    return D


def check_with_timeout(solver, timeout, args=None):
    """
    Runs the solver's check() method for at most t sec and asserts if it timed out.
    t sec is the minimum to return it has timed out
    :param args: [optional] arguments to pass to check
    :param solver: Z3 Solver instance
    :param timeout: number
    :return: <solver's result, timedout>
    """
    if timeout < 0:
        return unsat, True
    else:
        timer = timeit.default_timer()
        res = solver.check(*args) if args is not None else solver.check()
        timer = timeit.default_timer() - timer
        timedout = timer >= timeout
        return res, timedout


def p2m(s):
    """
    power to multiplication
    assumption: all exponents are positive integers
    :param s: string
    :return: s where every use of the exponent is replaced by a multiplication
    """
    r = r"(\w+)\*\*(\d+)"
    for (expr, pow) in re.findall(r, s):
        s = s.replace(expr + '**' + pow, '*'.join(expr for _ in range(int(pow))))
    return s


def to_rational(x):
    """
    :param x: a string or numerical representation of a number
    :return: sympy's rational representation
    """
    return sp.Rational(x)


def is_valid_matrix_A(A):
    """
    :param A: matrix
    :return: True iff the parameter is a valid (system) A matrix
    """
    A = sp.Matrix(A)
    is_squared = A.shape[0] == A.shape[1]
    is_real_and_neg = all(sp.re(sp.simplify(sp.re(l))) <= 0 for l in A.eigenvals().keys())
    valid = A is not None and \
           hasattr(A, "shape") and hasattr(A.shape, "__len__") and \
           len(A.shape) == 2 and \
           is_squared and \
           is_real_and_neg
    if not valid:
        log("Invalid A: (squared, %s), (real and negative eig, %s)" % (is_squared, is_real_and_neg))
        print('A eigenvalues are: {}'.format(A.eigenvals().keys()))
    return valid

# find if A matrix has zero eigen
def is_limit_matrix_A(A):
    """
    :param A: matrix
    :return: True iff the parameter is a valid (system) A matrix
    """
    A = sp.Matrix(A)
    is_limit = any(sp.re(sp.simplify(sp.re(l))) == 0 for l in A.eigenvals().keys())
    print('A is Limit Stable? {}'.format(is_limit))
    limit = is_valid_matrix_A(A) and is_limit
    return limit



def model_to_vector(model, vars, N=-1, target_z3=False):
    """
    :param model: a Z3 model
    :param vars: a set of variables (eg. P0...)
    :param N: dimension
    :param target_z3: True iff Z3 will receive the vector
    :return: a SymPy Matrix if not target_z3 else model
    """
    if target_z3:
        return model
    else:
        d = {str(x)[1:]: model[x] for x in vars}
        if len(d) < N:
            for i in range(N):
                if str(i) not in d:
                    d[str(i)] = 0
        d2 = sorted(d.items(), key=lambda t: int(t[0]))

        # watch out for truncated numbers (indicated by a ? character at the end)
        values = map(lambda (k, v): to_rational(str(v)) if str(v)[-1] != '?' else to_rational(str(v)[:-1]), d2)
        return sp.Matrix(values)


def diagonal_names(N, prefix=''):
    """

    :param prefix: [optional] string to prepend to each number
    :param N: dimension
    :return: an ordered list of strings of indices to the diagonal of a matrix
    """
    return [prefix + str(n) for n in range(0, N ** 2, N + 1)]


def get_random_P(dim=3):
    """
    P = [[P0 P1 ..] .. [..]] full, non-symmetric
    :param dim: dimension of P
    :return: SymPy P with Z3 symbols
    """
    return sp.Matrix([[z3.Real('P%d' % (r * dim + c)) for r in range(dim)] for c in range(dim)])


def _grand(r, min_val, max_val, int=True):
    while True:
        n = r(min_val, max_val)
        yield n if int else round(n, A_MAX_DIGITS)


def get_random_A(dim=3, valid=True, diagonal=True, ints=True, max_val=1e+3):
    """

    :param dim: dimension of A
    :param valid: True iff A's eigenvalues are all negative
    :param diagonal: if True, A will be diagonal
    :param ints: A will have only integer elements
    :param max_val: max value of A if diagonal, else the square of the max value
    :return: A
    """
    max_val = int(max_val)
    sgn = -1 if valid else 1
    r = random.randint if ints else random.uniform
    diag = set()
    # generate n unique numbers
    for x in itertools.takewhile(lambda x: len(diag) < dim, _grand(r, 1, max_val, int=ints)):
        diag.add(sgn * x)
    D = sp.diag(*diag)
    if not diagonal:
        M = sp.Matrix([[sp.nan]])
        while any(x == sp.nan for x in M):  # avoid nan (due to X)
            X = sp.randMatrix(dim, dim, min=-max_val, max=max_val, seed=int(time.time() * 1000))
            # eigenvals from D: det(XDX^-1 -tI) = det(X(D-tI)X^-1) = det(D-tI)
            M = X * D * (X ** -1)
        return M
    else:
        return D  # diagonal matrix with -r values


def random_non_diagonalizable(dim=3, valid=True, max_val=int(1e+3)):
    """

    :param max_val:
    :param dim: dimension of the result matrix A
    :param valid: True iff the result is a valid A
    :return: random A that's non-diagonalizable
    """
    A = get_random_A(dim=dim, valid=valid, diagonal=True, max_val=max_val)
    if dim > 1:
        A[0, 0] = A[1, 1]
        A[0, 1] = 1
    return A


def read_matrix_from_file(path):
    """
    Reads a square matrix from a text file;
    each line represents a row
    each number in a row is separated by spaces
    :param path: file path
    :return: <SymPy matrix, dimension>
    """
    M = sp.Matrix([])
    N = -1
    with open(path) as f:
        r = 0
        for line in f:
            if N < 0:
                first_line = line.split()
                N = len(first_line)
                M = sp.Matrix(N, N, lambda i, j: 0)
                for c in range(N):
                    M[r, c] = first_line[c]
            else:
                r += 1
                l = line.split()
                for c in range(N):
                    M[r, c] = l[c]
    return M, N


def x_dot(x, n):
    """

    :param x: vector
    :param n: power
    :return: (x.)^n
    """
    return x.applyfunc(lambda i: i ** n)

# creates a poly from roots
def createPoly(roots):
    l = sp.symbols('l')
    #l=1
    poly=1
    for r in roots:
        poly = poly*(l-r)
    return sp.expand(poly)

# creates fx of the form
# y^(i-th) + c0 y^(i-1 - th) + .... = 0
def createFX(poly):
    coeff = poly.coeffs()
    s = []
    for i in range(len(coeff)-1):
        if i < len(coeff)-2:
            s.append( sp.symbols('x[{}]'.format(i+1)) )
        else:
            sf = 0
            for j in range(len(coeff)-1):
                sf += - sp.Integer(coeff[-j-1]) * sp.symbols( 'x[{}]'.format(j) )
            s.append(sf)
    return s


def computeDist(p):
    # computes distance of a point p from the origin
    """
    :param p: point in space
    :return: dist of p from origin
    """
    dist = np.sum( np.square( np.array(list(p) ) ) )
    #print(dist)
    return math.sqrt(dist)


def termsWith(expr, s, blacklist=[]):
    """
    Example: termsWith(x**2y+3x**2+y, x**2, blacklist=[y]) == 3x**2

    :param expr: an expression
    :param s: a symbol in the expression
    :param blacklist: an iterable of symbols in the expression to avoid
    :return: the terms in expression with `s` but no term from `blacklist`
    """
    return sp.Add(*[x for x in expr.args if x.has(s) and not any(x.has(b) for b in blacklist)])


def diagonaliseVd(Vd_list, xs):
        """
        :param Vd_list: list of Vd
        :param xs: generators of Vd
        :return: Vd_diag: SOS terms of Vd, Vd_offdiag: non-SOS terms of Vd
        """

        Vd_d = []
        Vd_od = []
        for idx in range(2*len(Vd_list)):
            # need to find x**(2**(e+1)) in v
            # v is the current Vd in the list
            # expand is needed to make this method work
            exp = 2 * (idx + 1)
            for j in range(len(Vd_list)):
                v = sp.expand(Vd_list[j][0])
                t = 0
                for l in range(len(xs)):
                    # y = termsWith(v[0], xs[l] ** exp, blacklist=[xi for xi in xs if xs[l] != xi])
                    # NOTA: expan( expr, deep=True) is necessary to make sympy_converter work properly
                    #t += sp.expand( v[0].collect(xs[l]).coeff(xs[l] ** exp) * xs[l] ** exp, deep=True )
                    bl = [ x for i,x in enumerate(xs) if i!=l ]
                    t += termsWith(v, xs[l]**exp, blacklist=bl )
                Vd_d.append(sp.Matrix([t]))

                Vd_od.append( sp.Matrix([v-Vd_d[j][0]]) )

        return Vd_d, Vd_od