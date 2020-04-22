from __future__ import division
from z3 import *
import sympy as sp

TARGET_Z3 = 1
TARGET_SOLVER = 2
TARGET_DREAL = 3

# adapted from https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together


def _sympy_converter(var_map, exp, target, expand_pow=False):
    rv = None
    assert isinstance(exp, sp.Expr) and target is not None

    if isinstance(exp, sp.Symbol):
        rv = var_map.get(exp.name, None)
    elif isinstance(exp, sp.Number):
        try:
            rv = RealVal(exp) if target == TARGET_Z3 else sp.RealNumber(exp)
        except:  # Z3 parser error
            rep = sp.Float(exp, len(str(exp)))
            rv = RealVal(rep)
    elif isinstance(exp, sp.Add):
        # Add(exp_0, ...)
        rv = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)  # eval this expression
        for e in exp.args[1:]:  # add it to all other remaining expressions
            rv += _sympy_converter(var_map, e, target, expand_pow=expand_pow)
    elif isinstance(exp, sp.Mul):
        rv = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)
        for e in exp.args[1:]:
            rv *= _sympy_converter(var_map, e, target, expand_pow=expand_pow)
    elif isinstance(exp, sp.Pow):
        x = _sympy_converter(var_map, exp.args[0], target, expand_pow=expand_pow)
        e = _sympy_converter(var_map, exp.args[1], target, expand_pow=expand_pow)
        if expand_pow:
            try:
                i = float(e.sexpr())
                assert i.is_integer()
                i = int(i) - 1
                rv = x
                for _ in range(i):
                    rv *= x
            except:  # fallback
                _sympy_converter(var_map, exp, target, expand_pow=False)
        else:
            rv = x ** e

    assert rv is not None
    return rv


def sympy_converter(exp, target=TARGET_Z3, var_map={}):
    sympy_vars = exp.atoms(sp.Symbol)

    vars = []

    for var in sympy_vars:
        name = var.name
        if target == TARGET_Z3:
            var = var_map.get(name, None)
            if var is None:
                var = Real(name)
                var_map[name] = var
        else:
            var = var_map[name]
        vars += [var]

    return vars, var_map, _sympy_converter(var_map, exp, target)
