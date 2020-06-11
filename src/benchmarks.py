import sympy as sp
import re


###############################
### NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions forPolynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap
def nonpoly0(v):
    x, y = v
    return [
        -x + x*y,
        -y
    ]

def nonpoly1(v):
    x, y = v
    return  [
            -x + 2*x**2 * y,
            -y
            ]

def nonpoly2(v):
        #  EXAMPLE A: A two prey one predator model
    x, y, z = v
    return [
            x * (1 - z),
            y * (1 - 2*z),
            z * (-1 + x + y)
        ]


def nonpoly3(v):
    x, y, z = v
        # EXAMPLE B.The model of May-Leonard: a 3-species competitive Lotka-Volterra system
    a, b = 1, 2
    return [
            x * (1-x - a*y - b*z),
            y * (1 - b*x - y - a*z),
            z * (1 - a*x - b*y - z)
        ]


######################
### TACAS benchmarks
######################

def benchmark_0(v):
    # test function, not to be included
    x, y = v
    return [
        -x**2 + 1,
        -y
    ]

def benchmark_1(v):
    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf

    x, y, z = v
    return [
        -x**3 - x*z**2,
        -y - x**2 * y,
        -z + 3*x**2*z - (3*z)
    ]


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns

def benchmark_3(v):
    x,y = v
    return  [
            - x**3 + y,
            - x - y
            ]

def benchmark_4(v):
    x, y = v
    return [
        -x**3 - y**2,
        x*y - y**3
    ]

def benchmark_5(v):
    x, y = v
    return [
        -x - 1.5 * x**2 * y**3,
        -y**3 + 0.5 * x**3 * y**2
    ]

def benchmark_6(v):
    x, y, w, z = v
    return [
        -x + y**3 - 3*w*z,
        -x - y**3,
        x*z - w,
        x*w - z**3
    ]

def benchmark_7(v):
    x0, x1, x2, x3, x4, x5 = v
    return [
        - x0 ** 3 + 4 * x1 ** 3 - 6 * x2 * x3,
        -x0 - x1 + x4 ** 3,
        x0 * x3 - x2 + x3 * x5,
        x0 * x2 + x2 * x5 - x3 ** 3,
        - 2 * x1 ** 3 - x4 + x5,
        -3 * x2 * x3 - x4 ** 3 - x5
    ]

def benchmark_8(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]

def benchmark_9(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]

def benchmark_10(x):
    # stupid test model, NO benchmark
    return [
        -x[0] + x[1]**100,
        -x[0] - x[1]
    ]

def max_degree_fx(fx):
    return max(max_degree_poly(f) for f in fx)

def max_degree_poly(p):
    s = str(p)
    s = re.sub(r'x\d+', 'x', s)
    try:
        f = sp.sympify(s)
        return sp.degree(f)
    except:
        print("Exception in %s for %s" % (max_degree_poly.__name__, p))
        return 0
