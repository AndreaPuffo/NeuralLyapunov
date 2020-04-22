import sympy as sp
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


A = sp.Matrix([[1, 0], [0, 2]])

xL = sp.MatrixSymbol('xL', 1, 2)
x0 = sp.symbols('x0')
x1 = sp.symbols('x1', positive=True)
x = sp.symbols('x')
y = sp.symbols('y')


print A
print xL.as_explicit()

print sp.MatMul(xL, A, xL.T).as_explicit()


eq = (x1**2/49 - x1/49)*(x1**2 - x1) + (48*x0*x1/49 - 96*x0/49)*(x0*x1 - 2*x0)
eq1 = x1 - 1
eq2 = x0*x1 > 0

print sp.solve([eq], (x0, x1), dict = True)

def fun(x0, x1):
    return (x1**2/49 - x1/49)*(x1**2 - x1) + (48*x0*x1/49 - 96*x0/49)*(x0*x1 - 2*x0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 10
xs = [i for i in range(n) for _ in range(n)]
ys = list(range(n)) * n
zs = [fun(x, y) for x,y in zip(xs,ys)]

ax.plot_surfaces(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()