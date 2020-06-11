import numpy as np
import matplotlib.pyplot as plt
from src.plots.plots import barrier_3d, plot_square_sets, plot_parabola, vector_field
from matplotlib.patches import Rectangle


if __name__ == '__main__':
    plot_limit = 10
    X = np.linspace(-plot_limit, plot_limit, 100)
    Y = np.linspace(-plot_limit, plot_limit, 100)
    x0, x1 = np.meshgrid(X, Y)
    V = (x0**2 + x1**2)*(0.318*np.maximum(0.0, -557*x0/1000 - 9*x1/100 - 179/200) + 0.528*np.maximum(0.0, -251*x0/500 - 37*x1/125 - 92/125) + 0.388*np.maximum(0.0, -27*x0/250 + 219*x1/250 + 12/25) + 0.41*np.maximum(0.0, 609*x0/1000 - 109*x1/125 + 193/250) + 0.664*np.maximum(0.0, 619*x0/1000 + 259*x1/250 + 187/100))
    Vdot = (-x0 - x1)*(2*x1*(0.318*np.maximum(0.0, -557*x0/1000 - 9*x1/100 - 179/200) + 0.528*np.maximum(0.0, -251*x0/500 - 37*x1/125 - 92/125) + 0.388*np.maximum(0.0, -27*x0/250 + 219*x1/250 + 12/25) + 0.41*np.maximum(0.0, 609*x0/1000 - 109*x1/125 + 193/250) + 0.664*np.maximum(0.0, 619*x0/1000 + 259*x1/250 + 187/100)) + (x0**2 + x1**2)*(-0.02862*np.heaviside(-557*x0/1000 - 9*x1/100 - 179/200, 0) - 0.156288*np.heaviside(-251*x0/500 - 37*x1/125 - 92/125, 0) + 0.339888*np.heaviside(-27*x0/250 + 219*x1/250 + 12/25, 0) - 0.35752*np.heaviside(609*x0/1000 - 109*x1/125 + 193/250, 0) + 0.687904*np.heaviside(619*x0/1000 + 259*x1/250 + 187/100, 0))) + (-x0**3 + x1)*(2*x0*(0.318*np.maximum(0.0, -557*x0/1000 - 9*x1/100 - 179/200) + 0.528*np.maximum(0.0, -251*x0/500 - 37*x1/125 - 92/125) + 0.388*np.maximum(0.0, -27*x0/250 + 219*x1/250 + 12/25) + 0.41*np.maximum(0.0, 609*x0/1000 - 109*x1/125 + 193/250) + 0.664*np.maximum(0.0, 619*x0/1000 + 259*x1/250 + 187/100)) + (x0**2 + x1**2)*(-0.177126*np.heaviside(-557*x0/1000 - 9*x1/100 - 179/200, 0) - 0.265056*np.heaviside(-251*x0/500 - 37*x1/125 - 92/125, 0) - 0.041904*np.heaviside(-27*x0/250 + 219*x1/250 + 12/25, 0) + 0.24969*np.heaviside(609*x0/1000 - 109*x1/125 + 193/250, 0) + 0.411016*np.heaviside(619*x0/1000 + 259*x1/250 + 187/100, 0)))
    ax = barrier_3d(x0, x1, V)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('V')
    plt.title('Lyapunov fcn')

    ax = barrier_3d(x0, x1, Vdot)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('Vdot')
    plt.title('Lyapunov derivative')

    ################################
    # PLOT 2D
    ################################

    plt.figure()
    ax = plt.gca()

    def f(v):
        x, y = v
        dydt =[-x**3 + y, -x - y]
        return dydt


    # plot vector field
    xv = np.linspace(-plot_limit, plot_limit, 10)
    yv = np.linspace(-plot_limit, plot_limit, 10)
    Xv, Yv = np.meshgrid(xv, yv)
    t = np.linspace(0, 5, 100)
    vector_field(f, Xv, Yv, t)

    ax.contour(X, Y, V, 5, linewidths=2, colors='k')

    plt.title('Lyapunov Border')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
