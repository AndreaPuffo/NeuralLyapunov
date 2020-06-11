import torch
import torch.nn as nn
import numpy as np
from src.activations import ActivationType, activation, activation_der
from src.utils import Timer, timer

T = Timer()


class NN(nn.Module):
    def __init__(self, n_inp, *args, bias=False, activate=ActivationType.LIN_SQUARE, equilibria=0):
        super(NN, self).__init__()

        self.n_inp = n_inp
        n_prev = n_inp
        self.eq = equilibria
        self.acts = activate
        self._is_there_bias = bias
        self.layers = []
        k = 1
        for n_hid in args:
            layer = nn.Linear(n_prev, n_hid, bias=bias).double()
            self.register_parameter("W" + str(k), layer.weight)
            if (bias):
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # free output layer
        layer = nn.Linear(n_prev, 1, bias=False).double()
        self.register_parameter("W" + str(k), layer.weight)
        self.layers.append(layer)
        self.output_layer = torch.tensor(layer.weight)
        # or
        # self.output_layer = torch.ones(1, n_prev)

    # generalisation of forward with tensors
    def forward_tensors(self, x, xdot):
        """
        :param x: tensor of data points
        :param xdot: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                Vdot: tensor, evaluation of x in derivative net
                jacobian: tensor, evaluation of grad_net
        """
        y = x.double()
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.n_inp)).double()

        for idx, layer in enumerate(self.layers[:-1]):
            z = layer(y)
            y = activation(self.acts[idx], z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(torch.diag_embed(activation_der(self.acts[idx], z)), jacobian)

        numerical_v = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)
        numerical_vdot = torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1).double()

        return numerical_v[:, 0], numerical_vdot, jacobian[:, 0, :]

    def numerical_net(self, S, Sdot):
        """
        :param net: NN object
        :param S: tensor
        :param Sdot: tensor
        :return: V, Vdot, circle: tensors
        """
        assert (len(S) == len(Sdot))

        nn, grad_times_f, grad_nn = self.forward_tensors(S, Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)
        E, factors = 1, []
        for idx in range(self.eq.shape[0]):
            # S - self.eq == [ x-x_eq, y-y_eq ]
            # (vector_x - eq_0) = ( x-x_eq + y-y_eq )
            E *= torch.sum(S - torch.tensor(self.eq[idx, :]), dim=1)
            factors.append(torch.sum(S - torch.tensor(self.eq[idx, :]), dim=1))
        derivative_e = torch.stack([torch.sum(torch.stack(factors), dim=0), torch.sum(torch.stack(factors), dim=0)]).T

        # define E(x) := (x-eq_0) * ... * (x-eq_N)
        # V = NN(x) * E(x)
        V = nn * E
        # gradV = NN(x) * dE(x)/dx  + der(NN) * E(x)
        gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1).double()

        return V, Vdot, circle

    # backprop algo
    @timer(T)
    def learn(self, optimizer, S, S_dot):
        """
        :param optimizer: torch optimiser
        :param S: tensor of data
        :param S_dot: tensor contain f(data)
        :param margin: performance threshold
        :return: --
        """
        assert (len(S) == len(S_dot))

        batch_size = len(S)
        learn_loops = 1000
        margin = 0.01

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = self.numerical_net(S, S_dot)
            learn_accuracy = 0.5 * ( sum(Vdot <= -margin).item() + sum(V >= margin).item() )

            slope = 10 ** (self.orderOfMagnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(1 / slope)
            loss = (leaky_relu(Vdot + margin * circle)).mean() + (leaky_relu(-V + margin * circle)).mean()

            if t%100 == 0:
                print(t, "- loss:", loss.item(), "- acc:", learn_accuracy * 100 / batch_size, '%')

            loss.backward()
            optimizer.step()

            if learn_accuracy == batch_size:
                break

            # if self._is_there_bias:
            #     self.weights_projection()

    def weights_projection(self):
        # bias_vector = self.layers[0].bias.double()
        # constraints matrix
        _, _, c_mat = self.forward_tensors(self.equilibrium, self.equilibrium)
        # compute projection matrix
        if (c_mat == 0).all():
            projection_mat = torch.eye(self.layers[-1].weight.shape[1])
        else:
            projection_mat = torch.eye(self.layers[-1].weight.shape[1]).double() \
                                  - c_mat.T @ torch.inverse(c_mat @ c_mat.T) @ c_mat
        # make the projection w/o gradient operations with torch.no_grad
        with torch.no_grad():
            self.layers[-1].weight.data = self.layers[-1].weight @ projection_mat
            x0 = torch.zeros((1, self.n_inp)).double()
            # v0, _, _ = self.forward_tensors(x0, x0)
            # print('Zero in zero? V(0) = {}'.format(v0.data.item()))

    # todo: mv to utils
    def orderOfMagnitude(self, number):
        if number.item() != 0:
            return np.floor(np.log10(number))
        else:
            return 1.0

    @staticmethod
    def get_timer():
        return T


