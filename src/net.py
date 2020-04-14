import torch
import torch.nn as nn
import numpy as np


def activation(x):
    # h = int(len(x)/2)
    # x1, x2 = x[:h], x[h:]
    # return torch.cat([x1, torch.pow(x2, 2)]) # torch.pow(x, 2)
    return torch.pow(x, 2)
    # return x*torch.relu(x)


def activation_der(x):
    # h = int(len(x) / 2)
    # x1, x2 = x[:h], x[h:]
    # return torch.cat([torch.ones(1,h)[0], 2*x2])  # torch.ones(1,h)[0] because of dimension issues...
    return 2 * x
    # return 2*torch.relu(x)


class NN(nn.Module):
    def __init__(self, n_inp, *args, bias=False):
        super(NN, self).__init__()

        self.n_inp = n_inp
        n_prev = n_inp

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
        """
        y = x.double()
        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.n_inp)).double()

        for layer in self.layers[:-1]:
            z = layer(y)
            y = activation(z)

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(torch.diag_embed(activation_der(z)), jacobian)

        y = torch.matmul(y, self.layers[-1].weight.T)
        jacobian = torch.matmul(self.layers[-1].weight, jacobian)

        return y, torch.sum(torch.mul(jacobian[:, 0, :], xdot), dim=1).double()


    def numerical_net(self, S, Sdot):
        """
        :param net: NN object
        :param S: tensor
        :param Sdot: tensor
        :return: V, Vdot, circle: tensors
        """
        assert (len(S) == len(Sdot))

        V, Vdot = self.forward_tensors(S, Sdot)
        # circle = x0*x0 + ... + xN*xN
        circle = torch.pow(S, 2).sum(dim=1)

        return V, Vdot, circle

    # backprop algo
    def learn(self, optimizer, S, S_dot, margin):
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

        for t in range(learn_loops):
            optimizer.zero_grad()

            V, Vdot, circle = self.numerical_net(S, S_dot)
            learn_accuracy = 0.5 * ( sum(Vdot <= -margin).item() + sum(V >= margin).item() )

            slope = 10 ** (self.orderOfMagnitude(max(abs(Vdot)).detach()))
            leaky_relu = torch.nn.LeakyReLU(1 / slope)
            loss = (leaky_relu(Vdot + margin * circle)).mean() + (leaky_relu(-V + margin * circle)).mean()

            print(t, "- loss:", loss.item(), "- acc:", learn_accuracy * 100 / batch_size, '%')

            if learn_accuracy == batch_size:
                break

            if learn_accuracy / batch_size > 0.99:
                for k in range(batch_size):
                    if Vdot[k] > -margin:
                        print("Vdot" + str(S[k].tolist()) + " = " + str(Vdot[k].tolist()))

            loss.backward()
            optimizer.step()

    # todo: mv to utils
    def orderOfMagnitude(self, number):
        return np.floor(np.log10(number))

