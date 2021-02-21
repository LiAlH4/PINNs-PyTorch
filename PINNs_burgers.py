import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.io
from scipy import optimize
from tqdm import tqdm

from obj import PyTorchObjective

torch.cuda.is_available()
cuda0 = torch.device("cuda:0")

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class My_NNs(nn.Module):
    def __init__(self, nu=0.01 / np.pi):

        super().__init__()
        self.nu = nu
        self.layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.layerOp = nn.ModuleList()

        for j in range(len(self.layers) - 2):
            self.layerOp.append(nn.Linear(self.layers[j], self.layers[j + 1]))
            self.layerOp.append(nn.Tanh())
        self.layerOp.append(nn.Linear(self.layers[-2], self.layers[-1]))

    def u_net(self, xiti):

        X = (xiti - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        # normalized to [-1, 1]
        for cur_op in self.layerOp:
            X = cur_op(X)

        return X

    def f_net(self, X):

        x = X[:, 0:1]
        t = X[:, 1:2]

        x.requires_grad_(True)
        t.requires_grad_(True)

        xt = torch.cat((x, t), 1)

        u = self.u_net(xt)

        (u_t,) = torch.autograd.grad(u.sum(), t, create_graph=True)
        (u_x,) = torch.autograd.grad(u.sum(), x, create_graph=True)
        (u_xx,) = torch.autograd.grad(u_x.sum(), x, create_graph=True)

        return u_t + u * u_x - (self.nu) * u_xx

    def get_data(self, X_u, X_f, X_u_sol):
        self.X_u = X_u
        self.X_f = X_f
        self.X_u_sol = X_u_sol

    def forward(self, X_u=None):

        if X_u is None:
            X_u_pred = self.u_net(self.X_u)
            X_f_pred = self.f_net(self.X_f)
            loss = F.mse_loss(X_u_pred, self.X_u_sol)
            loss += torch.norm(X_f_pred)
            return loss

        return self.u_net(X_u)


if __name__ == "__main__":

    N_u = 100
    N_f = 1000

    data = scipy.io.loadmat("burgers_shock.mat")
    data_x = data["x"]  # len = 256
    data_t = data["t"]  # len = 100
    u_solution = data["usol"]  # size = (256, 100)

    data_x_mesh, data_t_mesh = np.meshgrid(data_x, data_t, indexing="ij")

    data_x_mesh = torch.from_numpy(data_x_mesh).unsqueeze(2).float()
    data_t_mesh = torch.from_numpy(data_t_mesh).unsqueeze(2).float()

    data_mesh = torch.cat((data_x_mesh, data_t_mesh), 2)
    u_mesh_solution = u_solution.T

    u_solution = torch.from_numpy(u_solution).float()
    u_mesh_solution = torch.from_numpy(u_mesh_solution)

    lower_bound = torch.Tensor([[-1.0, 0.0]]).float()
    upper_bound = torch.Tensor([[1.0, 1.0]]).float()

    X_initial_coordinate = (
        torch.stack((data_x_mesh[:, 0:1], data_t_mesh[:, 0:1])).squeeze().T
    )
    # [[-1, 0], [-.99, 0], ..., [1, 0]]
    u_initial_coordinate_solution = u_solution[:, 0:1]

    X_lower_bound_coordinate = (
        torch.stack((data_x_mesh[0:1, :].T, data_t_mesh[0:1, :].T)).squeeze().T
    )
    # [[-1, 0], [-1, .01], ..., [-1, 1]]
    u_lower_bound_coordinate_solution = u_solution[0:1, :].T

    X_upper_bound_coordinate = (
        torch.stack((data_x_mesh[-1:, :].T, data_t_mesh[-1:, :].T)).squeeze().T
    )
    # [[-1, 0], [-1, .01], ..., [-1, 1]]
    u_upper_bound_coordinate_solution = u_solution[-1:, :].T

    # X_u_train = (X_initial_coordinate).float()
    # X_u_solution = (u_initial_coordinate_solution).float()

    X_u_train = torch.cat(
        (X_initial_coordinate, X_lower_bound_coordinate, X_upper_bound_coordinate), 0
    ).float()
    X_u_solution = torch.cat(
        (
            u_initial_coordinate_solution,
            u_lower_bound_coordinate_solution,
            u_upper_bound_coordinate_solution,
        ),
        0,
    ).float()

    X_f_train = lower_bound + (upper_bound - lower_bound) * torch.rand(N_f, 2)

    X_f_train = torch.cat((X_f_train, X_u_train), 0)

    model = My_NNs()
    model.get_data(X_u_train, X_f_train, X_u_solution)

    # print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    maxiter = 40
    with tqdm(total=maxiter) as pbar:

        def verbose(xk):
            pbar.update(1)

        obj = PyTorchObjective(model)

        xL = optimize.minimize(
            obj.fun,
            obj.x0,
            method="L-BFGS-B",
            # jac=obj.jac,
            callback=verbose,
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps,
            },
        )

    print("Training Finished")

    u_verf = model(data_mesh)
    u_verf.squeeze_()

    print("L2 Error = ", torch.norm(u_verf - u_solution).item())

    print("Evaluation Finished")

