import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.io

print(torch.cuda.is_available())

np.random.seed(123)
torch.manual_seed(123)


class My_NNs(nn.Module):
    def __init__(self, lower_bound, upper_bound, nu=0.01 / np.pi):

        super().__init__()
        self.nu = nu
        self.layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        self.layerOp = nn.ModuleList()
        self.lb, self.ub = lower_bound, upper_bound

        for j in range(len(self.layers) - 2):
            self.layerOp.append(nn.Linear(self.layers[j], self.layers[j + 1]))
            self.layerOp.append(nn.Tanh())
        self.layerOp.append(nn.Linear(self.layers[-2], self.layers[-1]))

    def u_net(self, xi, ti):
        X = torch.cat((xi, ti), -1)
        X = (X - self.lb) / (self.ub - self.lb) * 2 - 1
        # normalized to [-1, 1]
        for cur_op in self.layerOp:
            X = cur_op(X)

        return X

    def f_net(self, xi, ti):

        xi.requires_grad_(True)
        ti.requires_grad_(True)

        u = self.u_net(xi, ti)

        (u_t,) = torch.autograd.grad(u.sum(), ti, create_graph=True)
        (u_x,) = torch.autograd.grad(u.sum(), xi, create_graph=True)
        (u_xx,) = torch.autograd.grad(u_x.sum(), xi, create_graph=True)

        return u_t + u * u_x - (self.nu) * u_xx

    def forward(self, X_u, X_f=None):

        X_u_pred = self.u_net(X_u[..., 0:1], X_u[..., 1:2])
        if X_f is not None:
            X_f_pred = self.f_net(X_f[..., 0:1], X_f[..., 1:2])
            return X_u_pred, X_f_pred
        return X_u_pred


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

    u_solution = torch.from_numpy(u_solution).float()

    lb = torch.Tensor([[-1.0, 0.0]])
    ub = torch.Tensor([[1.0, 1.0]])

    X_init_coord = data_mesh[:, 0, :]
    # [[-1, 0], [-.99, 0], ..., [1, 0]]
    u_init_coord_solution = u_solution[:, 0].view(-1, 1)

    X_lb_coord = data_mesh[0, :, :]
    # [[-1, 0], [-1, .01], ..., [-1, 1]]
    u_lb_coord_solution = u_solution[0, :].view(-1, 1)

    X_ub_coord = data_mesh[-1, :, :]
    # [[-1, 0], [-1, .01], ..., [-1, 1]]
    u_ub_coord_solution = u_solution[-1, :].view(-1, 1)

    X_u_train = torch.cat((X_init_coord, X_lb_coord, X_ub_coord), 0)
    X_u_solution = torch.cat(
        (u_init_coord_solution, u_lb_coord_solution, u_ub_coord_solution,), 0,
    )

    X_f_train = lb + (ub - lb) * torch.rand(N_f, 2)
    X_f_train = torch.cat((X_f_train, X_u_train), 0)

    X_f_solution = torch.zeros(X_f_train.shape)

    if torch.cuda.is_available():
        lb = lb.cuda()
        ub = ub.cuda()
        model = My_NNs(lb, ub)
        model.cuda()
        X_u_train = X_u_train.cuda()
        X_f_train = X_f_train.cuda()
        X_u_solution = X_u_solution.cuda()
        X_f_solution = X_f_solution.cuda()
        data_mesh = data_mesh.cuda()
        u_solution = u_solution.cuda()

    # print(model)

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(
        model.parameters(),
        history_size=50,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe",
    )

    for i in range(300):

        print("STEP: ", i, end="")

        def closure():
            optimizer.zero_grad()
            u_predict, f_predict = model(X_u_train, X_f_train)
            X1 = criterion(u_predict, X_u_solution)
            # X2 = criterion(f_predict, X_f_solution)
            X2 = torch.norm(f_predict) / 33
            Loss = X1 + X2
            print("Loss = %g, X1 = %g, X2 = %g" % (Loss.item(), X1.item(), X2.item()))
            Loss.backward()
            return Loss

        optimizer.step(closure)

    print("Training Finished")

    u_verf = model(data_mesh).squeeze()

    print(
        "L2 Error = ",
        torch.norm(u_verf - u_solution).item() / np.sqrt(torch.numel(u_solution)),
    )

    print("Evaluation Finished")

    u_output = np.savetxt("tmp.txt", u_verf.detach().cpu().numpy())

