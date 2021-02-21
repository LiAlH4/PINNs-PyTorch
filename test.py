import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from scipy import optimize

from tqdm import tqdm
from obj import PyTorchObjective
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # whatever this initialises to is our "true" W
    linear = nn.Linear(32, 32)
    linear = linear.eval()

    # input X
    N = 10000
    X = torch.Tensor(N, 32)
    X.uniform_(0.0, 1.0)  # fill with uniform
    eps = torch.Tensor(N, 32)
    eps.normal_(0.0, 1e-4)

    # output Y
    with torch.no_grad():
        Y = linear(X) + eps

    # make module executing the experiment
    class Objective(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)
            self.linear = self.linear.train()
            self.X, self.Y = X, Y

        def forward(self):
            output = self.linear(self.X)
            return F.mse_loss(output, self.Y).mean()

    objective = Objective()

    maxiter = 40
    with tqdm(total=maxiter) as pbar:

        def verbose(xk):
            pbar.update(1)

        # try to optimize that function with scipy
        obj = PyTorchObjective(objective)
        xL = optimize.minimize(
            obj.fun,
            obj.x0,
            method="BFGS",
            jac=obj.jac,
            callback=verbose,
            options={"gtol": 1e-6, "disp": True, "maxiter": maxiter},
        )
        # xL = optimize.minimize(obj.fun, obj.x0, method='CG', jac=obj.jac)# , options={'gtol': 1e-2})

