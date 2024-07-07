import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import scipy as sc

import torch 
import torch.nn.functional as F
from constants import *

def project(A, b, x, AA = None): 
    if AA is None: 
        AA = A.T @ torch.inverse(A @ A.T)
        print("NOO")
    # a1 = torch.sparse.mm(A, x.T)
    a1 = A @ x.T
    return (x.T - AA @ (a1.T - b).T).T

def project_pol(P, A, b, x, device, n=10, AA = None, tol = 0.01, offset = None): 

#     print(x.shape, A.shape, b.shape)
    p = torch.zeros_like(x,device=device)
    q = torch.zeros_like(x, device=device)

    cc = 0
    cnter = 0
    while True:
        cnter += 1
        y = project(A, b, x + p, AA = AA)
        p = x + p - y       
        x = F.relu(y + q)
        q = y + q - x

        err = torch.max(F.relu(-y))
        if err < tol: 
            return y

#         proj = A @ x.T
#         err = torch.max(torch.abs(proj.T - b))
#         print(err)
#         if err < tol: break
#     print(cnter)
    return x #project(A, b, x)

class ProjectNet(nn.Module):
    def __init__(self, A, b, dim, step = 3, rounds = 3, projection_count = 100, projection_tolerance = 0.01, factor = 1, xrho = 1e-2, offset = None):
        super(ProjectNet, self).__init__()

        self.projection_tolerance = projection_tolerance
        self.rounds = rounds
        self.p_count = projection_count
        self.factor = factor
            
            
        self.A = A#A.to_sparse().to(device)
        self.AA = A.T @ torch.inverse(A @ A.T)
        self.AA = self.AA.to(device)
        self.b = b.to(device)
        
        self.dim = dim
        self.device = device
        
        self.linear = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(2*dim, 1000)
        self.linear2 = nn.Linear(1000, dim)

        self.L = nn.Parameter(torch.randn(dim, dim))
        self.Lambda = nn.Parameter(torch.ones(dim) + torch.randn(dim) * 0.1)
        self.rho = step #nn.Parameter(torch.ones(1))
        self.xrho = xrho
        
    def save_map(self): 
        self.Q = self.L @ torch.diag(self.Lambda) @ torch.inverse(self.L)
        self.Q = self.Q.detach()

    def learned_step(self, x, c, xrho, rho):
        return xrho * (self.Q @ x.T).T + rho * c

    def step(self, x, c, xrho, rho): 
        # l = F.relu(self.linear1(torch.cat((x,c),dim=1)))
        # l = F.relu(self.linear2(l)) 
        # return xrho * l + rho * c
        # return xrho * self.linear(x) + rho * c
        # return xrho * (self.L @ x.T).T + rho * c
        return xrho * (self.L @ torch.diag(self.Lambda) @ torch.inverse(self.L) @ x.T).T + rho * c
        # return xrho * x + rho * c

    def forward(self, c, rounds = None, tol = None, xrho = 0.5, factor = None, fixed = False): 
        if rounds is None: rounds = self.rounds
        if tol is None: tol = self.projection_tolerance
        if factor is None: factor = self.factor
            
        x = torch.zeros_like(c, device = self.device) # torch.clone(c)# torch.zeros(c.shape[0], self.dim)

#         print(RHS.shape)
        rho = self.rho
        # xrho = self.xrho

        for i in range(rounds):
            if fixed: 
                x = x - self.learned_step(x, c, xrho, rho)
            else:
                x = x - self.step(x, c, xrho, rho)

            x = project_pol(None, self.A, self.b, x, self.device, AA = self.AA, n = self.p_count, tol = tol, offset = None)
            rho = rho * factor
            xrho = xrho * factor
        return x

#     def get_path(self, c, rounds = None):
#         if rounds is None: rounds = self.rounds        
#         x = torch.rand_like(c) # torch.clone(c)# torch.zeros(c.shape[0], self.dim)
#         rho = self.rho
#         all_ = [x]
#         for i in range(rounds):
#             aug = torch.cat((x, c), dim = 1)

#             x = x + rho * self.iteration_(aug)
#             x = project_pol(None, self.A, self.b, x, n = self.p_count, tol = self.projection_tolerance, offset = None)
#             all_.append(x.clone())
#             rho = rho * self.factor
#         return all_

    
# def train_projectnet(model, costs, epochs = 100, targets = None, lr = 0.01, verbose = False):    
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
# #     optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.95, weight_decay = 1)

#     losses = []
#     mses = []
#     batch_size = 10
#     for epoch in range(epochs):  # loop over the dataset multiple times

#         permutation = torch.randperm(costs.shape[0])
#         # for i in range(n):
#         for i in range(0, costs.shape[0], batch_size):
#             c = torch.tensor(costs[i:i+batch_size,:]).float()
#             if not targets is None:
#                 target = torch.tensor(targets[i:i+batch_size]).float()

#             optimizer.zero_grad()

#             outputs, all_ = model(c)
            
# #             print(outputs)
#             loss = torch.mean(torch.sum(outputs * c, dim = 1)) 
# #             loss = criterion(outputs, target)
#             cur_loss = loss.item()
#             loss.backward()

#             optimizer.step()       
                                   
#             # print statistics
# #             cur_loss = loss.item()
# #             print(cur_loss)
#             losses.append(cur_loss)

#             if verbose and i % 10 == 0:
#                 print("epoch ", epoch, " batch ", i, " mean loss: ", np.mean(losses[-100:]), " median ", np.median(losses))
# #                 print("MSE: ", np.mean(mses[-100:]))
#     print("MEAN REGRET PROJECT NET: ", np.mean(losses[-100:]))
#     return model, losses, mses