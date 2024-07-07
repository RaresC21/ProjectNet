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

def project_pol(P, A, b, x, DEVICE, n=10, AA = None, tol = 0.01, offset = None): 

#     print(x.shape, A.shape, b.shape)
    p = torch.zeros_like(x,device=DEVICE)
    q = torch.zeros_like(x, device=DEVICE)

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
    return x #project(A, b, x)

class ProjectNet(nn.Module):
    def __init__(self, A, b, dim, step = 3, rounds = 3, projection_count = 100, projection_tolerance = 0.01, factor = 1, xrho = 1e-2, offset = None):
        super(ProjectNet, self).__init__()

        self.projection_tolerance = projection_tolerance
        self.rounds = rounds
        self.p_count = projection_count
        self.factor = factor
            
            
        self.A = A#A.to_sparse().to(DEVICE)
        self.AA = A.T @ torch.inverse(A @ A.T)
        self.AA = self.AA.to(DEVICE)
        self.b = b.to(DEVICE)
        
        self.dim = dim
        
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

    def step(self, x, y, xrho, rho): 
        diff = x - y
        grad_step = 2 * diff 
        learn_step = self.linear(diff)
        return rho * diff + xrho * learn_step
        # l = F.relu(self.linear1(torch.cat((x,c),dim=1)))
        # l = F.relu(self.linear2(l)) 
        # return xrho * l + rho * c
        # return xrho * self.linear(x) + rho * c
        # return xrho * (self.L @ x.T).T + rho * c
        # return xrho * (self.L @ torch.diag(self.Lambda) @ torch.inverse(self.L) @ x.T).T + rho * c
        # return xrho * x + rho * c

    def forward(self, c, rounds = None, tol = None, xrho = 0.5, factor = None, fixed = False): 
        if rounds is None: rounds = self.rounds
        if tol is None: tol = self.projection_tolerance
        if factor is None: factor = self.factor
            
        x = torch.zeros_like(c, device = DEVICE) # torch.clone(c)# torch.zeros(c.shape[0], self.dim)

#         print(RHS.shape)
        rho = self.rho
        # xrho = self.xrho

        for i in range(rounds):
            if fixed: 
                x = x - self.learned_step(x, c, xrho, rho)
            else:
                x = x - self.step(x, c, xrho, rho)
            
            x = torch.cat((x, torch.zeros(x.shape[0], self.A.shape[1]-x.shape[1]).to(DEVICE)), 1).to(DEVICE).float()            
            x = project_pol(None, self.A, self.b, x, DEVICE, AA = self.AA, n = self.p_count, tol = tol, offset = None)[:,:24]
            rho = rho * factor
            xrho = xrho * factor
        return x
