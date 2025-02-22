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

    return x 

class ProjectNet(nn.Module):
    def __init__(self, A, b, dim, step = 3, rounds = 3, projection_count = 100, projection_tolerance = 0.01, factor = 1, xrho = 1e-2, offset = None):
        super(ProjectNet, self).__init__()

        self.projection_tolerance = projection_tolerance
        self.rounds = rounds
        self.p_count = projection_count
        self.factor = factor
            
            
        self.A = A
        self.AA = A.T @ torch.inverse(A @ A.T)
        self.AA = self.AA.to(device)
        self.b = b.to(device)
        
        self.dim = dim
        self.device = device
        
        self.L = nn.Parameter(torch.randn(dim, dim))
        self.Lambda = nn.Parameter(torch.ones(dim) + torch.randn(dim) * 0.1)
        self.rho = step 
        self.xrho = xrho
        
    def save_map(self): 
        self.Q = self.L @ torch.diag(self.Lambda) @ torch.inverse(self.L)
        self.Q = self.Q.detach()

    def learned_step(self, x, c, xrho, rho):
        return xrho * (self.Q @ x.T).T + rho * c

    def step(self, x, c, xrho, rho): 
        return xrho * (self.L @ torch.diag(self.Lambda) @ torch.inverse(self.L) @ x.T).T + rho * c

    def forward(self, c, rounds = None, tol = None, xrho = 0.5, factor = None, fixed = False): 
        if rounds is None: rounds = self.rounds
        if tol is None: tol = self.projection_tolerance
        if factor is None: factor = self.factor
            
        x = torch.zeros_like(c, device = self.device) 

        rho = self.rho

        for i in range(rounds):
            if fixed: 
                x = x - self.learned_step(x, c, xrho, rho)
            else:
                x = x - self.step(x, c, xrho, rho)

            x = project_pol(None, self.A, self.b, x, self.device, AA = self.AA, n = self.p_count, tol = tol, offset = None)
            rho = rho * factor
            xrho = xrho * factor
        return x
