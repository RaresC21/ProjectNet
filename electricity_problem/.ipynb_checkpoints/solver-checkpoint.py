#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim

from qpth.qp import QPFunction
from constants import *


def GLinearApprox(gamma_under, gamma_over):
    """ Linear (gradient) approximation of G function at z"""
    class GLinearApproxFn(Function):
        @staticmethod    
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.cdf(
                z.cpu().numpy()) - gamma_under)
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = (gamma_under + gamma_over) * pz
            dmu = -dz
            dsig = -(gamma_under + gamma_over)*(z-mu) / sig * pz
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GLinearApproxFn.apply


def GQuadraticApprox(gamma_under, gamma_over):
    """ Quadratic (gradient) approximation of G function at z"""
    class GQuadraticApproxFn(Function):
        @staticmethod
        def forward(ctx, z, mu, sig):
            ctx.save_for_backward(z, mu, sig)
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            res = torch.DoubleTensor((gamma_under + gamma_over) * p.pdf(
                z.cpu().numpy()))
            if USE_GPU:
                res = res.cuda()
            return res
        
        @staticmethod
        def backward(ctx, grad_output):
            z, mu, sig = ctx.saved_tensors
            p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
            pz = torch.tensor(p.pdf(z.cpu().numpy()), dtype=torch.double, device=DEVICE)
            
            dz = -(gamma_under + gamma_over) * (z-mu) / (sig**2) * pz
            dmu = -dz
            dsig = (gamma_under + gamma_over) * ((z-mu)**2 - sig**2) / \
                (sig**3) * pz
            
            return grad_output * dz, grad_output * dmu, grad_output * dsig

    return GQuadraticApproxFn.apply


class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.cuda()
        
    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) 
            for i in range(nBatch)], 0).double()
        p = (dg - d2g*z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out


class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """
    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = torch.tensor(np.vstack([D,-D]), dtype=torch.double, device=DEVICE)
        self.h = (self.c_ramp * torch.ones((self.n - 1) * 2, device=DEVICE)).double()
        self.e = torch.DoubleTensor()
        if USE_GPU:
            self.e = self.e.cuda()
        
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        
        # Find the solution via sequential quadratic programming, 
        # not preserving gradients
        z0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        mu0 = mu.detach() # Variable(1. * mu.data, requires_grad=False)
        sig0 = sig.detach() # Variable(1. * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0-z0_new).norm().item()
            # print("+ SQP Iter: {}, Solution diff = {}".format(i, solution_diff))
            z0 = z0_new
            if solution_diff < 1e-10:
                break
                  
        # Now that we found the solution, compute the gradient-propagating 
        # version at the solution
        dg = GLinearApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)
    
class SolvePointQP(nn.Module): 
    def __init__(self, params):
        super(SolvePointQP, self).__init__()

        self.DEVICE=DEVICE
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        self.n_vars = self.n * 3

        G = []
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + self.n] = -1
            G.append(cur)
        for i in range(self.n): 
            cur = [0] * self.n_vars 
            cur[i] = -1 
            cur[i + self.n * 2] = -1 
            G.append(cur)
            
        for i in range(self.n-1): 
            cur = [0] * self.n_vars 
            cur[i] = 1 
            cur[i + 1] = -1
            G.append(cur)

            cur = [0] * self.n_vars 
            cur[i] = -1
            cur[i + 1] = 1
            G.append(cur)

        for i in range(self.n_vars): 
            cur = [0] * self.n_vars 
            cur[i] = -1
            G.append(cur)

        self.G = torch.tensor(G, device=DEVICE).double() 
        self.e = torch.empty(0, device=DEVICE)

        self.Q = torch.eye(self.n_vars, device=DEVICE).double() * .5
        self.p = torch.tensor([0] * self.n + [params['gamma_over'] for _ in range(self.n)] + [params['gamma_under'] for _ in range(self.n)], device=DEVICE).double()

    def forward(self, pred):
        nBatch, n = pred.shape

        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))

        ramp_h = self.c_ramp * torch.ones((self.n - 1)*2, device=DEVICE).double()
        ramp_h = ramp_h.unsqueeze(0).expand(nBatch, ramp_h.size(0))

        h = torch.cat((pred, -pred, ramp_h, torch.zeros(nBatch, self.n_vars, device=DEVICE)), 1)

        return QPFunction(verbose=False)(self.Q, self.p, G, h, self.e, self.e)[:,:self.n]
