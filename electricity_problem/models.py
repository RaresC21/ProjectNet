
import numpy as np 

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
import torch.optim as optim
from constants import *

from functools import reduce
import operator

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T).to(DEVICE)
        b.data = torch.Tensor(Theta[-1,:]).to(DEVICE)
                
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        # layers = reduce(operator.add, 
        #     [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
        #         for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.ReLU()] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.lin(x) + self.net(x)
    
class Net2(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net2, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))
        
    def forward(self, x):
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)
