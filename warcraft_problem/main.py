import numpy as np
import os
import tarfile
import urllib.request
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import scipy as sc

from projectnet import ProjectNet
from train import train_projectnet
import process_data 

from constants import *
from utils import *

def evaluate(model, train_costs_edge): 
    model.eval()
    my_paths = get_sol(model(train_costs_edge).detach().cpu().numpy(), edge, map_size)
    print(my_paths.shape)
    pred = torch.sum(torch.tensor(train_weights).flatten(start_dim = 1) * my_paths.flatten(start_dim=1), dim = 1)
    opt = torch.sum(torch.tensor(train_weights).flatten(start_dim = 1) * train_labels, dim = 1)

    return torch.mean((pred-opt)/opt)
    

if __name__ == "__main__":
    
    map_size = 12
    train_images, test_images, train_costs, test_costs, train_weights, test_weights, train_labels, test_labels, max_training = process_data.get_data(map_size)
    A, b, train_costs_edge, test_costs_edge, used_indices, var_cnt, edge = process_data.get_constraints(map_size, train_costs, test_costs)
        
    # opt_path = get_paths_(train_costs, map_size)
    # print(train_labels[0,:])
    
    opt_train = path_to_edge(train_labels, map_size)
    # print(opt_train.shape, opt_train[0,:])
    # exit()
    
    rounds = 3

    projectnet = ProjectNet(A, b, var_cnt, rounds = rounds, step = 0.1, projection_tolerance = 0.01, factor = 1).to(device)
    train_projectnet(projectnet, train_costs_edge, epochs = 1, rounds = rounds, lr = 1e-5, verbose = True)
    
    test_loss = evaluate(projectnet, test_costs_edge)
    print("test loss:", test_loss.item())