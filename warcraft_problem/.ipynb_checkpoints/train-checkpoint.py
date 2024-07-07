
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
import time

from constants  import *
from utils import * 



def train_projectnet(model, costs, epochs = 100, rounds = 1, lr = 0.01, verbose = False):    
    optimizer = optim.Adam(model.parameters(), lr=lr)
#     optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.95, weight_decay = 1)

    losses = []
    batch_size = 50
    for epoch in range(epochs): 
        permutation = torch.randperm(costs.shape[0])
        for i in range(0, costs.shape[0], batch_size):
            c = costs[i:i+batch_size,:]
            # opt = opt_train[i:i+batch_size,:]
            
            optimizer.zero_grad()

            outputs = model(c)
            # print("opt", opt)

            # opt_loss = torch.sum(opt * c, dim=1)/4
            loss = torch.mean(torch.sum(outputs * c, dim = 1))
            # print("opt", opt_loss)
            # print("pred", pred_loss)
            # loss = torch.mean((pred_loss - opt_loss)/opt_loss) 
            
            cur_loss = loss.item()
            loss.backward()

            optimizer.step()       

            losses.append(cur_loss)
    
            if verbose and i//batch_size % 20 == 0:
                print("epoch ", epoch, " batch ", i, " mean loss: ", np.mean(losses[-100:]), " median ", np.median(losses), "cur: ", cur_loss)

    print("MEAN LOSS PROJECT NET: ", np.mean(losses[-100:]))    
    return model, losses


def train_net(model, train_inputs, train_outputs, batch_size = 10, epochs = 10, lr = 0.001, verbose = False):    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mses = []
    test_losses = []
    train_losses = []
    batch_size = 50
    for epoch in range(epochs):  # loop over the dataset multiple times

        permutation = torch.randperm(train_inputs.shape[0])
        for i in range(0, train_inputs.shape[0], batch_size):
            inp = train_inputs[i:i+batch_size,:,:,:]
            out = train_outputs[i:i+batch_size]

            optimizer.zero_grad()
            
            pred = model(inp)
            loss = criterion(pred, out)
            loss.backward()

            optimizer.step()       
            mses.append(loss.item())
        print('epoch', epoch, 'loss:', np.mean(mses[-100:]))
                
    print("MEAN MSE: ", np.mean(mses))
    return model, mses, train_losses, test_losses


def edge_forecast(p, used_indices):
    return p.repeat_interleave(8, dim = 1)[:,used_indices]

def train_with_pnet(model, projectnet, train_inputs, test_inputs, train_outputs, test_outputs, used_indices, edge, rounds=1, map_size=12, batch_size = 100, epochs = 10, lr = 0.001, verbose = False):    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mses = []
    test_losses = [] 
    train_losses = [] 
    times = []
    best_regret = 100
    for epoch in range(epochs):  # loop over the dataset multiple times

        permutation = torch.randperm(train_inputs.shape[0])
        
        start = time.time()
        for i in range(0, train_inputs.shape[0], batch_size):
            inp = train_inputs[i:i+batch_size,:,:,:]
            out = train_outputs[i:i+batch_size]

            optimizer.zero_grad()
            
            pred = model(inp)
            pred = projectnet(edge_forecast(pred, used_indices), rounds = rounds, tol=0.05)
            pred = get_sol(pred, edge, map_size).flatten(start_dim=1)
            
            loss = torch.mean(torch.sum(out * pred, dim = 1)) 
#             loss = criterion(pred, out)
            loss.backward()

            optimizer.step()       
            mses.append(loss.item())
            if i % 1000 == 0:
                print("epoch ", epoch, "batch:", i//batch_size, " mean loss: ", np.mean(mses[-100:]), " median ", np.median(mses[-100:]))
        end = time.time()
        length = end - start
        times.append(length)
        
        optimal_paths_true = get_paths(model, test_inputs, map_size)
        ccc1 = torch.mean(torch.sum(torch.tensor(test_outputs).flatten(start_dim = 1) * optimal_paths_true, dim = 1))
        # test_true_dist = torch.sum(test_outputs.cpu().flatten(start_dim = 1) * test_labels_batch.flatten(start_dim = 1), dim = 1)
        test_losses.append(ccc1.item())

#         optimal_paths_true = get_paths(model, train_inputs, map_size)
#         ccc2 = torch.mean(torch.sum(torch.tensor(train_outputs).flatten(start_dim = 1) * optimal_paths_true, dim = 1))
#         train_losses.append(ccc2.item())
        
        print("epoch: ", epoch)
        print("time:", np.mean(times))
        # print("train: ", ccc2.item())
        print("test: ", ccc1.item())
        print() 

        if ccc1.item() < best_regret: 
            best_regret = ccc1.item() 
            torch.save(model.state_dict(), 'best-model-parameters.pt')
        
    print("MEAN MSE: ", np.mean(mses))
    return model, mses, train_losses, test_losses


from comb_modules.losses import HammingLoss
from comb_modules.dijkstra import ShortestPath


def train_with_blackbox(model, train_inputs, test_inputs, train_outputs, test_outputs, map_size=12, batch_size = 10, epochs = 100, lr = 0.000012, verbose = False):    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    hamming_loss = HammingLoss()

    mses = []
    batch_size = 20
    best_regret = 100
    times = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        permutation = torch.randperm(train_inputs.shape[0])
        start = time.time()
        for i in range(0, train_inputs.shape[0], batch_size):
            inp = train_inputs[i:i+batch_size,:,:,:]
            out = train_outputs[i:i+batch_size]

            optimizer.zero_grad()
            
            suggested_weights = model(inp).reshape(-1, 12,12)
            suggested_shortest_paths = ShortestPath.apply(suggested_weights, 20) # Set the lambda hyperparameter
#             loss = hamming_loss(suggested_shortest_paths, out.reshape(-1, 12,12)) # Use e.g. Hamming distance as the loss function
            
#             print()
            loss = torch.mean(torch.sum(out * suggested_shortest_paths.flatten(start_dim=1), dim = 1))

            loss.backward() # The backward pass is handled automatically
            optimizer.step()       
            mses.append(loss.item())
        end = time.time()
        length = end - start
        times.append(length)

        optimal_paths_true = get_paths(model, test_inputs, map_size)
        ccc1 = torch.mean(torch.sum(torch.tensor(test_outputs).flatten(start_dim = 1) * optimal_paths_true, dim = 1))
#         test_losses.append(ccc1.item())

        # optimal_paths_true = get_paths(model, train_images, map_size)
        # ccc2 = torch.mean(torch.sum(torch.tensor(train_weights).flatten(start_dim = 1) * optimal_paths_true, dim = 1))
#         train_losses.append(ccc2.item())
        
        print("epoch: ", epoch)
        # print("train: ", ccc2.item())
        print("test: ", ccc1.item())
        print("time:", np.mean(times))
        print() 
        
        if ccc1.item() < best_regret: 
            best_regret = ccc1.item() 
            torch.save(model.state_dict(), 'blackbox-parameters.pt')
            
        print("epoch ", epoch, " mean loss: ", np.mean(mses[-100:]), " median ", np.median(mses[-100:]))
    print("MEAN MSE: ", np.mean(mses))
    return model, mses