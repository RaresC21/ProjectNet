
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

from constants import *

def get_data(size):

    data_dir = "/home/jovyan/Rares/projectnet/warcraft/data/{}x{}".format(size,size)

    train_prefix = "train"
    test_prefix = "test"

    n = 10000
    train_inputs = np.load(os.path.join(data_dir, train_prefix + "_maps" + ".npy"))[:n,:,:]
    train_weights = np.load(os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))[:n,:,:]
    train_labels = np.load(os.path.join(data_dir, train_prefix + "_shortest_paths.npy"))[:n,:,:]

    test_inputs = np.load(os.path.join(data_dir, test_prefix + "_maps" + ".npy"))[:n,:,:]
    test_weights = np.load(os.path.join(data_dir, test_prefix + "_vertex_weights.npy"))[:n,:,:]
    test_labels = np.load(os.path.join(data_dir, test_prefix + "_shortest_paths.npy"))[:n,:,:]

    #images = train_inputs.transpose(0,2,3,1).astype(np.uint8)
    images = train_inputs

    
    train_images = torch.tensor(train_inputs).permute(0, 3, 1, 2).float() / np.max(train_inputs) + 0.01
    test_images = torch.tensor(test_inputs).permute(0, 3, 1, 2).float() / np.max(train_inputs) + 0.01
    train_costs = torch.tensor(train_weights).flatten(start_dim = 1).float()
    test_costs = torch.tensor(test_weights).flatten(start_dim = 1).float()

    train_labels = torch.tensor(train_labels).flatten(start_dim = 1).float().to(device)
    test_labels = torch.tensor(test_labels).flatten(start_dim = 1).float().to(device)

    max_training = torch.max(train_costs) 
    train_costs /= max_training
    test_costs /= max_training
    test_weights /= max_training 
    train_weights /= max_training
    
    return train_images.to(device), test_images.to(device), train_costs, test_costs, train_weights, test_weights, train_labels, test_labels, max_training
    
def get_constraints(n_nodes, train_costs, test_costs): 
    cnt = 0 
    edge = {}
    
    train_costs_edge = []
    test_costs_edge = []
    used_indices = [] 

    total = n_nodes * n_nodes
    all_ = 0
    for i in range(n_nodes):
        for j in range(n_nodes): 
            k = 0
            for x,y in zip(dx, dy): 
                r = i + x 
                c = j + y 
                if r >= 0 and r < n_nodes and c >= 0 and c < n_nodes: 
                    edge[i,j,r,c] = cnt
                    cnt += 1
                    train_costs_edge.append(train_costs[:,i * n_nodes + j].numpy())
                    test_costs_edge.append(test_costs[:,i * n_nodes + j].numpy())
                    used_indices.append(all_)
                all_ += 1

    A = []
    b = []
    for i in range(n_nodes): 
        for j in range(n_nodes): 
            val = 0
            if i == 0 and j == 0: val = 1
            if i == n_nodes - 1 and j == n_nodes - 1: val = -1

            cur = [0] * cnt
            for x,y in zip(dx, dy): 
                r = i + x 
                c = j + y 
                if r >= 0 and r < n_nodes and c >= 0 and c < n_nodes: 
                    cur[edge[i,j,r,c]] = 1
                    cur[edge[r,c,i,j]] = -1
            A.append(cur)
            b.append(val)


    A = torch.tensor(A).float()[:-1,:]
    b = torch.tensor(b).float()[:-1]

    train_costs_edge = np.array(train_costs_edge)
    train_costs_edge = torch.tensor(train_costs_edge).T

    test_costs_edge = np.array(test_costs_edge)
    test_costs_edge = torch.tensor(test_costs_edge).T

    used_indices = torch.tensor(used_indices)

    return A.to(device), b.to(device), train_costs_edge.to(device), test_costs_edge.to(device), used_indices.to(device).long(), cnt, edge
    
def show_tiles(im, weights): 
    fig, ax = plt.subplots(1,2, figsize=(8,8))

    ax[0].imshow(im.astype(np.uint8))
    ax[0].set_title("Map")

    ax[1].set_title("True Vertex weights")
    ax[1].imshow(weights.astype(np.float32))

def show_solution(im, label, true_labels, weights):
    fig, ax = plt.subplots(1,4, figsize=(16,8))

    ax[0].imshow(im.astype(np.uint8))
    ax[0].set_title("Map")

    ax[2].imshow(label)
    ax[2].set_title("Proposed ProjectNet Path")

    ax[1].set_title("True Vertex weights")
    ax[1].imshow(weights.astype(np.float32))

    ax[3].imshow(true_labels)
    ax[3].set_title("True Shortest Path")
