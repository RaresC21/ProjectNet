import numpy as np 
import torch
from comb_modules.dijkstra import dijkstra
from constants import *


def get_paths(model, input_, n_nodes):
    pred = model(input_)
    optimal_paths = []
    return get_paths_(pred, n_nodes)
    
def get_paths_(all_weights, n_nodes):
    optimal_paths = []
    for weights in all_weights:
        res = dijkstra(weights.reshape(n_nodes,n_nodes).detach().cpu().numpy())[0]
        optimal_paths.append(res)
    optimal_paths = torch.tensor(optimal_paths).to(device)
    return optimal_paths.flatten(start_dim = 1)


def path_to_edge(path, n_nodes): 
    edge_vals = []
    for i in range(n_nodes):
        for j in range(n_nodes): 
            for x,y in zip(dx, dy): 
                r = i + x 
                c = j + y 
                if r >= 0 and r < n_nodes and c >= 0 and c < n_nodes: 
                    edge_vals.append(path[:,i * n_nodes + j].numpy())
    return torch.tensor(np.array(edge_vals)).T.to(device)


# convert edge solution to path 
def get_sol(sol, edge, n_nodes):
    total = 0
    label = torch.zeros((sol.shape[0], n_nodes, n_nodes)).to(device)
    for i in range(n_nodes): 
        for j in range(n_nodes): 
            for x,y in zip(dx, dy): 
                r = i + x 
                c = j + y 
                if r >= 0 and r < n_nodes and c >= 0 and c < n_nodes:
                    label[:,i,j] = label[:,i,j] + sol[:,edge[i,j,r,c]]
                    # print("inter2", label[:,i,j])
    label[:,n_nodes-1, n_nodes-1] = 1
    return label
