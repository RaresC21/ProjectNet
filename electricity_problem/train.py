


from solver import *

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

import time

def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 0.5 * torch.square(Y_sched - Y_actual)).mean(0)


def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 0.5 * torch.square(Y_sched - Y_actual))


def train_projectnet(model, Y_actual, params, epochs = 100, rounds = 1, lr = 0.01, verbose = False):    
    optimizer = optim.Adam(model.parameters(), lr=lr)
#     optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.95, weight_decay = 1)

    losses = []
    batch_size = 50
    for epoch in range(epochs): 
        for i in range(0, Y_actual.shape[0], batch_size):
            y = Y_actual[i:i+batch_size,:]
            
            optimizer.zero_grad()
            outputs = model(y)

            loss = task_loss(outputs, y, params).mean()
            
            cur_loss = loss.item()
            loss.backward()

            optimizer.step()       

            losses.append(cur_loss)
    
            # if verbose and i//batch_size % 20 == 0:
        print("epoch ", epoch, " mean loss: ", np.mean(losses[-100:]), " median ", np.median(losses), "cur: ", cur_loss)

    print("MEAN LOSS PROJECT NET: ", np.mean(losses[-100:]))    
    return model, losses

def train_with_pnet(model, projectnet, train_x, train_y, X_test, Y_test, params, rounds=5, map_size=12, batch_size = 100, epochs = 10, lr = 0.001, verbose = False):    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    times = []
    best_regret = 100
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = [] 
        
        start = time.time()
        for i in range(0, train_x.shape[0], batch_size):
            x = train_x[i:i+batch_size,:]
            y = train_y[i:i+batch_size,:]

            optimizer.zero_grad()
            
            pred = model(x)
            pred = projectnet(pred)
            
            loss = task_loss(pred, y, params).mean()
            loss.backward()

            optimizer.step()       
            train_losses.append(loss.item())
            
        end = time.time()
        length = end - start
        times.append(length)
                
        print("epoch ", epoch, " mean loss: ", np.mean(train_losses), "time", np.mean(times))
        
        test_costs = []
        for k in range(X_test.shape[0] // batch_size):
            p = model(X_test[k:k+batch_size,:]).detach()
            d = projectnet(p, rounds=5)
            cost_test = task_loss(d, Y_test[k:k+batch_size,:], params).mean()
            test_costs.append(cost_test.item())
        # p = model(train_x).detach()
        # d = projectnet(p, rounds=5)
        # cost = task_loss(d, train_y, params).mean()

        print("test:", np.mean(test_costs))

            
    # print("MEAN MSE: ", np.mean(mses))
    return model


def train_task_net(model, params, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = SolveScheduling(params)
    # solver = SolvePointQP(params)

    n_data = X_train.shape[0]
    batch = 25

    losses = []
    times = []

    for epoch in range(50):
        start = time.time()
        for i in range(0, n_data, n_data//batch):
            X = X_train[i:i+batch,:]
            Y = Y_train[i:i+batch,:]
            
            opt.zero_grad()
            mu_pred, sig_pred = model(X)
            y_sched = solver(mu_pred.double(), sig_pred.double())
            loss = task_loss(y_sched, Y, params).mean()

            loss.backward()
            opt.step()
        
            losses.append(loss.item())
        
        end = time.time()
        length = end - start
        times.append(length)
        
        print("epoch:", epoch, "loss:", np.mean(losses[-100:]), "times:", np.mean(times))
    
    return model
    

def train_mle_net(model, params, X_train, Y_train):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    n_data = X_train.shape[0]
    batch = 25

    losses = []
    times = []

    for epoch in range(50):
        start = time.time()
        for i in range(0, n_data, n_data//batch):
            X = X_train[i:i+batch,:]
            Y = Y_train[i:i+batch,:]
            
            opt.zero_grad()
            mu_pred = model(X)
            loss = criterion(mu_pred, Y)
            
            loss.backward()
            opt.step()
        
            losses.append(loss.item())
        
        end = time.time()
        length = end - start
        times.append(length)
        
        print("epoch:", epoch, "loss:", np.mean(losses[-100:]), "times:", np.mean(times))
    
    return model
    
    