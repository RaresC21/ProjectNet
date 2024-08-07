


from solver import *

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

import time

def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 0.5 * torch.square(Y_sched - Y_actual)).mean(0)
    # return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
    #         params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0)).mean(0)

def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 0.5 * torch.square(Y_sched - Y_actual))


def train_projectnet(model, Y_actual, params, Y_test = None, epochs = 100, rounds = 1, lr = 0.01, verbose = False):    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduling_solver = SolvePointQP(params)
#     optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.95, weight_decay = 1)

    losses = []
    batch_size = 100
    best_loss = 100
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
        if Y_test is not None:
            cur_loss = task_loss(model(Y_test), Y_test, params).mean().item()
            if cur_loss < best_loss: 
                best_loss = cur_loss 
                torch.save(model.state_dict(), 'saved_models/pnet-data-parameters.pt')
        if verbose and epoch % 10 == 0:
            print("epoch ", epoch, " mean loss: ", np.mean(losses[-100:]), " median ", np.median(losses))
            print("test", cur_loss)

                
    # print("MEAN LOSS PROJECT NET: ", np.mean(losses[-100:]))    
    if Y_test is not None:
        print("pnet test", task_loss(model(Y_test), Y_test, params).mean().item())
        model.load_state_dict(torch.load("saved_models/pnet-data-parameters.pt"))
    return model, losses

def train_with_pnet(model, projectnet, train_x, train_y, X_test, Y_test, params, rounds=5, map_size=12, batch_size = 25, epochs = 10, lr = 0.001, verbose = False):    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # n_data = train_x.shape[0]
    # n_train = int(n_data*1)
    # train_x_ = train_x[:n_train,:]
    # train_y_ = train_y[:n_train,:]
    # val_x = train_x[n_train:,:]
    # val_y = train_y[n_train:,:]
    
    times = []
    best_regret = 100
    best_test = 1e5
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = [] 
        
#         if epoch % 1 == 0:
#             model.eval()
#             projectnet.train()
#             train_projectnet(projectnet, model(train_x).detach(), params, Y_test = model(X_test).detach(), epochs=1, rounds=rounds, lr=1e-5, verbose=False)
#             model.train()
#             projectnet.eval()
        
        start = time.time()
        for i in range(0, train_x.shape[0], batch_size):
            x = train_x[i:i+batch_size,:]
            y = train_y[i:i+batch_size,:]

            optimizer.zero_grad()
            
            pred = model(x)
            mse = criterion(pred, y)
            pred = projectnet(pred, tol=1e-3)
            
            loss = task_loss(pred, y, params).mean()
            total_loss = loss #+ 10*mse
            total_loss.backward()

            optimizer.step()       
            train_losses.append(loss.item())
        
        end = time.time()
        length = end - start
        times.append(length)
        
        model.eval()

        d = projectnet(model(X_test).detach()).detach()
        val = task_loss_no_mean(d, Y_test, params).mean().item()
        # d15 = projectnet(model(X_test).detach(), rounds=15).detach()
        # d20 = projectnet(model(X_test).detach(), rounds=20).detach()
        # val10 = task_loss(d10, Y_test, params).mean().item()
        # val15 = task_loss(d15, Y_test, params).mean().item()
        # val20 = task_loss(d20, Y_test, params).mean().item()
        # test_costs = []        
        # for k in range(0, X_test.shape[0], batch_size):
        #     p = model(X_test[k:k+batch_size,:]).detach()
        #     d = projectnet(p)
        #     l = task_loss(d, Y_test[k:k+batch_size,:], params)
        #     cost_test = task_loss_no_mean(d, Y_test[k:k+batch_size,:], params).mean(1).sum().item()
        #     test_costs.append(cost_test)
        # val = np.sum(test_costs)/X_test.shape[0]
        
        model.train()
        if val < best_test: 
            print("SAVING:", val)
            best_test = val
            torch.save(model.state_dict(), 'saved_models/best-model-parameters.pt')      
            
        if epoch % 1 == 0:
            print("epoch ", epoch, " mean loss: ", np.mean(train_losses), "time", np.mean(times))
            print("end-to-end test:", val)
            print() 
            
    model.load_state_dict(torch.load("saved_models/best-model-parameters.pt"))
    return model

def train_task_net(model, params, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    # solver = SolveScheduling(params)
    solver = SolvePointQP(params)

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
            # mu_pred, sig_pred = model(X)
            mu_pred = model(X)
            # y_sched = solver(mu_pred.double(), sig_pred.double())
            y_sched = solver(mu_pred.double())
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
        
        # print("epoch:", epoch, "loss:", np.mean(losses[-100:]), "times:", np.mean(times))
    
    return model
    
    