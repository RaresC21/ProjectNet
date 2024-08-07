from constants import *
from solver import SolveScheduling
from get_data import *
from train import * 
from models import *
from projectnet import *
from parameters import *

import matplotlib.pyplot as plt
import os.path
import argparse
import time

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

import warnings
warnings.filterwarnings('ignore')


def eval_model(projectnet, test, gd=False, r=False):
    projectnet.eval()
    with torch.no_grad():
        d = projectnet(test, rounds=rounds, tol=1e-3, xrho=0 if gd else None)
        cost_test = task_loss(d, test, params).detach().cpu().numpy().mean()
    # for k in range(0, test.shape[0], batch_size):
    #     cur_data = test[k:k+batch_size,:]
    #     if r: 
    #         d = projectnet(cur_data, rounds=1, xrho=0, rho=0)
    #     else:
    #         d = projectnet(cur_data, rounds=rounds, xrho=0 if gd else None)
    #     cost_test = task_loss(d, test[k:k+batch_size,:], params).detach().cpu().numpy().mean()
    #     net_costs.append(cost_test)
    return cost_test

def get_opt(Y, scheduling_solver): 
    errs = []
    batch_size = 50
    for k in range(0, (Y.shape[0] // batch_size) * batch_size, batch_size):
        d = scheduling_solver(Y[k:k+batch_size,:])
        err = task_loss_no_mean(d, Y[k:k+batch_size,:], params).detach().cpu().numpy()
        errs.append(err)
    return np.mean(errs)

def setup_cvxpy(n):
    x = cp.Variable(n)
    over = cp.Variable(n)
    under = cp.Variable(n)
    y = cp.Parameter(n)
    constraints = [x >= 0]
    for i in range(n-1): 
        constraints.append(cp.abs(x[i] - x[i+1]) <= params['c_ramp'])
    for i in range(n): 
        constraints.append(over[i] >= 0)
        constraints.append(over[i] >= x[i] - y[i])
        
        constraints.append(under[i] >= 0)
        constraints.append(under[i] >= y[i] - x[i])
        
    # objective = cp.Minimize(cp.sum_squares(x - y) * 0.5 + cp.sum(cp.max(x - y, 0)) * params['gamma_over'] + cp.sum(cp.max(y - x, 0)) * params['gamma_under'])
    objective = cp.Minimize(cp.sum_squares(x-y) * 0.5 + cp.sum(over) * params['gamma_over'] + cp.sum(under) * params['gamma_under'])
    problem = cp.Problem(objective, constraints)
    return CvxpyLayer(problem, parameters=[y], variables=[x, over, under])

def eval_cvxpy(Y, cvxpylayer):    
    return cvxpylayer(Y)
        
def test_data():
    n_data = Y_train_pt.shape[0] 
    data_size = [i for i in range(50, 500, 50)] + [i for i in range(500, n_data, 200)]
    errs = []
    errs_learn = []
    for n in data_size:
        cur_data = Y_train_pt[-n:,:]
        pnet = ProjectNet(A, b, 24, params, rounds=rounds, step=step_size, learnable=False).to(DEVICE)
        pnet_learn = ProjectNet(A, b, 24, params, rounds=rounds, step=step_size, learnable=True).to(DEVICE)
        train_projectnet(pnet, cur_data, params, lr=1e-3, epochs=200, verbose=True)
        train_projectnet(pnet_learn, cur_data, params, Y_test=Y_test_pt, lr=1e-3, epochs=200, verbose=True)
        
        err = eval_model(pnet, Y_test_pt)
        err_learn = eval_model(pnet_learn, Y_test_pt)
        errs.append(err)
        errs_learn.append(err_learn)
        print(n, errs[-1], errs_learn[-1])
        
    plt.plot(data_size, errs, label="PNet single")
    plt.plot(data_size, errs_learn, label="PNet learn")
    plt.legend()
    plt.xlabel('data')
    plt.ylabel('cost')
    plt.title('Test Cost vs. Training Data')
    plt.savefig('saved_results/data_comparison.png')

def test_problem_size(): 
    n_data = Y_train_pt.shape[0]    
    prob_size = [i for i in range(1,7,1)]
    errs = []
    learn_errs = []
    gd_errs = []
    
    train_times = []
    test_times = []
    test_learn_times = []
    gd_times = []
    qp_times = []
    cvx_times = []
    base = []
    
    
    for n in prob_size:
        p = n*24
        Y_train, Y_test = get_data_size(n)
        params = {"n": 24*n, "c_ramp": 0.1, "gamma_under": 50, "gamma_over": 0.5}
        scheduling_solver = SolvePointQP(params)
        dist_solver = SolveScheduling(params)

        G = scheduling_solver.G[p*2:p*2+(p-1)*2,:]
        A = torch.cat((G, torch.eye(G.shape[0]).to(DEVICE)), dim=1).float().to(DEVICE)
        b = params['c_ramp'] * torch.ones((p - 1)*2, device=DEVICE).float() 
        
        pnet = ProjectNet(A, b, p, params, rounds=rounds, step=step_size, learnable=False).to(DEVICE)
        pnet_learn = ProjectNet(A, b, p, params, rounds=rounds, step=step_size, learnable=True).to(DEVICE)
        
        start = time.time()
        # train_projectnet(pnet, Y_train[-500:,:], params, lr=1e-3, epochs=200, verbose=True)
        end = time.time()
        train_times.append(end-start)
        
        # train_projectnet(pnet_learn, Y_train[-500:,:], params, lr=1e-3, epochs=200, verbose=True)
        
        start = time.time()
        err = eval_model(pnet, Y_test)
        end = time.time()
        test_times.append(end-start)
        
        start = time.time()
        learn_err = eval_model(pnet_learn, Y_test)
        end = time.time()
        test_learn_times.append(end-start)
        
        start = time.time()
        gd_err = eval_model(pnet, Y_test, gd=True)
        end = time.time()
        gd_times.append(end - start)

        # start = time.time() 
        # get_opt(Y_test, scheduling_solver)
        # end = time.time()
        # qp_times.append((end - start))

        
        cvxpy_layer = setup_cvxpy(Y_test.shape[1])
        start = time.time() 
        eval_cvxpy(Y_test, cvxpy_layer)
        end = time.time()
        cvx_times.append((end - start))

        
        base = eval_model(pnet, Y_test, r=True)
        
        learn_errs.append(learn_err)
        errs.append(err)
        gd_errs.append(gd_err)
        print("n days:", n)
        print("err --", " single pnet", errs[-1], "learn pnet",  learn_errs[-1], "GD",  gd_errs[-1], "qp")
        print("time", test_times[-1], test_learn_times[-1], gd_times[-1], cvx_times[-1]) # qp_times[-1],
        
    plt.plot([i*24 for i in prob_size], errs, label='PNet single')
    plt.plot([i*24 for i in prob_size], learn_errs, label='PNet learn')
    plt.plot([i*24 for i in prob_size], gd_errs, label='PGD')
    plt.legend()
    plt.xlabel('problem size')
    plt.ylabel('cost')
    plt.title('Test Cost vs. Problem Size')
    plt.savefig('saved_results/problem_size_comparison_new.png')

    plt.clf()  
    plt.plot([i*24 for i in prob_size], test_times, label='PNet single')
    plt.plot([i*24 for i in prob_size], test_learn_times, label='PNet learn')
    plt.plot([i*24 for i in prob_size], gd_times, label='PGD')
    # plt.plot([i*24 for i in prob_size], qp_times, label='QP / OptNet')
    plt.plot([i*24 for i in prob_size], cvx_times, label='CVX')
    plt.legend()
    plt.xlabel('problem size')
    plt.ylabel('eval time')
    plt.title('Evluation Times vs. Problem Size')
    plt.savefig('saved_results/time_size_comparison_new.png')

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true')    
    parser.add_argument('--prob_size', action='store_true')    
    args = parser.parse_args()
    
    params = {"n": 24, "c_ramp": 0.1, "gamma_under": 50, "gamma_over": 0.5}
    scheduling_solver = SolvePointQP(params)
    dist_solver = SolveScheduling(params)

    X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data()     

    G = scheduling_solver.G[24*2:24*2+23*2,:]
    A = torch.cat((G, torch.eye(G.shape[0]).to(DEVICE)), dim=1).float().to(DEVICE)
    b = params['c_ramp'] * torch.ones((24 - 1)*2, device=DEVICE).float() 
    
    
    if args.data:
        test_data()
    if args.prob_size: 
        test_problem_size()