from constants import *
from solver import SolveScheduling
from get_data import *
from train import * 
from models import *
from projectnet import *
import matplotlib.pyplot as plt
import os.path
import copy 

from parameters import *

import warnings
warnings.filterwarnings('ignore')


def try_get_pnet():
    fname = "saved_models/projectnet{}_{}.pt".format(params['c_ramp'], learnable)
    if os.path.isfile(fname): 
        projectnet = ProjectNet(A, b, 24, params, rounds=rounds, step=step_size, learnable=learnable).to(DEVICE)
        projectnet.load_state_dict(torch.load(fname))
        return projectnet
    return None 

def train_pnet_joint():
    print("train pnet joint")
    
    projectnet = try_get_pnet()
    yy = mle_net(X_test_pt).detach()
    if rerun_pnet or (projectnet is None): 
        projectnet = ProjectNet(A, b, 24, params, rounds=rounds, step=step_size, learnable=learnable).to(DEVICE)
        train_projectnet(projectnet, mle_net(X_train_pt).detach(), params, Y_test=yy, epochs=50, lr=1e-3, verbose=True)
        torch.save(projectnet.state_dict(), "saved_models/projectnet{}_{}.pt".format(params['c_ramp'], learnable))

    pred_test = projectnet(yy)
    t = task_loss(pred_test, yy, params).mean().item()
    print("test val", t)
        
    print('train end-to-end')
    model_pnet = copy.deepcopy(mle_net) #Net(X_train[:,:-1], Y_train, [200,200]).to(DEVICE)
    # model_pnet = Net(X_train[:,:-1], Y_train, [200,200]).to(DEVICE)
    train_with_pnet(model_pnet, projectnet, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt, params, rounds=rounds, batch_size=batch_size, epochs=200, lr=1e-5)
    
    # train_projectnet(projectnet, model_pnet(X_train_pt).detach(), params, epochs=50, verbose=True)
    # train_with_pnet(model_pnet, projectnet, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt, params, rounds=rounds, epochs=150, lr=1e-5)
    
    torch.save(model_pnet.state_dict(), "saved_models/model_pnet{}.pt".format(params['c_ramp']))
    return model_pnet, projectnet

def eval_model(model_pnet, projectnet):
    d = projectnet(model_pnet(X_test_pt).detach()).detach()
    return task_loss_no_mean(d, Y_test_pt, params).detach().cpu().numpy()

def get_task(): 
    task_net = Net2(X_train[:,:-1], Y_train, [200,200]).to(DEVICE)
    train_task_net(task_net, params, X_train_pt, Y_train_pt)
    torch.save(task_net.state_dict(), "saved_models/task_net.pt")
    return task_net

if __name__ == "__main__":
    
    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
    scheduling_solver = SolvePointQP(params)
    dist_solver = SolveScheduling(params)

    X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data()
    
    G = scheduling_solver.G[24*2:24*2+23*2,:]
    A = torch.cat((G, torch.eye(G.shape[0]).to(DEVICE)), dim=1).float().to(DEVICE)
    b = params['c_ramp'] * torch.ones((24 - 1)*2, device=DEVICE).float() 
    
    # MLE baseline 
    print("Training Baseline")
    mle_net = Net(X_train[:,:-1], Y_train, [200, 200]).to(DEVICE)
    train_mle_net(mle_net, params, X_train_pt, Y_train_pt)
    print("MLE TEST", torch.mean((mle_net(X_test_pt) - Y_test_pt) ** 2))
    
    model_pnet, projectnet = train_pnet_joint() 
    
    net_costs = eval_model(model_pnet, projectnet)
    print("cost", np.mean(net_costs))
    
    task_results = np.load("task_results.npy")
    mle_results = np.load("mle_results.npy")
    
    plt.plot(range(24), np.mean(net_costs,axis=0), label='pnet')
    plt.plot(range(24), task_results[:600,:].mean(0), label='task')
    plt.plot(range(24), mle_results.mean(0), label='mle')
    plt.legend()
    plt.savefig("saved_results/hour_comparison.png")