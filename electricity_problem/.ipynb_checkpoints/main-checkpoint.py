from constants import *
from solver import SolveScheduling
from get_data import *
from train import * 
from models import *
from projectnet import *

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
    scheduling_solver = SolveScheduling(params)
    
    X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data() 
    
    # model = Net(X_train[:,:-1], Y_train, [200,200]).to(DEVICE)
    # run_task_net(model, params, X_train_pt, Y_train_pt)
    
    A = torch.cat((scheduling_solver.G, torch.eye(scheduling_solver.G.shape[0]).to(DEVICE)), dim=1).float().to(DEVICE)
    b = params['c_ramp'] * torch.ones((24 - 1)*2, device=DEVICE).float()
    
    # for i in range(A.shape[0]):
    #     print(A[i,:])
        
    # print(torch.linalg.inv_ex(A @ A.T))
    
    projectnet = ProjectNet(A, b, 24, rounds=5).to(DEVICE)
    train_projectnet(projectnet, Y_train_pt, params, epochs=50, verbose=True) 
    
    model = Net(X_train[:,:-1], Y_train, [200,200]).to(DEVICE)
    train_with_pnet(model, projectnet, X_train_pt, Y_train_pt)