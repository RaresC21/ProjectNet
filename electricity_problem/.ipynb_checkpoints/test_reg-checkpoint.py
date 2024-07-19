from constants import *
from solver import SolveScheduling
from get_data import *
from train import * 
from models import *
from projectnet import *
import matplotlib.pyplot as plt

from exact_regularizer import RegApproximator

# import warnings
# warnings.filterwarnings('ignore')

params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
scheduling_solver = SolvePointQP(params)
dist_solver = SolveScheduling(params)

X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data() 


G = scheduling_solver.G[24*2:24*2+23*2,:]
A = torch.cat((G, torch.eye(G.shape[0]).to(DEVICE)), dim=1).float().to(DEVICE)
b = params['c_ramp'] * torch.ones((24 - 1)*2, device=DEVICE).float()

opt_solutions = []
batch_size = 10
for k in range(0, Y_train_pt.shape[0], batch_size):
    d = scheduling_solver(Y_train_pt[k:k+batch_size,:]).detach().cpu().numpy()
    opt_solutions.append(d)
opt_solutions = np.array(opt_solutions)
opt_solutions = np.array(opt_solutions).reshape(opt_solutions.shape[0] * opt_solutions.shape[1], 24)



reg_approximator = RegApproximator(opt_solutions)
alpha = 0.5
beta = 1.0
opt_L = reg_approximator.get_quadratic_regularizer(alpha, beta, verbose=True)

