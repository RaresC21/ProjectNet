import torch

use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

dx = [0,0,-1,1,1,1,-1,-1]
dy = [-1,1,0,0,1,-1,1,-1]
