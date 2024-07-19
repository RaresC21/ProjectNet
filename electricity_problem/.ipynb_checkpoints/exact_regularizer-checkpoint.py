import cvxpy as cp
import numpy as np

class RegApproximator():
    def __init__(self, data_solutions): 
        self.data = data_solutions 
        
        self.n_data = data_solutions.shape[0]
        self.n = data_solutions.shape[1]
        
        self.C = np.zeros((self.n, self.n))
        for i in range(self.n_data): 
            for j in range(self.n): 
                for k in range(self.n): 
                    self.C[j,k] += self.data[i,j] * self.data[i,k]
        
        
    # alpha <= eigenvalues <= beta
    def get_quadratic_regularizer(self, alpha, beta, verbose=False):
        L = cp.Variable((self.n, self.n), symmetric=True)
        eye = np.eye(self.n)
        constraints = [L - alpha * eye >> 0, beta * eye - L >> 0]
           
        obj = cp.sum(self.C.flatten() * cp.vec(L.T)) 
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(verbose=verbose)
        
        return np.array(L.value)
    
    def linear_par(self, alpha, beta, verbose=False):
        
        # L = cp.Variable((self.n, self.n), symmetric=True)
        B = cp.Variable((self.n*self.n, self.n))
        b = cp.Variable((self.n, self.n))
        
        eye = np.eye(self.n)
        
        
        # constraints = [L - alpha * eye >> 0, beta * eye - L >> 0]
        obj = 0
        constraints = []
        for i in range(self.n_data):
            mat = cp.reshape(B @ self.data[i,:], (self.n, self.n), 'C')+b
            obj += self.data[i,:] @ mat @ self.data[i,:]
            # constraints.append(mat - alpha * eye >> 0)
            constraints.append(cp.sum(mat) >= alpha)
                
        # obj = cp.sum(self.C.flatten() * cp.vec(L.T)) 
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(verbose=verbose)
        
        return np.array(B.value)
        
        