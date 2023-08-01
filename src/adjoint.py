# code for adjoint method
# see in notes/adjoint-method.py

import sys
sys.path.append('./')
import torch
from tqdm import tqdm

class AdjointSolver():
    def __init__(self, dt, A, dg_dx, num_step):
        self.A = A 
        self.dt = dt
        self.dg_dx = dg_dx
        self.num_step = num_step
        self.num_points = dg_dx.shape[0] 
    
    # dg_dx + lambda.T @ A + \dot{lambda.t} = 0, lambda(T)=0
    def calculate_lambda(self):
        print("start calculate adjoint lambda")
        # dg_dx: (num_step, n)
        # self.P = self.dg_dx
        self.P = torch.cat([self.dg_dx, torch.zeros(self.num_points, 1).cuda()], dim=1) # (num_step+1, n)

        # notice that we know lambda(num_step)=0, and the solver solves it from num_step to 0
        self.P = torch.flip(self.P, [1])
        lbd = torch.zeros(self.num_points, self.num_step + 1).cuda()
        
        # lambda(t)(0~num_points)
        lt = torch.zeros([self.num_points]).cuda()
        # now lambda(0) = 0

        # RK4 to solve differential equation for lambda
        for i in tqdm(range(self.num_step)):
            k1 = self.derivative(lt, i, False)
            k2 = self.derivative(lt + self.dt / 2 * k1, i, True)
            k3 = self.derivative(lt + self.dt / 2 * k2, i, True)
            k4 = self.derivative(lt + self.dt * k3, i+1, False)

            lt = lt + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            lbd[:, i+1] = lt
            
        # flip the lambda back
        lbd = torch.flip(lbd, [1])
        self.lbd = lbd # (num_points+1, num_step) with lambda(num_points) = 0
    
    def derivative(self, lt, i, half):
        if half:
            term0 = (self.P[:, i] + self.P[:, i+1]) / 2
        else:
            term0 = self.P[:, i]
        term1 = self.A.T @ lt
        return (term0 + term1)
    
    def get_grad(self, x, dA_dtheta, db_dtheta_t):
        # x: (num_step, num_point)
        # dA_dtheta: (num_theta, num_points, num_points)
        # db_dtheta_t: (num_theta, num_step, num_points)
        num_theta = dA_dtheta.shape[0]
        dA_dtheta = torch.flatten(dA_dtheta, start_dim=1, end_dim=2)
        dA_dtheta = dA_dtheta.T
        # dA_dtheta(flatted in dim0): (num_point ** 2, num_theta)

        # this is the grad from loss to theta
        dL_dtheta = torch.zeros(num_theta).cuda()
        
        # start to accumulate the grad
        print("start calculate adjoint grad")
        for i in tqdm(range(self.num_step)): # notice that because lambda(num_step)=0, don't need to calculate to num_step+1
            df_dA = x[:, i]
            df_dA = df_dA.unsqueeze(-1).unsqueeze(-1) * torch.eye(self.num_points).cuda() # (num_points, num_points, num_points)
            df_dA = df_dA.permute(1, 2, 0) 
            df_dA = torch.flatten(df_dA, start_dim=1, end_dim=2) # (num_points, num_points ** 2)
            df_dtheta = df_dA @ dA_dtheta
            
            if db_dtheta_t is not None:
                # df_db = torch.ones((db_dtheta_t.shape[-1])).cuda() # diag (num_points)
                db_dtheta = db_dtheta_t[:, i, :].T # (num_points, num_theta)
                # df_dtheta += df_db @ db_dtheta # (num_points, num_theta)
                df_dtheta += db_dtheta # df_db are identity
            
            add_term = self.lbd[:, i] @ df_dtheta
            dL_dtheta += add_term * self.dt # integration!
            
        return dL_dtheta
    
def calculate_A_b_grad(M, C, K, dM, dC, dK, f):
    # M, C, K: (num_points, num_points)
    # dM, dC, dK: grad for theta, (num_points, num_points)
    num_points = M.shape[0]
    num_theta = dM.shape[0]
    A = torch.zeros((2*num_points, 2*num_points)).cuda()
    M_inv = torch.inverse(M)
    A[:num_points, :num_points] = -M_inv @ C
    A[:num_points, num_points:] = -M_inv @ K
    A[num_points:, :num_points] = torch.eye(num_points)
    
    # calculate gradients
    dM_inv = -M_inv @ dM @ M_inv # (num_theta, num_points, num_points)
    dM_inv_C = dM_inv @ C + M_inv @ dC
    dM_inv_K = dM_inv @ K + M_inv @ dK
    
    dA = torch.zeros((num_theta, 2*num_points, 2*num_points)).cuda()
    dA[:, :num_points, :num_points] = -dM_inv_C
    dA[:, :num_points, num_points:] = -dM_inv_K
    
    # b is time-dependent! b = [M_inv f, 0]
    num_step = f.shape[0] # f: (num_steps, num_points)
    b = torch.zeros((num_step, 2*num_points)).cuda()
    b[:, :num_points] = (M_inv @ f.T).T
    
    db = torch.zeros((num_theta, num_step, 2*num_points)).cuda()
    db[:, :, :num_points] = (dM_inv @ f.T).transpose(1, 2)
    return A, dA, b, db

        