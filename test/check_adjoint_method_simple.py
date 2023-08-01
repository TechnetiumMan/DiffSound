import sys
sys.path.append('./')
import torch
import numpy as np
from src.solve import WaveSolver
from src.adjoint import AdjointSolver, calculate_A_b_grad

# 从最简单的例子开始：一维振动！
num_step = 1000
dt = 0.01
theta = torch.tensor([0.5], requires_grad=True, device="cuda:0")

def loss(predict):
    return torch.sum(predict) * dt

# dx = Ax + b
A = torch.zeros((1, 1)).cuda() # (num_points=1, num_points=1)
A[0, 0] = -theta
dA = torch.zeros((1, 1, 1)).cuda() # (num_theta=1, num_points=1, num_points=1)
dA[0, 0, 0] = -1

b_t = torch.zeros((num_step, 1)).cuda() # (num_step, num_points)
b_t[1, 0] = 1/theta
db_t = torch.zeros((1, num_step, 1)).cuda() # (num_theta, num_step, num_points)
db_t[0, 1, 0] = -1/(theta ** 2)

f = torch.zeros([num_step + 1, 1]).cuda() 
f[1, 0] = 1/theta 

# RK4 to solve the diff equation
def derivative(x, A, b, i, half):
    if half:
        term0 = (b[i] + b[i+1]) / 2
    else:
        term0 = b[i]
    term1 = A @ x
    return (term0 + term1)

xs = torch.zeros([num_step+1]).cuda() # x(0)=0, start calculate at x(1)
x = torch.zeros([1]).cuda()
for i in range(num_step):
    k1 = derivative(x, A, f, i, False)
    k2 = derivative(x + dt / 2 * k1, A, f, i, True)
    k3 = derivative(x + dt / 2 * k2, A, f, i, True)
    k4 = derivative(x + dt * k3, A, f, i+1, False)

    x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xs[i+1] = x
    
xs = xs[:num_step]
loss_origin = loss(xs)
loss_origin.backward()
print(theta.grad)
theta.grad.data.zero_()

# 现在我们使用adjoint方法重新计算
xs = xs.unsqueeze(0) # (num_points, num_step)
dg_dx = torch.ones_like(xs).cuda()
# dg_dx = torch.where(xs > 0, ones, -ones)

adjoint_solver = AdjointSolver(dt, A, dg_dx, num_step)
adjoint_solver.calculate_lambda()
dL_dtheta = adjoint_solver.get_grad(xs, dA, db_t)
print(dL_dtheta)
    

