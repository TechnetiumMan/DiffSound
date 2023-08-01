import sys
sys.path.append('./')
import torch
import numpy as np
from src.solve import WaveSolver
from src.adjoint import AdjointSolver, calculate_A_b_grad

# 从最简单的例子开始：一维振动！
num_step = 500
dt = 0.001
theta = torch.tensor([1.], requires_grad=True, device="cuda:0")
M = torch.tensor([[0.]]).cuda()
M = M + theta
C = torch.tensor([[0.]]).cuda()
K = torch.tensor([[1.]]).cuda()

# time-dependent force for the displacement
f = torch.zeros([num_step + 1, 1]).cuda() 
# force in timestep 1 instead of 0, 
# because force in timestep 0 in RK4 only work a half.
f[1, 0] = 1. / dt # fixed, with no grad.
def loss(predict):
    return torch.sum(predict) * dt

# 使用RK4算法计算predict以及loss
def C_matvec(x): return torch.zeros_like(x).cuda()
def K_matvec(x): return K @ x
def get_force(t): 
    # 存在t是dt的半整数倍的情况，此时需要判断，并返回前后两个值的平均值
    step_t = int(t / dt)
    half = (2*int(t / dt) != int(2*t / dt)) # 如为真，说明t是dt的半整数倍而非整数倍
    if half:
        return (f[step_t] + f[step_t + 1]) / 2
    else:
        return f[step_t]
    # return f[int(t / dt), 0]
    
solver = WaveSolver(M.to_sparse(), C_matvec, K_matvec, get_force, dt)
u, v = solver.solve(num_step, output_v=True)  # (num_step, num_point)
predict = torch.sum(u, dim=1)
loss_origin = loss(predict)
loss_origin.backward()
print(theta.grad)
theta.grad.data.zero_()

# 使用adjoint方法直接计算梯度
# 首先获得MCK对theta的梯度
dM = torch.tensor([[[1.]]]).cuda() # (num_theta, num_points, num_points)
dC = torch.tensor([[[0.]]]).cuda()
dK = torch.tensor([[[0.]]]).cuda()
A, dA, b, db = calculate_A_b_grad(M, C, K, dM, dC, dK, f[:num_step])

# calculate dg_dx
dg_du = torch.ones_like(u).cuda()
dg_dv = torch.zeros_like(v).cuda()
dg_dx = torch.cat([dg_dv, dg_du], dim=1).T

x = torch.cat([v, u], dim=1).T

adjoint_solver = AdjointSolver(dt, A, dg_dx, num_step)
adjoint_solver.calculate_lambda()
dL_dtheta = adjoint_solver.get_grad(x, dA, db)
print(dL_dtheta)
