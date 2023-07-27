import sys
sys.path.append('./')
import torch
import numpy as np
from src.solve import WaveSolver
from src.adjoint import AdjointSolver, calculate_A_grad

# 从最简单的例子开始：一维振动！
num_step = 100
dt = 0.1
theta = torch.tensor([-1.], requires_grad=True, device="cuda:0")
# M = torch.tensor([[0.]]).cuda()
# M = M + theta
# C = torch.tensor([[0.]]).cuda()
# K = torch.tensor([[1.]]).cuda()

# +2的目的是，RK4需要下一步的值来更精确计算，并且lambda边界条件需要多算一位
f = torch.zeros([num_step + 1]).cuda() 
f[0] = 1. / dt
gt = torch.zeros([num_step + 1]).cuda()
def loss(predict, gt):
    # return torch.sum(torch.abs(predict - gt))
    return torch.sum(predict) * dt

# # 使用RK4算法计算predict以及loss
# def C_matvec(x): return torch.zeros_like(x).cuda()
# def K_matvec(x): return K @ x
# def get_force(t): return f[int(t)]
# solver = WaveSolver(M.to_sparse(), C_matvec, K_matvec, get_force, dt)
# u, v = solver.solve(num_step, output_v=True)  # (num_step, num_point)
# predict = torch.sum(u, dim=1)
# loss_origin = loss(predict, gt)
# loss_origin.backward()
# print(theta.grad)
# theta.grad.data.zero_()

# # 使用adjoint方法直接计算梯度
# # 首先获得MCK对theta的梯度
# dM = torch.tensor([[[1.]]]).cuda() # (num_theta, num_points, num_points)
# dC = torch.tensor([[[0.]]]).cuda()
# dK = torch.tensor([[[0.]]]).cuda()
# A, dA = calculate_A_grad(M, C, K, dM, dC, dK)

# # calculate dg_dx
# ones = torch.ones_like(u).cuda()
# dg_du = torch.where(u > 0, ones, -ones)
# dg_dv = torch.zeros_like(v).cuda()
# dg_dx = torch.cat([dg_dv, dg_du], dim=1).T

# x = torch.cat([v, u], dim=1).T

# adjoint_solver = AdjointSolver(dt, A, dg_dx, num_step)
# adjoint_solver.calculate_lambda()
# dL_dtheta = adjoint_solver.get_grad(x, dA)
# print(dL_dtheta)

# 上面的例子仍然不对，我们需要更简单的测试！
# dx = Ax + b
A = torch.zeros((1, 1)).cuda() # (num_points=1, num_points=1)
A = A + theta
dA = torch.tensor([[[1.]]]).cuda() # (num_theta=1, num_points=1, num_points=1)

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
loss_origin = loss(xs, gt)
loss_origin.backward()
print(theta.grad)
theta.grad.data.zero_()

# 现在我们使用adjoint方法重新计算
xs = xs.unsqueeze(0) # (num_points, num_step)
dg_dx = torch.ones_like(xs).cuda() * dt
# dg_dx = torch.where(xs > 0, ones, -ones)

adjoint_solver = AdjointSolver(dt, A, dg_dx, num_step)
adjoint_solver.calculate_lambda()
dL_dtheta = adjoint_solver.get_grad(xs, dA)
print(dL_dtheta)
    
# 终于成功了！
