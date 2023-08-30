import sys
sys.path.append('./')
import torch
from src.solve import WaveSolver
from src.adjoint import AdjointSolver, calculate_A_b_grad

num_step = 5000
dt = 0.001
theta = torch.tensor([2., 1., 3.], requires_grad=True, device="cuda:0")

# 来点更复杂的情况：两个点！ num_point = 2
M_origin = torch.diag(torch.tensor([2., 1.])).cuda()
M = M_origin * theta[0]
C = torch.zeros((2, 2)).cuda()
C[0, 0] = theta[1]
K = torch.diag(torch.tensor([1., 1.])).cuda()
K[1, 1] = theta[2]

# time-dependent force for the displacement
f = torch.zeros([num_step + 1, 2]).cuda()  # (num_step, num_point)
# force in timestep 1 instead of 0, 
# because force in timestep 0 in RK4 only work a half.
f[1] = 1. / dt # fixed, with no grad.
def loss(predict):
    return torch.sum(torch.abs(predict))

# 使用RK4算法计算predict以及loss, 利用pytorch的自动微分计算梯度
def C_matvec(x): return C @ x
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
loss_origin = loss(u)
loss_origin.backward()
print(theta.grad)
theta.grad.data.zero_()

# 使用adjoint方法直接计算梯度
# 首先获得MCK对theta的梯度
dM = torch.zeros((3, 2, 2)).cuda() # (num_theta, num_points, num_points)
dM[0] = M_origin
dC = torch.zeros((3, 2, 2)).cuda()
dC[1, 0, 0] = 1.
dK = torch.zeros((3, 2, 2)).cuda()
dK[2, 1, 1] = 1.
A, dA, b, db = calculate_A_b_grad(M, C, K, dM, dC, dK, f[:num_step])

# calculate dg_dx
# dg_du = torch.ones_like(u).cuda()
ones = torch.ones_like(u).cuda()
dg_du = torch.where(u > 0, ones, -ones)

dg_dv = torch.zeros_like(v).cuda()
dg_dx = torch.cat([dg_dv, dg_du], dim=1).T

x = torch.cat([v, u], dim=1).T

adjoint_solver = AdjointSolver(dt, A, dg_dx, num_step)
adjoint_solver.calculate_lambda()
dL_dtheta = adjoint_solver.get_grad(x, dA, db)
print(dL_dtheta)
