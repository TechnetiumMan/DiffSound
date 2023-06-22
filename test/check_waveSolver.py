import sys
sys.path.append('./')
import torch
from src.solve import BiCGSTAB, WaveSolver
from src.utils import dense_to_sparse


def test_BiCGSTAB():
    # test the BiCGSTAB solver
    # A is random sparse matrix
    A = torch.rand(10, 10) + 1e-2 + torch.eye(10) * 5
    A = A + A.t()
    A = A.cuda()
    A_sparse = dense_to_sparse(A)
    b = torch.arange(10, dtype=torch.float32).cuda() + 2

    def matmul(x):
        return A_sparse @ x

    A_diag = torch.diagonal(A_sparse.to_dense())

    def preconditioner(x):
        return x / A_diag

    solver = BiCGSTAB(matmul, preconditioner)
    x = solver.solve(b, max_iter=200)
    # print(b)
    # print(A @ x)
    assert torch.allclose(A @ x, b, rtol=1e-4)


def test_inv_mass_matrix():
    # test the inverse of mass matrix
    # A is random sparse matrix
    A = torch.rand(10, 10) + 1e-2 + torch.eye(10) * 5
    A = A + A.t()
    A = A.cuda()
    A_sparse = dense_to_sparse(A)

    b = torch.arange(10, dtype=torch.float32).cuda() + 2

    solver = WaveSolver(A_sparse, None, None, None, None)
    x = solver.mass_linear_solver.solve(b)
    # print(b)
    # print(A @ x)
    assert torch.allclose(solver.mass_matrix_diag,
                          torch.diagonal(A_sparse.to_dense()))
    assert torch.allclose(A @ x, b, atol=1e-4)


import numpy as np
from torch.fft import fft as torch_fft
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def get_all_frequencies(signal, sample_rate, min_freq=0, max_freq=None, plot=False):
    # 计算信号FFT
    fft = torch_fft(signal)
    fft = fft[:len(signal) // 2]  # 保留正半轴
    magnitude = torch.abs(fft)
    if plot:
        # 频率
        freqs = np.arange(len(signal) // 2) / len(signal) * sample_rate
        plt.plot(freqs, magnitude.detach().cpu().numpy())
        plt.show()
    # 找到所有峰值点
    peaks, _ = find_peaks(magnitude.detach().cpu().numpy(),
                          height=magnitude.max().item() / 10)
    # 计算对应的频率
    freqs = peaks / len(signal) * sample_rate
    if max_freq is not None:
        freqs = freqs[freqs <= max_freq]
    if min_freq > 0:
        freqs = freqs[freqs >= min_freq]
    return freqs.tolist()


def test_wave_solver():
    omega = torch.Tensor([100, 200, 300]) * (2 * 3.1415926)
    stiffness_matrix = dense_to_sparse(torch.diag(omega**2).cuda())
    mass_matrix = dense_to_sparse(torch.diag(torch.ones(3)).cuda())
    damping_matrix = dense_to_sparse(torch.zeros(3, 3).cuda())
    dt = 1 / 2000
    def force(t): return torch.Tensor(
        [1, 4, 10]).cuda() if t < dt else torch.zeros(3).cuda()

    def damping_matvec(x): return damping_matrix @ x
    def stiffness_matvec(x): return stiffness_matrix @ x
    solver = WaveSolver(mass_matrix, damping_matvec,
                        stiffness_matvec, force, dt)
    x = solver.solve(1000)
    x = x.mean(dim=1)
    freqs = get_all_frequencies(x, 1 / dt)
    assert torch.allclose(torch.Tensor(freqs), omega /
                          (2 * 3.1415926), atol=10)


if __name__ == '__main__':
    # test_BiCGSTAB()
    # test_inv_mass_matrix()
    test_wave_solver()
    print('check wave solver passed!')
