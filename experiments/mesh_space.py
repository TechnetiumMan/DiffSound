import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

from src.diff_model import DiffSoundObj
from src.utils import LOBPCG_solver_freq
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def loss_func(pred, gt):
    R = (pred.unsqueeze(1) - gt) ** 2
    err, indices = R.min(dim=0)
    err = torch.sqrt(err) / gt
    err = err.mean()
    # fundamental_freq
    err += torch.abs(pred[0] - gt[0]) / gt[0]
    return err


eigenvalues = [10626, 26195, 46255, 70042, 77370, 84718]
mode_num = len(eigenvalues)
E = torch.tensor(eigenvalues, dtype=torch.float64).cuda() ** 2
predict_mode_num = 16

mesh_dir = '/data/xcx/mesh_data/full/2/'
model = DiffSoundObj(mesh_dir, order=1)

optimizer = Adam(model.material_model.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

EIGEN_DECOMPOSE_CYCLE = 500

for epoch_i in tqdm(range(50000)):
    if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
        with torch.no_grad():
            _, U_hat = LOBPCG_solver_freq(
                model.stiff_func, model.mass_matrix, k=predict_mode_num)

    E_hat = torch.diagonal(U_hat.T @ model.stiff_func(U_hat))
    loss = loss_func(E_hat, E)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch_i % 100 == 0:
        print('loss: ', loss.item())
        print('predicted eigenvalues:', E_hat**0.5)
        print('gt eigenvalues:', E**0.5)
