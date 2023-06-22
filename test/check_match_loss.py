# the code is to check if match_loss work in our pipeline
# we use 2 diff sound model: one has trainable youngs and poisson, another is fixed.
# we use match_loss to try to fit the trainable one to the fixed one.

import sys
sys.path.append('./')
import torch
import os
import configparser
import numpy as np
from datetime import datetime
from src.diff_model import DiffSoundObj
from src.mesh import TetMesh
from src.utils import LOBPCG_solver_freq, mode_loss, load_audio, plot_signal, plot_spec, comsol_mesh_loader
from src.spectrogram import resample
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator
from src.linearFEM.fem import FEMmodel
from src.material_model import LinearElastic, MatSet, Material
from torch.utils.tensorboard import SummaryWriter
from src.solve import BiCGSTAB, WaveSolver

# debug: match loss for two diffSoundObj


def model_match_loss(model1, model2, mode_num=10, sample_num=8, smooth_value=1, eps=1e-3, alpha=2):
    x = torch.ones(mode_num, sample_num, dtype=torch.float64).cuda()

    # (mode_num, sample_num)
    predict = model1.U_hat.T @ model1.stiff_func(model1.U_hat @ x)
    predict = torch.sqrt(predict) / 2 / np.pi
    predict_mean = predict.mean(1)
    predict_std = predict.std(1) + eps  # avoid the value is 0
    # (mode_num, sample_num)
    target = model2.U_hat.T @ model2.stiff_func(model2.U_hat @ x)
    target = torch.sqrt(target) / 2 / np.pi
    target_mean = target.mean(1)
    target_std = target.std(1) + eps  # avoid the value is 0

    # KL divergence matrix (mode_num, mode_num)
    predict_mean = predict_mean.unsqueeze(1)
    predict_std = predict_std.unsqueeze(1)
    target_mean = target_mean.unsqueeze(0)
    target_std = target_std.unsqueeze(0)
    kl = torch.log(target_std / predict_std) + \
        (predict_std**2 + (predict_mean - target_mean)**2) / \
        (2 * target_std**2 + smooth_value) - 0.5

    # loss of foundamental frequency
    loss_base = kl[0, 0]

    # set to set match loss
    weights = 1.0 / (kl + eps)
    # target_importance = 1.0 / (damps + eps) # give higher weight for lower damping
    target_importance = torch.ones((predict.shape[0])).cuda()
    loss = model1.set2set_loss(
        kl, weights, target_importance) + loss_base * alpha
    return loss


config_file_name = 'experiments/configs/check_pipeline.ini'
# Read the configuration file
dir_name = './runs/' + \
    os.path.basename(config_file_name)[
        :-4] + '_' + datetime.now().strftime("%b%d_%H-%M-%S")
config = configparser.ConfigParser()
config.read(config_file_name)

# Create the TensorBoard summary writer
writer = SummaryWriter(dir_name)

# Read the configuration parameters
audio_dir = config.get('audio', 'dir')
sample_rate = config.getint('audio', 'sample_rate')
mode_num = config.getint('train', 'mode_num')
freq_list = config.get('train', 'freq_list').split(', ')
freq_list = [float(f) for f in freq_list]
freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
damp_list = config.get('train', 'damp_list').split(', ')
damp_list = [float(d) for d in damp_list]
noise_rate = config.getfloat('train', 'noise_rate')
max_epoch = config.getint('train', 'max_epoch')
mesh_dir = config.get('mesh', 'dir')
mesh_order = config.getint('mesh', 'order')
frame_num = config.getint('audio', 'frame_num')
# force_frame_num = config.getint('audio', 'force_frame_num')

# the test tet
# vertices = torch.Tensor(
#     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()
# tets = torch.Tensor([[0, 1, 2, 3]]).long().cuda()
vertices, tets = comsol_mesh_loader("assets/multitet.txt")
model = DiffSoundObj(mesh_dir=None, vertices=vertices,
                     tets=tets, mode_num=mode_num, order=1)
gt_model = DiffSoundObj(mesh_dir=None, vertices=vertices,
                        tets=tets, mode_num=mode_num, order=1)
gt_model.material_model = LinearElastic(
    MatSet.Ceramic[1] / MatSet.Ceramic[0], MatSet.Ceramic[2])

optimizer_mat = Adam(model.material_model.parameters(), lr=0.002)
scheduler_mat = lr_scheduler.StepLR(optimizer_mat, step_size=300, gamma=0.99)

EIGEN_DECOMPOSE_CYCLE = 500

for epoch_i in tqdm(range(100000)):

    if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
        # with torch.no_grad():
        #     _, U_hat = LOBPCG_solver_freq(
        #         model.stiff_func, model.mass_matrix, k=mode_num)
        model.eigen_decomposition()
        gt_model.eigen_decomposition()

    loss_model = model_match_loss(model, gt_model)
    writer.add_scalar('loss_model', loss_model.item(), epoch_i)
    loss = loss_model

    optimizer_mat.zero_grad()
    # with torch.autograd.detect_anomaly():
    loss.backward()
    optimizer_mat.step()
    scheduler_mat.step()
    if epoch_i % 100 == 0:
        print('loss_model: ', loss_model.item())

        # get youngs and poisson
        youngs = model.material_model.youngs()
        poisson = model.material_model.poisson()
        writer.add_scalar('youngs', youngs.item(), epoch_i)
        writer.add_scalar('poisson', poisson.item(), epoch_i)
        print('youngs:', youngs, "poisson:", poisson)
        print('gt_youngs:', gt_model.material_model.youngs_modulus,
              "gt_poisson:", gt_model.material_model.poisson_ratio)
