# the code is for checking our full pipeline
# we build a/some simple tet(s), use fem to get its 1st-order modal signal as a ground truth,
# and use our method to fit the result

# for higher order, we have to use real sounds, because we have no high-order nonlinear modal gt.
# it is trained in experiments/pipeline.py

import sys
sys.path.append('./')
import torch
import os
import configparser
from datetime import datetime
from src.diff_model import DiffSoundObj
from src.mesh import TetMesh
from src.utils import LOBPCG_solver_freq, plot_spec, comsol_mesh_loader, reconstruct_signal
from src.spectrogram import resample
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator
from src.linearFEM.fem import FEMmodel
from src.material_model import LinearElastic, MatSet, Material
from torch.utils.tensorboard import SummaryWriter
from src.solve import BiCGSTAB, WaveSolver

# from ddsp.py
# Get the directory name from the command line argument
if len(sys.argv) != 2:
    config_file_name = 'experiments/configs/check_pipeline.ini'
else:
    config_file_name = sys.argv[1]

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
vertices, tets = comsol_mesh_loader("assets/plate_coarse.txt")
model = DiffSoundObj(mesh_dir=None, vertices=vertices,
                     tets=tets, mode_num=8, order=1)


# generage the ground truth audio
mesh = TetMesh(vertices, tets)
fem = FEMmodel(mesh.vertices, mesh.tets, Material(MatSet.Ceramic))
gt_forces = torch.zeros((1, frame_num)).cuda()
gt_forces[0, 0] = 1

# we have to use model reduction to avoid high freq modes
gt_eigenvalue, gt_U = LOBPCG_solver_freq(
    fem.stiffness_matrix, fem.mass_matrix, k=10, freq_limit=sample_rate / 2)
gt_S = torch.diag(gt_eigenvalue)

# damping_matrix = torch.zeros(12, 12).cuda()


def damping_matvec(x): return torch.zeros_like(x).cuda()
def stiff_matvec(x): return gt_S @ x


def get_force(t):
    f = torch.zeros((fem.stiffness_matrix.shape[0])).cuda()
    if (t == 0):
        f[0] = 1e10  # for faster convergence
    force = gt_U.t() @ f
    return force


solver = WaveSolver('identity', damping_matvec,
                    stiff_matvec, get_force, 1 / sample_rate / 5)
with torch.no_grad():
    gt_audios = solver.solve(frame_num * 5)
    gt_audios = gt_audios.transpose(0, 1)
    gt_audios = gt_U @ gt_audios  # (point_num, sample_num)
    gt_audios = gt_audios[0].unsqueeze(0)
    # gt_audios = gt_audios[:, 0] # sample first node
    # resample
    gt_audios = resample(gt_audios, sample_rate * 5, sample_rate)


# gt_audios = torch.stack(gt_audios)
# gt_forces = torch.stack(gt_forces)
log_range_step = config.getint('train', 'log_range_step')

oscillator = DampedOscillator(gt_forces, len(
    gt_audios), mode_num, frame_num, sample_rate, freq_list, damp_list).cuda()

# Create the MSSLoss object
ddsp_loss_func = MSSLoss([2048, 1024, 512, 256, 128, 64]).cuda()
log_spec_funcs = [
    ddsp_loss_func.losses[i].log_spec for i in range(len(ddsp_loss_func.losses))]

# Create the optimizer and scheduler
optimizer_osc = Adam(oscillator.parameters(), lr=0.002)
optimizer_mat = Adam(model.material_model.parameters(), lr=0.005)
scheduler_osc = lr_scheduler.StepLR(optimizer_osc, step_size=1000, gamma=0.98)
scheduler_mat = lr_scheduler.StepLR(optimizer_mat, step_size=100, gamma=0.99)

EIGEN_DECOMPOSE_CYCLE = 100
WARMUP_DDSP_EPOCH = 30000

for epoch_i in tqdm(range(60000)):

    # train oscillator
    predict_signal = oscillator(
        non_linear_rate=freq_nonlinear, noise_rate=noise_rate)
    loss_osc = ddsp_loss_func(predict_signal, gt_audios)
    writer.add_scalar('loss_osc', loss_osc.item(), epoch_i)
    if epoch_i % 100 == 0:
        print('loss_osc: ', loss_osc.item())
    loss = loss_osc
    optimizer_osc.zero_grad()
    undamped_freq, damp = oscillator.get_sorted_freq_damp()

    # train model
    if epoch_i >= WARMUP_DDSP_EPOCH:
        if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0 or epoch_i == WARMUP_DDSP_EPOCH + 1:
            model.eigen_decomposition()
        loss_model = model.match_loss(undamped_freq, damp)
        writer.add_scalar('loss_model', loss_model.item(), epoch_i)
        loss += loss_model
        optimizer_mat.zero_grad()
        if epoch_i % 100 == 0:
            print('loss_model: ', loss_model.item())
            print('undamped f:', undamped_freq.mean(
                0), '±', undamped_freq.std(0))
            print('predict f:', model.predict.mean(
                1), '±', model.predict.std(1))
            print('damp:', damp)
            print('youngs module:', model.material_model.youngs().item())
            print('poisson ratio:', model.material_model.poisson().item())


    optimizer_osc.zero_grad()
    optimizer_mat.zero_grad()
    # with torch.autograd.detect_anomaly():
    loss.backward()
    optimizer_osc.step()
    optimizer_mat.step()
    # if epoch_i >= WARMUP_DDSP_EPOCH:
    #     scheduler_mat.step()
    if epoch_i % 500 == 0:
        with torch.no_grad():
            freq_gt = reconstruct_signal(
                undamped_freq.mean(0), damp, frame_num, sample_rate)
            
            for audio_idx in range(0, len(predict_signal), log_range_step):
                for spec_idx in range(len(log_spec_funcs)):
                    writer.add_figure(
                        '{}th_{}'.format(
                            audio_idx, ddsp_loss_func.n_ffts[spec_idx]),
                        plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx]),
                                  log_spec_funcs[spec_idx](predict_signal[audio_idx]),
                                    log_spec_funcs[spec_idx](freq_gt)
                        ),
                        epoch_i)

