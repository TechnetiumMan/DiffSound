import sys
sys.path.append('./')

import torch
import os
import configparser
from datetime import datetime
from src.diff_model import DiffSoundObj, MatSet, TrainableLinear, TrainableNeohookean, Material
import torchaudio
from src.utils import load_audio, plot_signal, plot_spec, reconstruct_signal
from src.spectrogram import resample
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator, init_damps
from torch.utils.tensorboard import SummaryWriter
import numpy as np

config_file_name = 'experiments/configs/2.ini'
config = configparser.ConfigParser()
config.read(config_file_name)
# Read the configuration parameters
audio_dir = config.get('audio', 'dir')
sample_rate = config.getint('audio', 'sample_rate')
freq_list = config.get('train', 'freq_list').split(', ')
freq_list = [float(f) for f in freq_list]
freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
noise_rate = config.getfloat('train', 'noise_rate')
max_epoch = config.getint('train', 'max_epoch')
mesh_dir = config.get('mesh', 'dir')
mesh_order = config.getint('mesh', 'order')
eigen_num = config.getint('mesh', 'eigen_num')
material = config.get('mesh', 'material')
mic_pos = config.get('bem', 'mic_pos').split(', ')
mic_pos = [float(p) for p in mic_pos]

# Load the audio data
audios, forces, origin_sr = load_audio(audio_dir)
audios = audios[:config.getint('audio', 'audio_num')]
forces = forces[:config.getint('audio', 'audio_num')]
frame_num = config.getint('audio', 'frame_num')
force_frame_num = config.getint('audio', 'force_frame_num')
gt_audios = []
gt_forces = []
for audio in audios:
    gt_audios.append(
        resample(audio.cuda(), origin_sr, sample_rate)[:frame_num])
for force in forces:
    gt_forces.append(
        resample(force.cuda(), origin_sr, sample_rate)[:force_frame_num])

gt_audios = torch.stack(gt_audios)
gt_forces = torch.stack(gt_forces)
material_coeff = getattr(MatSet, material)
oscillator = DampedOscillator(gt_forces, len(gt_audios), eigen_num, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()
log_dir = '/home/jxt/High-Order-DiffFEM/runs/2_May20_15-38-03'
oscillator.load_state_dict(torch.load(log_dir + '/oscillator.pth'))
model = DiffSoundObj(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, mat_model=TrainableLinear)
model.material_model.load_state_dict(torch.load(log_dir + '/model.pth'))

import meshio
import scipy.spatial as spatial
# from src.visualize import viewer
surf_mesh = meshio.read(mesh_dir + 'model.stl__sf.obj')
surf_veticies = surf_mesh.points
surf_triangles = surf_mesh.cells[0].data
model.eigen_decomposition()

kdtree = spatial.KDTree(model.tetmesh.vertices.cpu().numpy())
_, idx = kdtree.query(surf_veticies)
U = model.U_hat.reshape(-1, 3, eigen_num)
U = U[idx]
U_tris = (U[surf_triangles[:, 0]] + U[surf_triangles[:, 1]] + U[surf_triangles[:, 2]]) / 3
e1 = torch.tensor(surf_veticies[surf_triangles[:, 0]] - surf_veticies[surf_triangles[:, 1]]).cuda()
e2 = torch.tensor(surf_veticies[surf_triangles[:, 0]] - surf_veticies[surf_triangles[:, 2]]).cuda()
surf_normal = torch.cross(e1, e2).unsqueeze(-1)
surf_normal = surf_normal / torch.norm(surf_normal, dim=1, keepdim=True)
U_tris = torch.abs((U_tris * surf_normal).sum(dim=1))
U_tris = U_tris.cpu().numpy()

np.save(log_dir + '/U_tris.npy', U_tris)
np.save(log_dir + '/surf_veticies.npy', surf_veticies)
np.save(log_dir + '/surf_triangles.npy', surf_triangles)
np.save(log_dir + '/eigenvalues.npy', model.eigenvalues[6:].cpu().numpy())
print('Eigenvector saved.')


lbd = model.eigenvalues[6:].unsqueeze(0).unsqueeze(-1)
damp = 0.5 * (oscillator.alpha() + oscillator.beta() * lbd)
freq = (lbd - damp**2)**0.5 / (2 * np.pi)  
print(freq.shape)
omega = (2 * np.pi * freq)[0, :, 0].detach().cpu().numpy()
print(omega)
np.save(log_dir + '/omega.npy', omega)


from src.bem import BEMModel
from numba import njit

@njit()
def unit_cube_surface_points(res):
    face_datas = np.array([
        [ 0, 0,  1],[ 0, 1, 0],[ 1, 0, 0],
        [ 0, 0, -1],[ 0, 1, 0],[ 1, 0, 0],
        [ 0, 1,  0],[ 1, 0, 0],[ 0, 0, 1],
        [ 0,-1,  0],[ 1, 0, 0],[ 0, 0, 1],
        [ 1, 0,  0],[ 0, 1, 0],[ 0, 0, 1],
        [-1, 0,  0],[ 0, 1, 0],[ 0, 0, 1],
    ]).reshape(-1, 3, 3)
    points = np.zeros((6, res, res, 3))
    for face_idx, data in enumerate(face_datas):
        normal, w, h = data
        dw, dh = w/(res-1), h/(res-1)
        p0 = 0.5*normal - 0.5*w - 0.5*h
        for i in range(res):
            for j in range(res):
                points[face_idx, i, j] = p0 + i*dw + j*dh
    return points

bem = BEMModel(surf_veticies, surf_triangles)
c = 343
air_density = 1.225
wave_num = omega / c
mic_pos = np.array(mic_pos).reshape(1, 3)
mic_pressure = []
img_res = 32
cube_points = unit_cube_surface_points(img_res).reshape(-1, 3)*0.15
imgs = []
for i in tqdm(range(eigen_num)):
    neumann = U_tris[:, i] * air_density * omega[i]**2
    bem.boundary_equation_solve(neumann, wave_num[i])
    mic_pressure.append(np.abs(bem.potential_solve(mic_pos)))
    cube_pressure = np.abs(bem.potential_solve(cube_points))
    imgs.append(cube_pressure.reshape(6*img_res, img_res))

print(mic_pressure)
mic_pressure = np.array(mic_pressure).reshape(-1)
np.save(log_dir + '/mic_pressure.npy', mic_pressure)

for i in range(16):
    amp = oscillator.amp()[i].reshape(-1).detach().cpu().numpy()
    np.save(log_dir + '/amp_{}.npy'.format(i), amp)
