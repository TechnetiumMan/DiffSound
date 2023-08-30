import sys
sys.path.append('./')
import torch
from src.solve import WaveSolver
from src.adjoint import AdjointSolver, calculate_A_b_grad
import os
import configparser
from datetime import datetime
from src.diff_model import DiffSoundObj, MatSet, TrainableLinear, TrainableNeohookean, Material, build_model
import torchaudio
from src.utils import load_audio, plot_signal, plot_spec, reconstruct_signal
from src.spectrogram import resample
from torchaudio.functional import highpass_biquad
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.mesh import TetMesh
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator, init_damps
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Read the configuration file
config_file_name = sys.argv[1]
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
freq_limit = config.getint('train', 'freq_limit')
upsample = config.getint('train', 'upsample')
max_epoch = config.getint('train', 'max_epoch')
freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
damp_nonlinear = config.getfloat('train', 'damp_nonlinear')
mesh_dir = config.get('mesh', 'dir')
mesh_order = config.getint('mesh', 'order')
eigen_num = config.getint('mesh', 'eigen_num')
material = config.get('mesh', 'material')
audio_num = config.getint('audio', 'audio_num')
frame_num = config.getint('audio', 'frame_num')
force_frame_num = config.getint('audio', 'force_frame_num')
task = config.get('train', 'task') # now have "material" and "shape"
if task == "shape":
    scale_range = config.get('shape', 'scale_range').split(', ')
    scale_range = [float(f) for f in scale_range]
else:
    scale_range = None
    
# Load the audio data
audios, forces, origin_sr = load_audio(audio_dir)
audios = audios[:config.getint('audio', 'audio_num')]
forces = forces[:config.getint('audio', 'audio_num')]
frame_num = config.getint('audio', 'frame_num')
force_frame_num = config.getint('audio', 'force_frame_num')
gt_audios = []
gt_forces = []
for audio in audios:
    audio0 = resample(audio.cuda(), origin_sr, sample_rate)[:frame_num]
    audio0 = highpass_biquad(audio0, sample_rate, 100)
    
    # normalize audio
    audio0 = audio0 / torch.max(torch.abs(audio0))
    gt_audios.append(audio0)
    
for force in forces:
    force = resample(force.cuda(), origin_sr, sample_rate)[:frame_num+1] # +1 is used for RK4 last step iter
    force[force_frame_num:] = 0
    gt_forces.append(force.double())
    
gt_audios = torch.stack(gt_audios)
gt_forces = torch.stack(gt_forces)

material_coeff = getattr(MatSet, material)
late_loss_func = MSSLoss([512, 256, 128, 64, 32], sample_rate, type='l1_loss').cuda()
log_spec_funcs = [
    late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]
material_coeff = getattr(MatSet, material)
model = build_model(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task)

# now for testing, we calculate only 1 epoch
# at first model reduction
model.eigen_decomposition(freq=freq_limit)
eigen_num = model.U_hat.shape[1]
model.linear_step_init(gt_forces, 1./(sample_rate * upsample), audio_num, sample_rate, eigen_num)

# using wavesolver to forward
model.get_grad_eigenvalues()
predict_signal_origin = model.forward(int(frame_num * upsample)) # used for adjoint method

# write as a function to calculate the jacobian for adjoint method
def get_loss(predict_signal_origin):
    predict_signal = torch.sum(predict_signal_origin.transpose(-1, -2), dim=-1)
    predict_signal = predict_signal[:, ::upsample]
    
    predict_max = torch.max(torch.abs(predict_signal), dim=1)[0].view(-1, 1)
    predict_signal = predict_signal / predict_max
    
    loss = late_loss_func(predict_signal, gt_audios, None, 1)
    return loss

loss = get_loss(predict_signal_origin)
loss.backward()
print(model.material_model.youngs_value.grad, model.material_model.poisson_value.grad)

# now in adjoint method, we don't need any autograd
# we need to calculate the jacobian of the loss w.r.t. the predict x
dg_dx = torch.autograd.functional.jacobian(get_loss, predict_signal_origin)
print(dg_dx.shape)

# calculate dM/dtheta and dK/dtheta in original M and K (before model reduction)
# when theta is youngs and poisson, dM/dtheta are zero
dM0_dtheta = torch.zeros_like(model.mass_matrix) 
dK0_dtheta = model.jacobian_dK_dtheta()

dS_dtheta = model.U_hat.transpose(-1, -2) @ dK0_dtheta @ model.U_hat

dM_dtheta = torch.zeros_like(dS_dtheta)
dC_dtheta = model.material_model.mat.beta * dS_dtheta
dK_dtheta = dS_dtheta







