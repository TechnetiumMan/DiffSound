import sys
sys.path.append('./')
import torch
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
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator, init_damps
from src.solve import WaveSolver
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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
freq_list = config.get('train', 'freq_list').split(', ')
freq_list = [float(f) for f in freq_list]
freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
noise_rate = config.getfloat('train', 'noise_rate')
max_epoch = config.getint('train', 'max_epoch')
mesh_dir = config.get('mesh', 'dir')
mesh_order = config.getint('mesh', 'order')
eigen_num = config.getint('mesh', 'eigen_num')
material = config.get('mesh', 'material')
early_loss_epoch = config.getint('train', 'early_loss_epoch')
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
    gt_audios.append(audio0[100:]) # to avoid initial nonlinear
for force in forces:
    gt_forces.append(
        resample(force.cuda(), origin_sr, sample_rate)[:force_frame_num])
gt_audios = torch.stack(gt_audios)
gt_forces = torch.stack(gt_forces)

material_coeff = getattr(MatSet, material)
mat = Material(material_coeff)
model = build_model(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task="gt", scale_range=scale_range)

# we don't need osc. now we use model reduction to simulate the audio
# first, decompsite the model
# model.eigen_decomposition()
# U = model.U_hat.float()
# S = model.eigenvalues[6:].float()
# torch.save(U, "saves/U.pt")
# torch.save(S, "saves/S.pt")
U = torch.load("saves/U.pt")
S = torch.load("saves/S.pt")

def stiff_matvec(x): return S * x
def damping_matvec(x): 
    return (mat.alpha + mat.beta * S) * x # Rayleigh damping
    # return torch.zeros_like(x).cuda()

# to prevent divergence, upsample in RK4 step simulation
upsample_ratio = 10

# get force
force_node_idx = 5706 * 3 + 2 # force direction is to negative z axis
def get_force(t):
    t_idx = int(t * sample_rate * upsample_ratio)
    force_all_nodes = torch.zeros(
        model.mass_matrix.shape[0]).cuda()
    if t_idx < force_frame_num:
        force_all_nodes[force_node_idx] = -gt_forces[0, t_idx] # force is to negative z axis
        
    # now in model reduction, the force is: f = U^T * force_all_nodes
    f = U.t() @ force_all_nodes
    return f
    
mass_matrix = torch.eye(S.shape[0]).cuda().to_sparse()
dt = 1 / sample_rate / upsample_ratio
solver = WaveSolver("identity", damping_matvec, stiff_matvec, get_force, dt)
with torch.no_grad():
    x = solver.solve(int(frame_num * upsample_ratio))
x = x.transpose(0, 1)
# u=Uq
x = U @ x

# resample to sample_rate
x_sum = torch.sum(x, dim=0)
predict_signal = x_sum[::upsample_ratio]
predict_signal = predict_signal / torch.max(torch.abs(predict_signal))
predict_signal = predict_signal[100:].unsqueeze(0)


# Create the MSSLoss object
late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda() # for spectrogram
log_spec_funcs = [late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]

# now start to synthesis sound
# model.eigen_decomposition()
# undamped_freq = model.get_undamped_freqs().float()

# predict_signal = oscillator(undamped_freq,
#     non_linear_rate=freq_nonlinear, noise_rate=noise_rate)

spec_scale = 1 #0.2 + 0.8 * epoch_i / max_epoch

with torch.no_grad():
    for audio_idx in range(0, len(predict_signal)):
        for spec_idx in range(len(log_spec_funcs)):
            writer.add_figure(
                '{}th_{}'.format(
                    audio_idx, late_loss_func.n_ffts[spec_idx]),
                plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx], spec_scale),
                            log_spec_funcs[spec_idx](predict_signal[audio_idx], spec_scale),
                ),
                0)
    torchaudio.save(dir_name + '/predict.mp3',
                    predict_signal[0].detach().cpu().unsqueeze(0), sample_rate)
    torchaudio.save(dir_name + '/gt.mp3',
                    gt_audios[0].detach().cpu().unsqueeze(0), sample_rate)
