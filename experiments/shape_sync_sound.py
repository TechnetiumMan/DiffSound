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
from src.mesh import TetMesh
from src.ddsp.oscillator import DampedOscillator, init_damps, GTDampedOscillator, TraditionalDampedOscillator
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
frame_num = config.getint('audio', 'frame_num')
force_frame_num = config.getint('audio', 'force_frame_num')
freq_list = config.get('train', 'freq_list').split(', ')
freq_list = [float(f) for f in freq_list]
freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
noise_rate = config.getfloat('train', 'noise_rate')
max_epoch = config.getint('train', 'max_epoch')
early_loss_epoch = config.getint('train', 'early_loss_epoch')
mesh_dir = config.get('mesh', 'dir')
mesh_order = config.getint('mesh', 'order')
eigen_num = config.getint('mesh', 'eigen_num')
material = config.get('mesh', 'material')
task = config.get('train', 'task') # now have "material" and "shape"
if task == "shape":
    scale_range = config.get('shape', 'scale_range').split(', ')
    scale_range = [float(f) for f in scale_range]
else:
    scale_range = None

# instead of load real audios, we produce audios using gt_osc, and the force is a impulse.
audio_num = 1
gt_forces = torch.zeros((1, force_frame_num)).cuda()
gt_forces[0, 0] = 1

# origin 1st mesh
mesh = TetMesh.from_triangle_mesh(mesh_dir + "model.stl")
gt_vertices = mesh.vertices
gt_tets = mesh.tets

# get gt audios
material_coeff = getattr(MatSet, material)
gt_model = build_model(mesh_dir=None, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task="gt", \
    scale_range=scale_range, vertices=gt_vertices, tets=gt_tets)
gt_oscillator = TraditionalDampedOscillator(gt_forces, audio_num, eigen_num, frame_num, sample_rate, Material(material_coeff)).cuda()
print("before gt eigen decomp")
gt_model.eigen_decomposition()
gt_undamped_freq = gt_model.get_undamped_freqs().float()
# gt_undamped_freq = (torch.randn((1, 16)) * 1000 + 5000).cuda()
gt_audios = gt_oscillator(gt_undamped_freq)

# scale for shape task
gt_scale = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float64).cuda()
vertices = gt_vertices.clone()
tets = gt_tets.clone()

# scale the mesh
vertices = vertices * (1 / gt_scale)
# now ground truth scale are scale defined above

init_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64).cuda()
model = build_model(mesh_dir=None, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task, \
    vertices=vertices, tets=tets, scale_range=scale_range, init_scale=init_scale)

# model.init_scale_coeffs(init_scale)

# oscillator = DampedOscillator(gt_forces, audio_num, eigen_num, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()
# init_damps(oscillator)

# Create the MSSLoss object
early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda()
rmse_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='rmse_loss').cuda()

log_spec_funcs = [
    late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]

# Create the optimizer and scheduler
# optimizer_osc = Adam(oscillator.parameters(), lr=1e-2)
optimizer_model = Adam(model.parameters(), lr=1e-2)

# scheduler_osc = lr_scheduler.StepLR(optimizer_osc, step_size=100, gamma=0.99)
scheduler_model = lr_scheduler.StepLR( 
    optimizer_model, step_size=100, gamma=0.98)

EIGEN_DECOMPOSE_CYCLE = 10

for epoch_i in tqdm(range(max_epoch)):
    if epoch_i < early_loss_epoch:
        loss_func = early_loss_func
    else:
        loss_func = late_loss_func
    
    # redefine optimizer to refresh gradient and momentum caused by early loss(which is much larger than late one)
    if epoch_i == early_loss_epoch:
        optimizer_model = Adam(model.parameters(), lr=5e-3) 
        scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size=100, gamma=0.98)
        
    if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
        model.eigen_decomposition()
    undamped_freq = model.get_undamped_freqs().float()

    predict_signal = gt_oscillator(undamped_freq)
    damped_freq = gt_oscillator.damped_freq

    spec_scale = 1 
    loss = loss_func(predict_signal, gt_audios, damped_freq, spec_scale)
    writer.add_scalar('loss', loss.item(), epoch_i)
    # optimizer_osc.zero_grad()
    optimizer_model.zero_grad()
    loss.backward()
    # optimizer_osc.step()
    optimizer_model.step()
    # scheduler_osc.step()
    scheduler_model.step()

    if epoch_i % (EIGEN_DECOMPOSE_CYCLE*5) == 0:
        with torch.no_grad():
            print('loss_model: ', loss.item())
            print('undamped f:', undamped_freq)
            
            RMSE_loss = rmse_loss_func(predict_signal, gt_audios)
            print('RMSE loss:', RMSE_loss.item())
            writer.add_scalar("RMSE", RMSE_loss.item(), epoch_i)
        
            if task == "material":
                print('youngs module:', model.material_model.youngs().item())
                print('poisson ratio:', model.material_model.poisson().item())
                writer.add_scalar("youngs", model.material_model.youngs().item(), epoch_i)
                writer.add_scalar("poisson", model.material_model.poisson().item(), epoch_i)
                
            elif task == "shape":     
                scalex = model.scale_model()[0, 0].item()
                scaley = model.scale_model()[1, 1].item()
                scalez = model.scale_model()[2, 2].item()
                print("scalex:", scalex)
                print("scaley:", scaley)
                print("scalez:", scalez)
                writer.add_scalar("scalex", scalex, epoch_i)
                writer.add_scalar("scaley", scaley, epoch_i)
                writer.add_scalar("scalez", scalez, epoch_i)
                
            for audio_idx in range(0, len(predict_signal)):
                for spec_idx in range(len(log_spec_funcs)):
                    writer.add_figure(
                        '{}th_{}'.format(
                            audio_idx, late_loss_func.n_ffts[spec_idx]),
                        plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx], spec_scale),
                                  log_spec_funcs[spec_idx](predict_signal[audio_idx], spec_scale),
                        ),
                        epoch_i)
            
            predict_save = predict_signal[0] / torch.max(torch.abs(predict_signal[0]))
            gt_save = gt_audios[0] / torch.max(torch.abs(gt_audios[0]))
            torchaudio.save(dir_name + '/predict.mp3',
                            predict_save.detach().cpu().unsqueeze(0), sample_rate)
            torchaudio.save(dir_name + '/gt.mp3',
                            gt_save.detach().cpu().unsqueeze(0), sample_rate)
            if epoch_i == 0:
                writer.add_figure(
                        'init',
                        plot_spec(log_spec_funcs[1](gt_audios[0], spec_scale),
                                  log_spec_funcs[1](predict_signal[0], spec_scale),
                        ),
                        epoch_i)
                torchaudio.save(dir_name + '/init.mp3',
                            predict_save.detach().cpu().unsqueeze(0), sample_rate)
                
    if epoch_i % (EIGEN_DECOMPOSE_CYCLE*100) == 0:
        if task == "material":
            torch.save(model.material_model.state_dict(), dir_name + '/model.pth')
        else:
            torch.save(model.scale_model.state_dict(), dir_name + '/model.pth')
        # torch.save(oscillator.state_dict(), dir_name + '/oscillator.pth')
