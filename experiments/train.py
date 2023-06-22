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
from src.ddsp.oscillator import DampedOscillator, init_damps, GTDampedOscillator
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# from ddsp.py
# Get the directory name from the command line argument
if len(sys.argv) != 2:
    config_file_name = 'experiments/configs/2.ini'
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
    gt_audios.append(audio0)

for force in forces:
    gt_forces.append(
        resample(force.cuda(), origin_sr, sample_rate)[:force_frame_num])

gt_audios = torch.stack(gt_audios)
gt_forces = torch.stack(gt_forces)
log_range_step = config.getint('train', 'log_range_step')
for i in range(0, len(gt_forces), log_range_step):
    writer.add_figure(
        'force_{}th'.format(i),
        plot_signal(gt_forces[i]),
        0)
    
# Create the MSSLoss object
early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
late_loss_func = MSSLoss([512, 256, 128, 64, 32], sample_rate, type='l1_loss').cuda()
log_spec_funcs = [
    late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]
material_coeff = getattr(MatSet, material)
model = build_model(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task, scale_range=scale_range)

pre_osc = GTDampedOscillator(gt_forces, len(
    gt_audios), eigen_num * 16, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()

optimizer_pre_osc = Adam(pre_osc.parameters(), lr=5e-3)
scheduler_pre_osc = lr_scheduler.StepLR(optimizer_pre_osc, step_size=100, gamma=0.99)
for epoch_i in tqdm(range(max_epoch)):
    predict_signal = pre_osc(noise_rate=noise_rate)
    loss = late_loss_func(predict_signal, gt_audios)
    optimizer_pre_osc.zero_grad()
    loss.backward()
    optimizer_pre_osc.step()
    scheduler_pre_osc.step()
    writer.add_scalar('pre_osc_loss', loss.item(), epoch_i)
    if epoch_i % 2000 == 0:
        for i in range(0, len(gt_forces), log_range_step):
            for j in range(len(log_spec_funcs)):
                writer.add_figure(
                    'pre_osc_{}th_{}th'.format(i, j),
                    plot_spec(log_spec_funcs[j](gt_audios[i]),
                                log_spec_funcs[j](predict_signal[i])),
                    epoch_i)

damping = pre_osc.damping()
freq_linear = pre_osc.freq_linear()
mask = damping < 100
damping = damping[mask]
freq_linear = freq_linear[mask]
x = []
y = []
freq_step = 1000
for i in range(20, 20000, freq_step):
    mask = (freq_linear > i) & (freq_linear < i + freq_step)
    damping_ = damping[mask]
    if damping_.shape[0] == 0:
        continue
    x.append(i + freq_step // 2)
    y.append(damping_.min().item())

from scipy import interpolate
damping_curve = interpolate.interp1d(x, y, fill_value="extrapolate")


oscillator = DampedOscillator(gt_forces, len(
    gt_audios), eigen_num, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()
init_damps(oscillator)

# Create the MSSLoss object
early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda()
rmse_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='rmse_loss').cuda()


# Create the optimizer and scheduler
optimizer_osc = Adam(oscillator.parameters(), lr=1e-2)
if task == "material":
    optimizer_model = Adam(model.parameters(mat_param="youngs"), lr=1e-2)
else:
    optimizer_model = Adam(model.parameters(), lr=1e-2)

scheduler_osc = lr_scheduler.StepLR(optimizer_osc, step_size=100, gamma=0.98)
scheduler_model = lr_scheduler.StepLR( 
    optimizer_model, step_size=100, gamma=0.98)

EIGEN_DECOMPOSE_CYCLE = 15

for epoch_i in tqdm(range(max_epoch)):
    # change loss func and optimizer for epoch
    if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
        model.eigen_decomposition()
    undamped_freq = model.get_undamped_freqs().float()

    if epoch_i < early_loss_epoch:
        loss_func = early_loss_func
        predict_signal = oscillator.early(undamped_freq, damping_curve)
    else:
        loss_func = late_loss_func
        predict_signal = oscillator(undamped_freq,
        non_linear_rate=freq_nonlinear, noise_rate=noise_rate)
    
    
    if epoch_i == early_loss_epoch: # now change material model to train poisson
        optimizer_model = Adam(model.parameters(), lr=5e-3)
        scheduler_model = lr_scheduler.StepLR( 
            optimizer_model, step_size=100, gamma=0.98)
    
    damped_freq = oscillator.damped_freq

    spec_scale = 1 #0.2 + 0.8 * epoch_i / max_epoch
    loss = loss_func(predict_signal, gt_audios, damped_freq, spec_scale)
    if epoch_i < early_loss_epoch:
        writer.add_scalar('loss_early', loss.item(), epoch_i)
    else:
        writer.add_scalar('loss_late', loss.item(), epoch_i)
        
    optimizer_osc.zero_grad()
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_osc.step()
    scheduler_osc.step()
    optimizer_model.step()
    scheduler_model.step()

    if epoch_i % (EIGEN_DECOMPOSE_CYCLE*5) == 0:
        with torch.no_grad():
            print('loss_model: ', loss.item())
            print('undamped f:', undamped_freq)
            
            # get RMSE loss
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
                
            for audio_idx in range(0, len(predict_signal), log_range_step):
                for spec_idx in range(len(log_spec_funcs)):
                    writer.add_figure(
                        '{}th_{}'.format(
                            audio_idx, late_loss_func.n_ffts[spec_idx]),
                        plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx], spec_scale),
                                  log_spec_funcs[spec_idx](predict_signal[audio_idx], spec_scale),
                        ),
                        epoch_i)
            torchaudio.save(dir_name + '/predict.mp3',
                            predict_signal[0].detach().cpu().unsqueeze(0), sample_rate)
            torchaudio.save(dir_name + '/gt.mp3',
                            gt_audios[0].detach().cpu().unsqueeze(0), sample_rate)
    if epoch_i % (EIGEN_DECOMPOSE_CYCLE*100) == 0:
        if task == "material":
            torch.save(model.material_model.state_dict(), dir_name + '/model.pth')
        else:
            torch.save(model.scale_model.state_dict(), dir_name + '/model.pth')
        torch.save(oscillator.state_dict(), dir_name + '/oscillator.pth')
