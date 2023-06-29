# this file is for new pipeline: step simulation for non-linear FEM sound synthesis.

import sys
sys.path.append('./')
import torch
import os
import configparser
from datetime import datetime
from src.diff_model import DiffSoundObj, MatSet, TrainableLinear, TrainableNeohookean, Material, build_model
import torchaudio
from src.utils import load_audio, plot_signal, plot_spec, reconstruct_signal, comsol_mesh_loader
from src.spectrogram import resample
from torchaudio.functional import highpass_biquad
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator, init_damps, GTDampedOscillator
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
freq_limit = config.getint('train', 'freq_limit')
upsample = config.getint('train', 'upsample')
max_epoch = config.getint('train', 'max_epoch')
nonlinear_rate = config.getfloat('train', 'freq_nonlinear')
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
audios = audios[:audio_num]
forces = forces[:audio_num]

gt_audios = []
gt_forces = []
for audio in audios:
    audio0 = resample(audio.cuda(), origin_sr, sample_rate)[:frame_num]
    audio0 = highpass_biquad(audio0, sample_rate, 100)
    
    # normalize audio
    audio0 = audio0 / torch.max(torch.abs(audio0))
    gt_audios.append(audio0)

for force in forces:
    # gt_forces.append(
    #     resample(force.cuda(), origin_sr, sample_rate)[:force_frame_num])
    force = resample(force.cuda(), origin_sr, sample_rate)[:frame_num+1] # +1 is used for RK4 last step iter
    force[force_frame_num:] = 0
    gt_forces.append(force.double())


gt_audios = torch.stack(gt_audios)
gt_forces = torch.stack(gt_forces)
log_range_step = config.getint('train', 'log_range_step')
for i in range(0, len(gt_forces), log_range_step):
    writer.add_figure(
        'force_{}th'.format(i),
        plot_signal(gt_forces[i]),
        0)
    
# Create the MSSLoss object
loss_func = MSSLoss([512, 256, 128, 64, 32], sample_rate, type='l1_loss').cuda()
rmse_loss_func = MSSLoss([512, 256, 128, 64, 32], sample_rate, type='rmse_loss').cuda()
log_spec_funcs = [
    loss_func.losses[i].log_spec for i in range(len(loss_func.losses))]

material_coeff = getattr(MatSet, material)
vertices, tets = comsol_mesh_loader("assets/bowl_coarse.txt")
model = build_model(mesh_dir=None, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task, \
    scale_range=scale_range, vertices=vertices, tets=tets)

# init eigen decomp for step init
model.eigen_decomposition(freq=freq_limit)
# now because of freq limit, the real eigen num != the argument eigen_num.
# so we need to update the eigen_num to real value
eigen_num = model.U_hat.shape[1]
model.step_init(gt_forces, 1./(sample_rate * upsample), audio_num, sample_rate, eigen_num, nonlinear_rate)

# Create the optimizer and scheduler
optimizer_model = Adam(model.parameters(), lr=1e-2)
scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size=100, gamma=0.98)

EIGEN_DECOMPOSE_CYCLE = 1

for epoch_i in tqdm(range(max_epoch)):
    # change loss func and optimizer for epoch
    if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
        model.eigen_decomposition() # now eigen_num has been updated, so don't need to use freq_limit
        model.get_graded_eigenvalues()
        # model.step_update()
    
    predict_signal = model.forward(int(frame_num * upsample)) # 10x upsample
    predict_signal = torch.sum(predict_signal.transpose(-1, -2), dim=-1)
    predict_signal = predict_signal[:, ::upsample]
    
    # normalize predict signal
    # Question: should we use a trainable amp for each mode and not normalize it?
    predict_max = torch.max(predict_signal, dim=1)[0].view(-1, 1)
    predict_signal = predict_signal / predict_max

    spec_scale = 1 # 0.2 + 0.8 * epoch_i / max_epoch
    loss = loss_func(predict_signal, gt_audios, None, spec_scale)
    writer.add_scalar('loss', loss.item(), epoch_i)

    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    scheduler_model.step()

    if epoch_i % (EIGEN_DECOMPOSE_CYCLE*5) == 0:
        with torch.no_grad():
            print('loss_model: ', loss.item())
            
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
                            audio_idx, loss_func.n_ffts[spec_idx]),
                        plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx], spec_scale),
                                  log_spec_funcs[spec_idx](predict_signal[audio_idx], spec_scale),
                        ),
                        epoch_i)
            # torchaudio.save(dir_name + '/predict.mp3',
            #                 predict_signal[0].detach().cpu().unsqueeze(0), sample_rate)
            # torchaudio.save(dir_name + '/gt.mp3',
            #                 gt_audios[0].detach().cpu().unsqueeze(0), sample_rate)
    # if epoch_i % (EIGEN_DECOMPOSE_CYCLE*100) == 0:
    #     if task == "material":
    #         torch.save(model.material_model.state_dict(), dir_name + '/model.pth')
    #     else:
    #         torch.save(model.scale_model.state_dict(), dir_name + '/model.pth')

