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
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.mesh import TetMesh
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator, init_damps, GTDampedOscillator
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# # from ddsp.py
# # Get the directory name from the command line argument
# config_file_name = sys.argv[1]

# # Read the configuration file
# dir_name = './runs/' + \
#     os.path.basename(config_file_name)[
#         :-4] + '_' + datetime.now().strftime("%b%d_%H-%M-%S")
# config = configparser.ConfigParser()
# config.read(config_file_name)

# # Create the TensorBoard summary writer
# writer = SummaryWriter(dir_name)

# # Read the configuration parameters
# audio_dir = config.get('audio', 'dir')
# sample_rate = config.getint('audio', 'sample_rate')
# freq_list = config.get('train', 'freq_list').split(', ')
# freq_list = [float(f) for f in freq_list]
# freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
# noise_rate = config.getfloat('train', 'noise_rate')
# max_epoch = config.getint('train', 'max_epoch')
# mesh_dir = config.get('mesh', 'dir')
# mesh_order = config.getint('mesh', 'order')
# eigen_num = config.getint('mesh', 'eigen_num')
# material = config.get('mesh', 'material')
# task = config.get('train', 'task') # now have "material" and "shape"
# if task == "shape":
#     scale_range = config.get('shape', 'scale_range').split(', ')
#     scale_range = [float(f) for f in scale_range]
# else:
#     scale_range = None
    
    
# for scale check, we only load model and print its corrsponding matrix
tetmesh = TetMesh.from_triangle_mesh("data/mesh_data/full/2/model.stl")
scale = torch.tensor([[10, 0, 0], [0, 1, 0], [0, 0, 1]],dtype=torch.float64).cuda()
new_vertices = tetmesh.vertices.clone()
new_vertices[:, 0] *= 10 # *10 in x axis
new_tets = tetmesh.tets    

model1 = build_model(mesh_dir=None, mode_num=0, order=1, mat=MatSet.Ceramic, \
    task="debug", vertices=tetmesh.vertices, tets=tetmesh.tets, scale_range=None)
model2 = build_model(mesh_dir=None, mode_num=0, order=1, mat=MatSet.Ceramic, \
    task="debug", vertices=new_vertices, tets=new_tets, scale_range=None)
print("666")
    
    
# # Load the audio data
# audios, forces, origin_sr = load_audio(audio_dir)
# audios = audios[:config.getint('audio', 'audio_num')]
# forces = forces[:config.getint('audio', 'audio_num')]
# frame_num = config.getint('audio', 'frame_num')
# force_frame_num = config.getint('audio', 'force_frame_num')
# gt_audios = []
# gt_forces = []
# for audio in audios:
#     gt_audios.append(
#         resample(audio.cuda(), origin_sr, sample_rate)[:frame_num])
# for force in forces:
#     gt_forces.append(
#         resample(force.cuda(), origin_sr, sample_rate)[:force_frame_num])

# gt_audios = torch.stack(gt_audios)
# gt_forces = torch.stack(gt_forces)
# log_range_step = config.getint('train', 'log_range_step')
# for i in range(0, len(gt_forces), log_range_step):
#     writer.add_figure(
#         'force_{}th'.format(i),
#         plot_signal(gt_forces[i]),
#         0)

# material_coeff = getattr(MatSet, material)

# ### TEST ###
# # we load a model and scale it 0.5x, the trained scale should be 2x
# tetmesh = TetMesh.from_triangle_mesh(mesh_dir + "model.stl")
# vertices = tetmesh.vertices
# vertices[:, 0] *= (10./12)
# # vertices[:, 1] *= (10./12)
# # vertices[:, 2] *= (10./13)
# tets = tetmesh.tets

# model = build_model(mesh_dir=None, mode_num=eigen_num, order=mesh_order, mat=material_coeff, \
#     task=task, vertices=vertices, tets=tets, scale_range=scale_range)
# model.init_scale_coeffs(torch.tensor([1, 1, 1]).cuda())

# oscillator = DampedOscillator(gt_forces, len(
#     gt_audios), eigen_num, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()
# init_damps(oscillator)

# # Create the MSSLoss object
# ddsp_loss_func = MSSLoss([1024, 512, 256, 128, 64, 32], overlap=0.85).cuda()
# log_spec_funcs = [
#     ddsp_loss_func.losses[i].log_spec for i in range(len(ddsp_loss_func.losses))]

# # Create the optimizer and scheduler
# optimizer_osc = Adam(oscillator.parameters(), lr=1e-2)
# optimizer_model = Adam(model.parameters(), lr=5e-3)

# scheduler_osc = lr_scheduler.StepLR(optimizer_osc, step_size=100, gamma=0.99)
# scheduler_model = lr_scheduler.StepLR( 
#     optimizer_model, step_size=100, gamma=0.99)


# # I found that the base freq is not match! the model only stop at a very local minimal,
# # but not try to match the base freq to get a suitable scale!
# # so we need to compare the base freq and add it to loss!
# # we train a gt osc to get the base freq
# # gt_osc = GTDampedOscillator(gt_forces, len(
# #     gt_audios), 128, frame_num, sample_rate, freq_list, Material(material_coeff)).cuda()
# # optimizer_gt_osc = Adam(gt_osc.parameters(), lr=1e-3)
# # init_damps(gt_osc)
# # for epoch_i in tqdm(range(100)):
# #     predict_signal = gt_osc(non_linear_rate=0, noise_rate=0)
# #     gt_loss = ddsp_loss_func(predict_signal, gt_audios) 
# #     optimizer_gt_osc.zero_grad()
# #     gt_loss.backward()
# #     optimizer_gt_osc.step()
# #     if epoch_i % 100 == 0:
# #         print("gt_loss: ", gt_loss.item())

# # with torch.no_grad():
# #     freq= gt_osc.get_sorted_freq()
# #     base_freq = freq[0]

# EIGEN_DECOMPOSE_CYCLE = 15

# for epoch_i in tqdm(range(max_epoch)):
#     if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
#         model.eigen_decomposition()
#     undamped_freq = model.get_undamped_freqs().float()
#     predict_signal = oscillator(undamped_freq,
#         non_linear_rate=freq_nonlinear, noise_rate=noise_rate)
#     loss = ddsp_loss_func(predict_signal, gt_audios, epoch_i)
#     writer.add_scalar('loss', loss.item(), epoch_i)
#     optimizer_osc.zero_grad()
#     optimizer_model.zero_grad()
#     loss.backward()
#     optimizer_osc.step()
#     optimizer_model.step()
#     scheduler_osc.step()
#     scheduler_model.step()

#     if epoch_i % (EIGEN_DECOMPOSE_CYCLE*5) == 0:
#         with torch.no_grad():
#             print('loss_model: ', loss.item())
#             print('undamped f:', undamped_freq)
        
#             if task == "material":
#                 print('youngs module:', model.material_model.youngs().item())
#                 print('poisson ratio:', model.material_model.poisson().item())
#                 writer.add_scalar("youngs", model.material_model.youngs().item(), epoch_i)
#                 writer.add_scalar("poisson", model.material_model.poisson().item(), epoch_i)
                
#             elif task == "shape":     
#                 scalex = model.scale_model()[0, 0].item()
#                 scaley = model.scale_model()[1, 1].item()
#                 scalez = model.scale_model()[2, 2].item()
#                 print("scalex:", scalex)
#                 print("scaley:", scaley)
#                 print("scalez:", scalez)
#                 writer.add_scalar("scalex", scalex, epoch_i)
#                 writer.add_scalar("scaley", scaley, epoch_i)
#                 writer.add_scalar("scalez", scalez, epoch_i)
                
#             for audio_idx in range(0, len(predict_signal), log_range_step):
#                 for spec_idx in range(len(log_spec_funcs)):
#                     writer.add_figure(
#                         '{}th_{}'.format(
#                             audio_idx, ddsp_loss_func.n_ffts[spec_idx]),
#                         plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx]),
#                                   log_spec_funcs[spec_idx](predict_signal[audio_idx]),
#                                   ),
#                         epoch_i)
#             torchaudio.save(dir_name + '/predict.mp3',
#                             predict_signal[0].detach().cpu().unsqueeze(0), sample_rate)
#             torchaudio.save(dir_name + '/gt.mp3',
#                             gt_audios[0].detach().cpu().unsqueeze(0), sample_rate)
