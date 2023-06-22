import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import os
from src.utils import load_audio
from src.spectrogram import resample
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import DampedOscillator
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import configparser
import shutil
from datetime import datetime


def plot_spec(spec_gt, spec_predict):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(spec_gt.detach().cpu().numpy(),
                  origin="lower", aspect="auto", cmap='magma')
    axs[1].imshow(spec_predict.detach().cpu().numpy(),
                  origin="lower", aspect="auto", cmap='magma')
    axs[0].set_title('Ground Truth')
    axs[1].set_title('Predict')
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    fig.tight_layout(pad=0)
    return fig


def plot_signal(siganl):
    fig, ax = plt.subplots(1, 1)
    ax.plot(siganl.detach().cpu().numpy())
    fig.tight_layout(pad=0)
    return fig


if __name__ == "__main__":
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
    mode_num = config.getint('train', 'mode_num')
    freq_list = config.get('train', 'freq_list').split(', ')
    freq_list = [float(f) for f in freq_list]
    freq_nonlinear = config.getfloat('train', 'freq_nonlinear')
    damp_list = config.get('train', 'damp_list').split(', ')
    damp_list = [float(d) for d in damp_list]
    noise_rate = config.getfloat('train', 'noise_rate')
    max_epoch = config.getint('train', 'max_epoch')
    warmup_epoch = config.getint('train', 'warmup_epoch')

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
    log_range_step = config.getint('train', 'log_range_step')
    for i in range(0, len(gt_forces), log_range_step):
        writer.add_figure(
            'force_{}th'.format(i),
            plot_signal(gt_forces[i]),
            0)

    # Create the DampedOscillator object and ModalInternalForce object
    oscillator = DampedOscillator(gt_forces, len(
        gt_audios), mode_num, frame_num, sample_rate, freq_list, damp_list).cuda()
    # modal_force = ModalInternalForce(freq_list, mode_num).cuda()

    # Create the MSSLoss object
    loss_func = MSSLoss([2048, 1024, 512, 256, 128, 64]).cuda()
    log_spec_funcs = [
        loss_func.losses[i].log_spec for i in range(len(loss_func.losses))]

    # Create the optimizer and scheduler
    optimizer_osc = Adam(oscillator.parameters(), lr=0.002)
    # optimizer_modal = Adam(modal_force.parameters(), lr=0.001)

    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

    for epoch_i in tqdm(range(max_epoch)):
        # Train the oscillator
        predict_signal = oscillator(
            non_linear_rate=freq_nonlinear, noise_rate=noise_rate)
        loss1 = loss_func(predict_signal, gt_audios)
        loss = loss1
        writer.add_scalar('loss_osc', loss1.item(), epoch_i)

        # warm up
        # if epoch_i > 20000:
        #     # Train the modal force
        #     q, k = oscillator.get_modal_data()
        #     predict_k = modal_force(q)
        #     loss2 = F.l1_loss(predict_k.log(), k.log())
        #     writer.add_scalar('loss_modal', loss2.item(), epoch_i)
        #     loss += loss2
        # else:
        #     loss2 = torch.tensor(torch.nan)

        optimizer_osc.zero_grad()
        # optimizer_modal.zero_grad()
        loss.backward()
        optimizer_osc.step()
        # optimizer_modal.step()

        if epoch_i % 1000 == 0:
            with torch.no_grad():
                print('epoch: {}, loss1: {}'.format(
                    epoch_i, loss1.item()))
                for audio_idx in range(0, len(predict_signal), log_range_step):
                    for spec_idx in range(len(log_spec_funcs)):
                        writer.add_figure(
                            '{}th_{}'.format(
                                audio_idx, loss_func.n_ffts[spec_idx]),
                            plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx]),
                                      log_spec_funcs[spec_idx](
                                          predict_signal[audio_idx]),
                                      ),
                            epoch_i)
                torchaudio.save(dir_name + '/predict.mp3',
                                predict_signal[0].detach().cpu().unsqueeze(0), sample_rate)
                torchaudio.save(dir_name + '/gt.mp3',
                                gt_audios[0].detach().cpu().unsqueeze(0), sample_rate)

    # Save the model
    torch.save(oscillator.state_dict(), dir_name + '/oscillator.pth')
    writer.close()
