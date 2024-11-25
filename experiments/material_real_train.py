# material parameter inference using real audio data
# we load real-recorded audio data, and optimize the material parameters to make the generated modal audio to fit the real audio
# we train oscillator to get damping curve at first, then optimize material model to infer material parameters
# we have 4 mode of experiment: 
# 0: ord=1,no poisson (baseline); 1:ord=2, no poisson; 2:ord=1, learnable poisson; 3:ord=2, learnable poisson (our DiffSound)

import sys
sys.path.append('./')
import torch
import os
import argparse
import json
from datetime import datetime
from src.diffelastic.diff_model import Material, build_model, MatSet
import torchaudio
from src.utils.utils import plot_spec, resample
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import TraditionalDampedOscillator, GTDampedOscillator, DampedOscillator, init_damps
from torchaudio.functional import highpass_biquad
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')

    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]
            
    print("Config / Flags:")
    print("---------")
    for key in FLAGS.__dict__.keys():
        print(key, FLAGS.__dict__[key])
    print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    exp_mode = FLAGS.exp_mode # 0: ord=1,no poisson; 1:ord=2, no poisson; 2:ord=1, learnable poisson; 3:ord=2, learnable poisson
    if exp_mode == 0 or exp_mode == 2:
        mesh_order = 1
    else:
        mesh_order = 2
    if exp_mode == 0 or exp_mode == 1:
        task = "mat_baseline"
    else:
        task = "material"

    dir_name = FLAGS.out_dir + FLAGS.mesh_name + str(FLAGS.exp_mode) + '_' + datetime.now().strftime("%b%d_%H-%M-%S")

    # Create the TensorBoard summary writer
    writer = SummaryWriter(dir_name+"/train")
    writer_gt = SummaryWriter(dir_name+"/gt")

    # Read the configuration parameters
    sample_rate = FLAGS.sample_rate
    max_epoch = FLAGS.max_epoch
    mesh_dir = FLAGS.mesh_dir
    eigen_num = FLAGS.mode_num
    early_loss_epoch = FLAGS.early_loss_epoch
    log_range_step = FLAGS.log_range_step
    frame_num = FLAGS.frame_num
    force_frame_num = FLAGS.force_frame_num

    material_coeff = getattr(MatSet, FLAGS.material) # initial material parameters
    print("init youngs:", material_coeff[1], "poisson:", material_coeff[2])

    # generate ground truth sync audio based on specific material parameters
    gt_forces = torch.zeros((1, force_frame_num)).cuda()
    gt_forces[0, 0] = 1 # impulse
    
    # load ground truth audios
    audio_dir = FLAGS.audio_dir
    audios = []
    subdirs = glob(audio_dir + "/*")
    for filename in subdirs:
        if "metadata" in filename:
            f = open(filename)
            yaml_data = yaml.safe_load(f)
            gain = yaml_data.get("gain")
            pad = yaml_data.get("pad")
    for filename in subdirs:
        if "mic" in filename:
            audio, sr = torchaudio.load(filename)
            audio = torchaudio.functional.gain(audio, gain[1])
            audio = audio[:, pad[1] * sr:]
            audios.append(audio)
    audios = audios[:FLAGS.audio_num]
    gt_audios = []
    for audio in audios:
        audio0 = resample(audio.cuda(), sr, sample_rate).squeeze(0)[:frame_num]
        audio0 = highpass_biquad(audio0, sample_rate, 100)
        
        # normalize audio
        audio0 = audio0 / torch.max(torch.abs(audio0))
        gt_audios.append(audio0)
    gt_audios = torch.stack(gt_audios)
    gt_forces = gt_forces.repeat(len(gt_audios), 1)
    
    # first training damping curve
    early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
    late_loss_func = MSSLoss([512, 256, 128, 64, 32], sample_rate, type='l1_loss').cuda()
    log_spec_funcs = [late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]
    
    pre_osc = GTDampedOscillator(gt_forces, len(
    gt_audios), eigen_num * 16, frame_num, sample_rate, [20, 16000], Material(material_coeff)).cuda()
    optimizer_pre_osc = Adam(pre_osc.parameters(), lr=5e-3)
    scheduler_pre_osc = lr_scheduler.StepLR(optimizer_pre_osc, step_size=100, gamma=0.99)
    for epoch_i in tqdm(range(2001)):
        predict_signal = pre_osc(noise_rate=2e-4)
        loss = late_loss_func(predict_signal, gt_audios)
        optimizer_pre_osc.zero_grad()
        loss.backward()
        optimizer_pre_osc.step()
        scheduler_pre_osc.step()
        writer.add_scalar('pre_osc_loss', loss.item(), epoch_i)
        if epoch_i % 1000 == 0:
            for i in range(0, len(gt_forces), log_range_step):
                for j in range(len(log_spec_funcs)):
                    writer.add_figure(
                        'pre_osc_{}th_{}th'.format(i, j),
                        plot_spec(log_spec_funcs[j](gt_audios[i]),
                                    log_spec_funcs[j](predict_signal[i])),
                        epoch_i)

    damping = pre_osc.damping()
    freq_linear = pre_osc.freq_linear()
    mask = damping < 300
    damping = damping[mask]
    freq_linear = freq_linear[mask]
    x = []
    y = []
    freq_step = 500
    for i in range(20, 20000, freq_step):
        mask = (freq_linear > i) & (freq_linear < i + freq_step)
        damping_ = damping[mask]
        if damping_.shape[0] == 0:
            continue
        x.append(i + freq_step // 2)
        y.append(damping_.min().item())

    from scipy import interpolate
    damping_curve = interpolate.interp1d(x, y, fill_value="extrapolate")
    
    
    # then build our material model and optimize it
    model = build_model(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task)

    oscillator = DampedOscillator(gt_forces, len(
        gt_audios), eigen_num, frame_num, sample_rate, f_range=[20, 16000], mat=Material(material_coeff)).cuda()
    init_damps(oscillator)

    # Create the MSSLoss object
    early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
    late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda()
    rmse_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='rmse_loss').cuda()
    log_spec_funcs = [
        late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]

    # Create the optimizer and scheduler
    optimizer_model = Adam(model.parameters(), lr=1e-3)
    scheduler_model = lr_scheduler.StepLR( 
        optimizer_model, step_size=100, gamma=0.9)

    # for faster speed, we do eigen decomposition every 15 epochs
    EIGEN_DECOMPOSE_CYCLE = 15

    for epoch_i in tqdm(range(max_epoch)):
        # change loss func and optimizer for epoch
        if epoch_i % EIGEN_DECOMPOSE_CYCLE == 0:
            model.eigen_decomposition()
        undamped_freq = model.get_undamped_freqs().float()

        if epoch_i < early_loss_epoch:
            loss_func = early_loss_func
        else:
            loss_func = late_loss_func
        predict_signal = oscillator.forward_curve(undamped_freq, damping_curve)
        # predict_signal = predict_signal / torch.max(torch.abs(predict_signal))
        
        
        if epoch_i == early_loss_epoch: # reset optimizer
            optimizer_model = Adam(model.parameters(), lr=2e-3)
            scheduler_model = lr_scheduler.StepLR( 
                optimizer_model, step_size=100, gamma=0.95)
        
        damped_freq = oscillator.damped_freq

        spec_scale = 1
        loss = loss_func(predict_signal, gt_audios, damped_freq, spec_scale)
        if epoch_i < early_loss_epoch:
            writer.add_scalar('loss_early', loss.item(), epoch_i)
        else:
            writer.add_scalar('loss', loss.item(), epoch_i)
            
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        scheduler_model.step()

        if epoch_i % (EIGEN_DECOMPOSE_CYCLE) == 0:
            with torch.no_grad():
                print('loss_model: ', loss.item())
                print('undamped f:', undamped_freq)
                
                # get RMSE loss
                RMSE_loss = rmse_loss_func(predict_signal, gt_audios)
                print('RMSE loss:', RMSE_loss.item())
                writer.add_scalar("RMSE", RMSE_loss.item(), epoch_i)
                print('youngs module:', model.material_model.youngs().item() * 2700)
                print('poisson ratio:', model.material_model.poisson().item())
                # print("target youngs:", gt_material_coeff[1], "poisson:", gt_material_coeff[2])
                writer.add_scalar("youngs", model.material_model.youngs().item() * 2700, epoch_i)
                writer.add_scalar("poisson", model.material_model.poisson().item(), epoch_i)
                # writer_gt.add_scalar("youngs", gt_material_coeff[1], epoch_i)
                # writer_gt.add_scalar("poisson", gt_material_coeff[2], epoch_i)
                    
                for audio_idx in range(0, len(predict_signal), log_range_step):
                    for spec_idx in range(len(log_spec_funcs)):
                        writer.add_figure(
                            '{}th_{}'.format(
                                audio_idx, late_loss_func.n_ffts[spec_idx]),
                            plot_spec(log_spec_funcs[spec_idx](gt_audios[audio_idx], spec_scale),
                                    log_spec_funcs[spec_idx](predict_signal[audio_idx], spec_scale),
                            ),
                            epoch_i)
                        
                gt_audios_save = gt_audios / torch.max(torch.abs(gt_audios))
                predict_signal_save = predict_signal / torch.max(torch.abs(predict_signal))
                torchaudio.save(dir_name + '/predict.mp3',
                                predict_signal_save[0].detach().cpu().unsqueeze(0), sample_rate)
                torchaudio.save(dir_name + '/gt.mp3',
                                gt_audios_save[0].detach().cpu().unsqueeze(0), sample_rate)
        if epoch_i % (EIGEN_DECOMPOSE_CYCLE*100) == 0:
            torch.save(model.material_model.state_dict(), dir_name + '/model.pth')
            
    # save the result
    save_name = dir_name + "/result.txt"
    file = open(save_name, 'a+')
    # file.write("material:" + str(mat_num) + "\n")
    file.write("youngs:" + str(model.material_model.youngs().item() * 2700) + "\n") # youngs
    file.write("poisson:" + str(model.material_model.poisson().item()) + "\n") # poisson
    # file.write("target youngs:" + str(gt_material_coeff[1]) + "\n") # target youngs
    # file.write("target poisson:" + str(gt_material_coeff[2]) + "\n") # target poisson
    file.write("RMSE:" + str(RMSE_loss.item()) + "\n") # RMSE loss
    file.close()
