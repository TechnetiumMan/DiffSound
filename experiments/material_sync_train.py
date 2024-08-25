# material parameter inference in synthetic material data
# we use 16 sets of random material parameters (Young's modulus and Poisson's ratio) to generate ground truth audio
# for each set of ground truth material parameters, we optimize the trainable material parameters to make its audio to fit the ground truth audio
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
from src.utils.utils import plot_spec
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from src.ddsp.mss_loss import MSSLoss
from src.ddsp.oscillator import TraditionalDampedOscillator
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)

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
        
    # generate the random material init and target coeff in range
    min_material_coeff = getattr(MatSet, "RandomMin")
    max_material_coeff = getattr(MatSet, "RandomMax")
    min_youngs = min_material_coeff[1]
    max_youngs = max_material_coeff[1]
    min_poisson = min_material_coeff[2]
    max_poisson = max_material_coeff[2]
    init_youngs = torch.rand(16) * (max_youngs - min_youngs) + min_youngs
    init_poisson = torch.rand(16) * (max_poisson - min_poisson) + min_poisson
    target_youngs = torch.rand(16) * (max_youngs - min_youngs) + min_youngs
    target_poisson = torch.rand(16) * (max_poisson - min_poisson) + min_poisson
    print(init_youngs, init_poisson, target_youngs, target_poisson)
    random_mats_init = torch.tensor(min_material_coeff).repeat(16, 1) 
    random_mats_init[:, 1] = init_youngs
    random_mats_init[:, 2] = init_poisson
    random_mats_target = torch.tensor(min_material_coeff).repeat(16, 1)
    random_mats_target[:, 1] = target_youngs
    random_mats_target[:, 2] = target_poisson
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    torch.save((random_mats_init, random_mats_target), dir_name + "/random_material.pth")

    # we use 16 sets of random material parameters
    init_mats, target_mats = torch.load(dir_name +"/random_material.pth")
    for mat_num in range(16):
        print("mat_num:", mat_num)
        gt_material_coeff = target_mats[mat_num].tolist()
        material_coeff = init_mats[mat_num].tolist()
        print("target youngs:", gt_material_coeff[1], "poisson:", gt_material_coeff[2])
        print("init youngs:", material_coeff[1], "poisson:", material_coeff[2])

        # generate ground truth sync audio based on specific material parameters
        gt_forces = torch.zeros((1, force_frame_num)).cuda()
        gt_forces[0, 0] = 1 # impulse

        gt_osc = TraditionalDampedOscillator(gt_forces, 1, eigen_num, frame_num, \
            sample_rate, Material(gt_material_coeff)).cuda()

        gt_model = build_model(mesh_dir, mode_num=eigen_num, order=2, mat=gt_material_coeff, task="gt") 
        gt_model.eigen_decomposition()

        gt_undamped_freq = gt_model.get_undamped_freqs().float()
        print("gt undamped f:", gt_undamped_freq)
        gt_audios = gt_osc(gt_undamped_freq)
        # gt_audios = gt_audios / torch.max(torch.abs(gt_audios))

        # then build our material model and optimize it
        model = build_model(mesh_dir, mode_num=eigen_num, order=mesh_order, mat=material_coeff, task=task)
        oscillator = TraditionalDampedOscillator(gt_forces, len(
            gt_audios), eigen_num, frame_num, sample_rate, Material(material_coeff)).cuda()

        # Create the MSSLoss object
        early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
        late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda()
        rmse_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='rmse_loss').cuda()
        log_spec_funcs = [
            late_loss_func.losses[i].log_spec for i in range(len(late_loss_func.losses))]

        # Create the optimizer and scheduler
        optimizer_model = Adam(model.parameters(), lr=5e-3)
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
            predict_signal = oscillator(undamped_freq)
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
                    print("target youngs:", gt_material_coeff[1], "poisson:", gt_material_coeff[2])
                    writer.add_scalar("youngs", model.material_model.youngs().item() * 2700, epoch_i)
                    writer.add_scalar("poisson", model.material_model.poisson().item(), epoch_i)
                    writer_gt.add_scalar("youngs", gt_material_coeff[1], epoch_i)
                    writer_gt.add_scalar("poisson", gt_material_coeff[2], epoch_i)
                        
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
        file.write("material:" + str(mat_num) + "\n")
        file.write("youngs:" + str(model.material_model.youngs().item() * 2700) + "\n") # youngs
        file.write("poisson:" + str(model.material_model.poisson().item()) + "\n") # poisson
        file.write("target youngs:" + str(gt_material_coeff[1]) + "\n") # target youngs
        file.write("target poisson:" + str(gt_material_coeff[2]) + "\n") # target poisson
        file.write("RMSE:" + str(RMSE_loss.item()) + "\n") # RMSE loss
        file.close()
