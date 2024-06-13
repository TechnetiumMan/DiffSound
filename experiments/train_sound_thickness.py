# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

# this file is for training an object's thickness (it is hollow or not) based on its modal sound(not image at all!)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import time                                 
import argparse
import json

import numpy as np
import torch
# import nvdiffrast.torch as dr
# import xatlas

# import meshio
from tqdm import tqdm

import sys
sys.path.append("src/dmtet/")

# Import topology / geometry trainers
from geometry.dmtet_thickness import DMTetGeometry

# import render.renderutils as ru
from render import obj
# from render import material
from render import util
# from render import mesh
# from render import texture
# from render import mlptexture
# from render import light
# from render import render
# from geometry.sdf import train_sdfnerf
from src.diffelastic.diff_model import DiffSoundObj
from torch.utils.tensorboard import SummaryWriter

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)
###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, geometry, FLAGS):
        super(Trainer, self).__init__()
        self.geometry = geometry
        self.FLAGS = FLAGS
        self.geo_params = list(self.geometry.parameters()) 

    def forward(self, target, it):
        return self.geometry.tick(target, it, FLAGS)

def optimize_mesh(
    geometry,
    target,
    FLAGS,
    # warmup_iter=0,
    log_interval=10,  
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================
    learning_rate_thickness = FLAGS.learning_rate
    trainer = Trainer(geometry, FLAGS)
    optimizer_thickness = torch.optim.Adam(trainer.geo_params, lr=learning_rate_thickness)

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    audio_loss_vec = []
    iter_dur_vec = []
    
    # for it, target in enumerate(dataloader_train):
    for it in tqdm(range(FLAGS.iter)):
        iter_start_time = time.time()

        optimizer_thickness.zero_grad()

        audio_loss = trainer(target, it)
        audio_loss_vec.append(audio_loss.item())
        audio_loss.backward()
        
        optimizer_thickness.step()

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            audio_loss_avg = np.mean(np.asarray(audio_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg
            print("iter=%5d, audio_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, audio_loss_avg, optimizer_thickness.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))

    return geometry

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--isosurface', default='dmtet', choices=['dmtet', 'flexicubes'])
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 32                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.5                        # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = False
    FLAGS.local_rank = 0
    
    # for multi model(audio) training
    FLAGS.mode_num = 32
    FLAGS.order = 1
    # FLAGS.audio_weight = 0.1
    FLAGS.fixed_obs = True
    FLAGS.obs_range = 0.2
    FLAGS.freq_num = 3
    
    # for nerf sdf training
    FLAGS.init_mesh = True

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # first, generate meshes with certain thickness(0.3, 0.4, 0.5, 0.6, 0.7)
    thickness_list = [0.3,0.4,0.5,0.6,0.7]
    for thickness in thickness_list:
        init_geometry = DMTetGeometry(128, FLAGS.mesh_scale, FLAGS)
        init_geometry.apply_sdf(FLAGS.init_mesh_dir, FLAGS)
        init_triangle_mesh = init_geometry.getMesh(return_triangle=True, thickness_coef=thickness)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), init_triangle_mesh, name=f"thickness{thickness}.obj")
    
    # then try to fit the thickness and save the result to file
    file = open(os.path.join(FLAGS.out_dir, "result.txt"), "a+", encoding="utf-8")
    file.write(f"material:{FLAGS.mat}\n")
    total_error = 0
    for thickness in thickness_list:
        target_dir = os.path.join(FLAGS.out_dir, f"mesh/thickness{thickness}.obj")
        target_geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        target_geometry.apply_sdf(target_dir, FLAGS)
        target_vals = target_geometry.get_eigenvalues(thickness_coef=1.0)
        print("ground truth eigenvalues:", target_vals)

        # Setup geometry for optimization
        geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        
        # if we have initial mesh for sdf nerf training, train it
        geometry.apply_sdf(FLAGS.init_mesh_dir, FLAGS)

        # Run optimization
        geometry = optimize_mesh(geometry, target_vals, FLAGS)

        final_mesh = geometry.getMesh(return_triangle=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh, name=f"result{thickness}.obj")
        
        result_thickness = geometry.marching_tets.thickness_coef().item()
        total_error += (result_thickness - thickness) ** 2 / 5
        
        print(f"target:{thickness} result:{result_thickness}")
        file.write(f"target:{thickness} result:{result_thickness}\n")
    print(f"total error:{total_error}")
    file.write(f"total error:{total_error}\n")
    file.close()
        

#----------------------------------------------------------------------------
