# Infer an object's thickness based on modal sound
# we generated meshes with different thicknesses, 
# for each mesh, we generate its modal eigenvalues as the target for the thickness inference

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time                                 
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append("src/dmtet/")

# Import topology / geometry trainers
from geometry.dmtet_thickness import DMTetGeometry

from render import obj
from render import util

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
        
        if it % log_interval == 0:
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    
    FLAGS = parser.parse_args()

    # for multi modal(audio) training
    FLAGS.mode_num = 32
    FLAGS.order = 1

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

    thickness_list = FLAGS.thickness_list # [0.3, 0.4, 0.5, 0.6, 0.7]

    # try to fit the thickness and save the result to file
    file = open(os.path.join(FLAGS.out_dir, f"result_{FLAGS.mesh_name}.txt"), "a+", encoding="utf-8")
    file.write(f"material:{FLAGS.mat}\n")
    total_error = 0
    for thickness in thickness_list:
        target_dir = os.path.join(FLAGS.target_mesh_dir, FLAGS.mesh_name, f"thickness{thickness}.obj")
        target_geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        target_geometry.apply_sdf(target_dir)
        target_vals = target_geometry.get_eigenvalues(thickness_coef=1.0)
        print("ground truth eigenvalues:", target_vals)

        # Setup geometry for optimization
        geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
        
        # use the initial solid mesh as the input mesh
        geometry.apply_sdf(FLAGS.init_mesh_dir + FLAGS.mesh_name + ".obj")

        # Run thickness optimization
        geometry = optimize_mesh(geometry, target_vals, FLAGS)

        final_mesh = geometry.getMesh(return_triangle=True)
        os.makedirs(os.path.join(FLAGS.out_dir, f"{FLAGS.mesh_name}"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, f"{FLAGS.mesh_name}/"), final_mesh, name=f"result{thickness}.obj")
        
        result_thickness = geometry.marching_tets.thickness_coef().item()
        total_error += (result_thickness - thickness) ** 2 / 5
        
        print(f"target:{thickness} result:{result_thickness}")
        file.write(f"target:{thickness} result:{result_thickness}\n")
        
    print(f"total error:{total_error}")
    file.write(f"total error:{total_error}\n")
    file.close()
        
#----------------------------------------------------------------------------
