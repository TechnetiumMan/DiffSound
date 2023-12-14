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
import nvdiffrast.torch as dr
import xatlas

import meshio
from tqdm import tqdm

import sys
sys.path.append("src/dmtet/")

# Import topology / geometry trainers
from geometry.dmtet_thickness import DMTetGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render
from geometry.sdf import train_sdfnerf
from src.diffelastic.diff_model import DiffSoundObj

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

# @torch.no_grad()
# def createLoss(FLAGS):
#     if FLAGS.loss == "smape":
#         return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
#     elif FLAGS.loss == "mse":
#         return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
#     elif FLAGS.loss == "logl1":
#         return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
#     elif FLAGS.loss == "logl2":
#         return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
#     elif FLAGS.loss == "relmse":
#         return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
#     else:
#         assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

# @torch.no_grad()
# def prepare_batch(target, bg_type='black'):
#     assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
#     if bg_type == 'checker':
#         background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
#     elif bg_type == 'black':
#         background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
#     elif bg_type == 'white':
#         background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
#     elif bg_type == 'reference':
#         background = target['img'][..., 0:3]
#     elif bg_type == 'random':
#         background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
#     else:
#         assert False, "Unknown background type %s" % bg_type

#     target['mv'] = target['mv'].cuda()
#     target['mvp'] = target['mvp'].cuda()
#     target['campos'] = target['campos'].cuda()
#     target['img'] = target['img'].cuda()
#     target['background'] = background

#     target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

#     return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

# @torch.no_grad()
# def xatlas_uvmap(glctx, geometry, mat, FLAGS):
#     eval_mesh = geometry.getMesh(mat)[0]
    
#     # Create uvs with xatlas
#     v_pos = eval_mesh.v_pos.detach().cpu().numpy()
#     t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
#     vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

#     # Convert to tensors
#     indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
#     uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
#     faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

#     new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

#     mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
#     if FLAGS.layers > 1:
#         kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

#     kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
#     ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
#     nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

#     new_mesh.material = material.Material({
#         'bsdf'   : mat['bsdf'],
#         'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
#         'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
#         'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
#     })

#     return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

# def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
#     kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
#     ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
#     nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
#     if mlp:
#         mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
#         mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
#         mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
#         mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
#     else:
#         # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
#         if FLAGS.random_textures or init_mat is None:
#             num_channels = 4 if FLAGS.layers > 1 else 3
#             kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
#             kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

#             ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
#             ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
#             ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

#             ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
#         else:
#             kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
#             ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

#         # Setup normal map
#         if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
#             normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
#         else:
#             normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

#         mat = material.Material({
#             'kd'     : kd_map_opt,
#             'ks'     : ks_map_opt,
#             'normal' : normal_map_opt
#         })

#     if init_mat is not None:
#         mat['bsdf'] = init_mat['bsdf']
#     else:
#         mat['bsdf'] = 'pbr'

#     return mat

###############################################################################
# Validation & testing
###############################################################################

# def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
#     result_dict = {}
#     with torch.no_grad():
#         lgt.build_mips()
#         if FLAGS.camera_space_light:
#             lgt.xfm(target['mv'])

#         buffers = geometry.render(glctx, target, lgt, opt_material)

#         result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
#         result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
#         result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

#         if FLAGS.display is not None:
#             white_bg = torch.ones_like(target['background'])
#             for layer in FLAGS.display:
#                 if 'latlong' in layer and layer['latlong']:
#                     if isinstance(lgt, light.EnvironmentLight):
#                         result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
#                     result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
#                 elif 'relight' in layer:
#                     if not isinstance(layer['relight'], light.EnvironmentLight):
#                         layer['relight'] = light.load_env(layer['relight'])
#                     img = geometry.render(glctx, target, layer['relight'], opt_material)
#                     result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
#                     result_image = torch.cat([result_image, result_dict['relight']], axis=1)
#                 elif 'bsdf' in layer:
#                     buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
#                     if layer['bsdf'] == 'kd':
#                         result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
#                     elif layer['bsdf'] == 'normal':
#                         result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
#                     else:
#                         result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
#                     result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
#         return result_image, result_dict

# def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

#     # ==============================================================================================
#     #  Validation loop
#     # ==============================================================================================
#     mse_values = []
#     psnr_values = []

#     dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

#     os.makedirs(out_dir, exist_ok=True)
#     with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
#         fout.write('ID, MSE, PSNR\n')

#         print("Running validation")
#         for it, target in enumerate(dataloader_validate):

#             # Mix validation background
#             target = prepare_batch(target, FLAGS.background)

#             result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
#             # Compute metrics
#             opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
#             ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

#             mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
#             mse_values.append(float(mse))
#             psnr = util.mse_to_psnr(mse)
#             psnr_values.append(float(psnr))

#             line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
#             fout.write(str(line))

#             for k in result_dict.keys():
#                 np_img = result_dict[k].detach().cpu().numpy()
#                 util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

#         avg_mse = np.mean(np.array(mse_values))
#         avg_psnr = np.mean(np.array(psnr_values))
#         line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
#         fout.write(str(line))
#         print("MSE,      PSNR")
#         print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
#     return avg_psnr

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
    # def lr_schedule(iter, fraction):
    #     if iter < warmup_iter:
    #         return iter / warmup_iter 
    #     return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    trainer = Trainer(geometry, FLAGS)
    # if FLAGS.isosurface == 'flexicubes':
    #     betas = (0.7, 0.9)
    # else:
    #     betas = (0.9, 0.999)

    optimizer_thickness = torch.optim.Adam(trainer.geo_params, lr=learning_rate_thickness)
    # optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
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
    FLAGS.mesh_scale          = 2.7                        # Scale of tet grid box. Adjust to cover the model
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
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device)

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    target_mesh         = mesh.load_mesh(FLAGS.target_mesh, FLAGS.mtl_override)
    
    # load tet mesh (not triangle!) for ground truth audio
    # target_tetmesh = meshio.read(FLAGS.target_mesh[:-4] + ".msh")
    
    # generate ground truth audio (now: eigenvalues of each mode)
    # vertices = torch.Tensor(target_tetmesh.points).cuda()
    # tets = torch.Tensor(target_tetmesh.cells[0].data).long().cuda()
    # print('Load tetramesh with ', len(vertices),
    #     ' vertices & ', len(tets), ' tets')
    
    # target_sound_obj = DiffSoundObj(vertices, tets, mode_num=FLAGS.mode_num, order=FLAGS.order)
    # target_sound_obj.eigen_decomposition()
    # target_vals = target_sound_obj.get_vals()
    target_geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
    target_geometry.apply_sdf(FLAGS.init_mesh_dir, FLAGS)
    target_vals = target_geometry.get_eigenvalues(thickness_coef=0.2)
    print("ground truth eigenvalues:", target_vals)
    # target_vals = 0 # debug

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)
    
    # if we have initial mesh for sdf nerf training, train it
    # if FLAGS.init_mesh: # we need init mesh!
    geometry.apply_sdf(FLAGS.init_mesh_dir, FLAGS)

    # Run optimization
    geometry = optimize_mesh(geometry, target_vals, FLAGS)

    final_mesh = geometry.getMesh(return_triangle=True)
    os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)

#----------------------------------------------------------------------------
