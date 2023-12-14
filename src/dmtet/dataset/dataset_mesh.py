# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
import sys
sys.path.append("./")

from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset
from src.diffelastic.diff_model import DiffSoundObj

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetMesh(Dataset):

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False, fixed_obs=False, ref_tetmesh=None):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
        
        self.fixed_obs = fixed_obs
        self.obs_range = FLAGS.obs_range

        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        
        self.ref_mesh = mesh.compute_tangents(ref_mesh)
        
        # generate ground truth audio (now: eigenvalues of each mode)
        if ref_tetmesh and FLAGS.audio_weight > 0:
            vertices = torch.Tensor(ref_tetmesh.points).cuda()
            tets = torch.Tensor(ref_tetmesh.cells[0].data).long().cuda()
            print('Load tetramesh with ', len(vertices),
                ' vertices & ', len(tets), ' tets')
            
            sound_obj = DiffSoundObj(vertices, tets, mode_num=self.FLAGS.mode_num, order=self.FLAGS.order)
            sound_obj.eigen_decomposition()
            self.vals = sound_obj.get_vals()
            print("ground truth eigenvalues:", self.vals)
        else:
            self.vals = 0


    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 10) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension
    
    # for audio optimization, fix observation angle and use audio to infer the unseen part 
    def _fixed_scene(self):
        
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        mv     = util.translate(0, 0, -self.cam_radius)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension
    
    # completely fix angle is too difficult, just give it a little rotate
    def _small_rotate_scene(self):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # random rotate for a small range
        ang_xy    = (torch.rand(2) - 0.5) * self.obs_range * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(ang_xy[0]) @ util.rotate_y(ang_xy[1])) # camera pos (0, 0, -3), z < 0 is front
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp
    
    # only for validation of sound
    def _valid_scene(self, itr):
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        
        if itr % 2 == 0: # valid rotate
            ang    = (itr / 8) * np.pi * 2
            mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(ang))
        else: # the same as train small rotate (for debug)
            ang_xy    = (torch.rand(2) - 0.5) * self.obs_range * np.pi * 2
            mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(ang_xy[0]) @ util.rotate_y(ang_xy[1]))
            
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            if self.fixed_obs:
                mv, mvp, campos, iter_res, iter_spp = self._valid_scene(itr)
            else:
                mv, mvp, campos, iter_res, iter_spp = self._rotate_scene(itr)
        else:
            if self.fixed_obs:
                mv, mvp, campos, iter_res, iter_spp = self._small_rotate_scene()
            else:
                mv, mvp, campos, iter_res, iter_spp = self._random_scene()

        img = render.render_mesh(self.glctx, self.ref_mesh, mvp, campos, self.envlight, iter_res, spp=iter_spp, 
                                num_layers=self.FLAGS.layers, msaa=True, background=None)['shaded']

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img,
            'eigenvalue' : self.vals
        }
