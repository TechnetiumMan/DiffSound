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

from render import mesh
from render import render

import sys
sys.path.append("./")
from src.diffelastic.diff_model import DiffSoundObj

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1],
                [1, 0, 2, -1, -1, -1],
                [4, 0, 3, -1, -1, -1],
                [1, 4, 2, 1, 3, 4],
                [3, 1, 5, -1, -1, -1],
                [2, 3, 0, 2, 5, 3],
                [1, 4, 0, 1, 5, 4],
                [4, 2, 5, -1, -1, -1],
                [4, 5, 2, -1, -1, -1],
                [4, 1, 0, 4, 5, 1],
                [3, 2, 0, 3, 5, 2],
                [1, 3, 5, -1, -1, -1],
                [4, 1, 2, 4, 3, 1],
                [3, 0, 4, -1, -1, -1],
                [2, 0, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.long,
            device="cuda",
        )

        self.num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
            dtype=torch.long,
            device="cuda",
        )
        self.base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda"
        )

        self.num_tets_table = torch.tensor(
            [0, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1],
            dtype=torch.long,
            device="cuda",
        )

        self.tet_table = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # (0,0,0,0)
                [0, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1],  # (0,0,0,1)
                [1, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1],  # (0,0,1,0)
                [7, 1, 8, 6, 5, 1, 7, 6, 5, 0, 1, 6],  # (0,0,1,1)
                [2, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1],  # (0,1,0,0)
                [4, 0, 6, 7, 9, 0, 7, 6, 7, 0, 9, 2],  # (0,1,0,1)
                [4, 1, 9, 8, 5, 1, 9, 4, 5, 1, 2, 9],  # (0,1,1,0)
                [6, 0, 1, 2, 8, 6, 1, 2, 9, 6, 8, 2],  # (0,1,1,1)
                [3, 6, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1],  # (1,0,0,0)
                [5, 0, 4, 8, 5, 0, 8, 3, 5, 8, 9, 3],  # (1,0,0,1)
                [1, 4, 7, 3, 4, 7, 6, 3, 9, 6, 7, 3],  # (1,0,1,0)
                [0, 1, 5, 3, 5, 1, 9, 3, 5, 1, 7, 9],  # (1,0,1,1)
                [5, 2, 3, 7, 3, 6, 5, 8, 3, 5, 7, 8],  # (1,1,0,0)
                [0, 4, 7, 8, 0, 3, 8, 7, 0, 3, 7, 2],  # (1,1,0,1)
                [4, 1, 2, 3, 4, 3, 2, 5, 4, 3, 5, 6],  # (1,1,1,0)
                [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1],  # (1,1,1,1)
            ],
            dtype=torch.long,
            device="cuda",
        )

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            # occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)
        
        # 必须在这里去除碎片，以使得uv_idx和去除碎片后的mesh相统一！
        verts_connected, faces_connected = get_largest_connected_component_triangle(verts, faces)
        
        # 现在，我们必须更新valid tets, 只留下去除碎片后的tets
        # 首先找到原有顶点中哪些顶点在去除碎片后仍然存在
        verts_connect_matches = torch.all(torch.eq(verts.unsqueeze(0), verts_connected.unsqueeze(1)), dim=-1)
        verts_connect_idx = torch.where(verts_connect_matches)[1] # (n_verts_connected) in range (n_verts)
        
        # 找到含有剩余顶点的面
        faces_origin_idx = verts_connect_idx[faces_connected] # 是faces的真子集
        faces_connect_matches = torch.all(torch.eq(faces.unsqueeze(0), faces_origin_idx.unsqueeze(1)), dim=-1)
        faces_connect_idx = torch.where(faces_connect_matches)[1] # 每个剩余的面在faces中的下标

        # Get global face index (static, does not depend on topology)
        # 找到所有有效面，用其下标做uv map
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets] # 所有有效四面体在所有四面体中的全局编号
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0) # 由于每个四面体至多有两个有效面，所以直接用四面体的全局编号*2作为有效面的全局编号

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)
        
        # 为了使得uv map对应上去除碎片后的tet，需要更新uv_idx
        uv_idx_connected = uv_idx[faces_connect_idx]
        
        # generate tetmesh (the same of triangle mesh above)(verts and vert indices of each tet)
        num_tets_of_each_tet = self.num_tets_table[tetindex]
        valid_tets_vert_idx = tet_fx4[valid_tets]
        num_verts = pos_nx3.shape[0]
        
        idx_map += num_verts
        tet_verts_and_edges = torch.cat([valid_tets_vert_idx, idx_map], dim=1)
        
        side_tets = torch.cat((
                torch.gather(
                    input=tet_verts_and_edges[num_tets_of_each_tet == 1],
                    dim=1,
                    index=self.tet_table[tetindex[num_tets_of_each_tet == 1]][:, :4],
                ).reshape(-1, 4),
                torch.gather(
                    input=tet_verts_and_edges[num_tets_of_each_tet == 3],
                    dim=1,
                    index=self.tet_table[tetindex[num_tets_of_each_tet == 3]][:, :12],
                ).reshape(-1, 4),
            ),dim=0,
        )
        
        # add inner tets to tetmesh
        all_verts = torch.cat([pos_nx3, verts], dim=0)
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(
                -1, 4
            )
            occ_sum = torch.sum(occ_fx4, -1) 
            inner_tets_bool = occ_sum == 4 
            inner_tets = tet_fx4[inner_tets_bool] 

            all_tets = torch.cat([side_tets, inner_tets], dim=0) 
            
        # remove duplicate
        all_unique_tets, all_unique_verts_idx_map = torch.unique(
            all_tets.reshape(-1), return_inverse=True
        )
        all_tets_tetmesh = all_unique_verts_idx_map.reshape(-1, 4)
        all_verts_tetmesh = all_verts[all_unique_tets]

        return verts_connected, faces_connected, uvs, uv_idx_connected, all_verts_tetmesh, all_tets_tetmesh

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

class PositionalEncoding(torch.nn.Module):
    def __init__(self, freq_num=1, scale=1.0):
        super(PositionalEncoding, self).__init__()
        self.freq_num = freq_num
        self.freqs = [2**i for i in range(freq_num)]
        self.scale = scale

    def forward(self, x):
        x_in = x
        for freq in self.freqs:
            x = torch.cat(
                [
                    x,
                    torch.sin(freq * np.pi * x_in / self.scale),
                    torch.cos(freq * np.pi * x_in / self.scale),
                ],
                dim=-1,
            )
        return x


class NerfWithPositionEncoding(torch.nn.Module):
    def __init__(self, freq_num=1, scale=1.0, layer_num=3, hidden_dim=256):
        super(NerfWithPositionEncoding, self).__init__()
        self.freq_num = freq_num
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.pos_enc = PositionalEncoding(freq_num, scale=scale)
        self.layer_0 = torch.nn.Linear(6 * freq_num + 3, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_num)]
        )
        self.final_layer = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.functional.relu

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.layer_0(x)
        x = self.activation(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x

###############################################################################
#  Geometry interface
###############################################################################

import scipy.sparse as sp

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.scale = scale
        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
        
        # self.sdf_regularizer = 0.02

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.base_verts = torch.tensor(
            tets["vertices"], dtype=torch.float32, device="cuda"
        )
        self.verts = self.base_verts * self.scale
        # self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        
        # self.sdf_nerf = NerfWithPositionEncoding(
        #     freq_num=FLAGS.freq_num, scale=scale, layer_num=3, hidden_dim=256
        # ).cuda()
        
        # 将sdf network分为正面和背面，正面只用图像，背面加入声音
        # z > 0为正面
        self.z_threshold = -0.2
        self.sdf_nerf_front = NerfWithPositionEncoding(
            freq_num=FLAGS.freq_num, scale=scale, layer_num=3, hidden_dim=256
        ).cuda()
        
        self.sdf_nerf_back = NerfWithPositionEncoding(
            freq_num=FLAGS.freq_num, scale=scale, layer_num=3, hidden_dim=256
        ).cuda()

        # Random init
        # sdf = torch.rand_like(self.verts[:,0]) - 0.1
        # self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        # self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)
        
    def mesh_template_loss(self, nodes, signed_distance, margin):
        sdf = self.sdf_nerf(nodes[signed_distance > margin]) # only input non-margin inside query nodes to NeRF, and output sdf(need to >0)
        loss = 0
        return_none = True
        if len(sdf[sdf <= margin]) > 0: # if nerf gives a inside point <0 sdf, give it a loss
            loss += -(sdf[sdf <= margin]).mean()
            return_none = False
        sdf = self.sdf_nerf(nodes[signed_distance < -margin])
        if len(sdf[sdf >= -margin]) > 0:
            loss += (sdf[sdf >= -margin]).mean()
            return_none = False
        if return_none:
            return None
        return loss

    @property
    def sdf(self):
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        # print(v_deformed.min(), v_deformed.max())
        # print(self.verts.min(), self.verts.max())
        
        # 将所有顶点分成正面和背面
        sdf_front = self.sdf_nerf_front(v_deformed[v_deformed[:,2] >= self.z_threshold]).squeeze(-1)
        sdf_back = self.sdf_nerf_back(v_deformed[v_deformed[:,2] < self.z_threshold]).squeeze(-1)
        
        # 将其合并为一个sdf
        sdf = torch.zeros_like(v_deformed[:,0])
        sdf[v_deformed[:,2] >= self.z_threshold] = sdf_front
        sdf[v_deformed[:,2] < self.z_threshold] = sdf_back
        
        # sdf = self.sdf_nerf(v_deformed)
        return sdf
    
    def sdf_nerf(self, verts):
        # 将所有顶点分成正面和背面
        sdf_front = self.sdf_nerf_front(verts[verts[:,2] >= self.z_threshold]).squeeze(-1)
        sdf_back = self.sdf_nerf_back(verts[verts[:,2] < self.z_threshold]).squeeze(-1)
        
        # 将其合并为一个sdf
        sdf = torch.zeros_like(verts[:,0])
        sdf[verts[:,2] >= self.z_threshold] = sdf_front
        sdf[verts[:,2] < self.z_threshold] = sdf_back
        return sdf
    
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        verts, faces, uvs, uv_idx, verts_tetmesh, tets_tetmesh = self.marching_tets(v_deformed, self.sdf, self.indices)
        
        # build triangle mesh
        if material:
            imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

            # Run mesh operations to generate tangent space
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
            
            # build tetmesh
            verts_tetmesh, tets_tetmesh = self.get_largest_connected_component(verts_tetmesh, tets_tetmesh)
            sound_obj = DiffSoundObj(verts_tetmesh, tets_tetmesh, mode_num=self.FLAGS.mode_num, order=self.FLAGS.order)

            return imesh, sound_obj
        else:
            return verts_tetmesh, tets_tetmesh
        
    def get_largest_connected_component(self, verts, tets):
        rows = torch.cat([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
        cols = torch.cat([tets[:, 1], tets[:, 2], tets[:, 3], tets[:, 0]])
        data = torch.ones_like(rows, dtype=torch.float32)
        A = sp.coo_matrix(
            (data.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
            shape=(len(verts), len(verts)),
        ).tocsr()
        n_components, labels = sp.csgraph.connected_components(A, directed=False)
        # print("n_components: {}".format(n_components))
        if n_components == 1:
            return verts, tets

        # Find the largest component
        largest_idx = 0
        largest_size = 0
        for i in range(n_components):
            num = (labels == i).sum()
            if num > largest_size:
                largest_size = num
                largest_idx = i
        labels = torch.tensor(labels, dtype=torch.long, device="cuda")
        verts_subset = verts[labels == largest_idx]
        # compute the map from old indices to new indices
        new_indices = torch.zeros_like(labels) - 1
        new_indices[labels == largest_idx] = torch.arange(
            largest_size, dtype=torch.long, device="cuda"
        )
        tets = new_indices[tets]
        # remove tets with -1
        valid_tets = (tets >= 0).all(dim=1)
        return verts_subset, tets[valid_tets]

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return render.render_mesh(glctx, opt_mesh[0], target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                        msaa=True, background=target['background'], bsdf=bsdf)


    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, FLAGS):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        # buffers = self.render(glctx, target, lgt, opt_material)
        
        mesh, sound_obj = self.getMesh(opt_material)
        buffers = render.render_mesh(glctx, mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                        msaa=True, background=target['background'])

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005
        
        # add audio loss
        if FLAGS.audio_weight > 0:
            sound_obj.eigen_decomposition()
            vals = sound_obj.get_vals()
            # print(vals[0,0])
            audio_loss = ((vals - target["eigenvalue"]) ** 2 / vals**2).mean()
        else:
            audio_loss = torch.tensor([0.]).cuda()
        return img_loss, reg_loss, audio_loss

def get_largest_connected_component_triangle(verts, triangles):
    rows = torch.cat([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    cols = torch.cat([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    data = torch.ones_like(rows, dtype=torch.float32)
    A = sp.coo_matrix(
        (data.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
        shape=(len(verts), len(verts)),
    ).tocsr()
    n_components, labels = sp.csgraph.connected_components(A, directed=False)
    # print("n_components: {}".format(n_components))
    if n_components == 1:
        return verts, triangles

    # Find the largest component
    largest_idx = 0
    largest_size = 0
    for i in range(n_components):
        num = (labels == i).sum()
        if num > largest_size:
            largest_size = num
            largest_idx = i
    labels = torch.tensor(labels, dtype=torch.long, device="cuda")
    verts_subset = verts[labels == largest_idx]
    # compute the map from old indices to new indices
    new_indices = torch.zeros_like(labels) - 1
    new_indices[labels == largest_idx] = torch.arange(
        largest_size, dtype=torch.long, device="cuda"
    )
    triangles = new_indices[triangles]
    # remove tets with -1
    valid_triangles = (triangles >= 0).all(dim=1)
    return verts_subset, triangles[valid_triangles]
