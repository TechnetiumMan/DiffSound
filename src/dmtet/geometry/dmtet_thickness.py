# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

from typing import Iterator
import numpy as np
import torch
from torch.nn.parameter import Parameter

from render import mesh
from render import render

import sys
sys.path.append("./")
from src.diffelastic.diff_model import DiffSoundObj
import open3d as o3d
from src.dmtet.geometry.sdf import WeightedParam

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
        
        # thickness coef in (0, 1) -> (0, max(self.sdf))
        self.thickness_list = torch.linspace(0, 1, steps=32)
        self.thickness_coef = WeightedParam(self.thickness_list)
        

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

    # def map_uv(self, faces, face_gidx, max_idx):
    #     N = int(np.ceil(np.sqrt((max_idx+1)//2)))
    #     tex_y, tex_x = torch.meshgrid(
    #         torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
    #         torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
    #         indexing='ij'
    #     )

    #     pad = 0.9 / N

    #     uvs = torch.stack([
    #         tex_x      , tex_y,
    #         tex_x + pad, tex_y,
    #         tex_x + pad, tex_y + pad,
    #         tex_x      , tex_y + pad
    #     ], dim=-1).view(-1, 2)

    #     def _idx(tet_idx, N):
    #         x = tet_idx % N
    #         y = torch.div(tet_idx, N, rounding_mode='trunc')
    #         return y * N + x

    #     tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
    #     tri_idx = face_gidx % 2

    #     uv_idx = torch.stack((
    #         tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
    #     ), dim = -1). view(-1, 3)

    #     return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4, thickness_coef=None):
        
        if thickness_coef is None:
            thickness = self.thickness_coef() * self.max_thickness # (0, 1) -> (0, max(self.sdf))
        else:
            thickness = thickness_coef * self.max_thickness
        
        with torch.no_grad():
            # add thickness here for hollow objects
            # occ_n = (sdf_n > 0 and sdf_n < thickness)
            occ_n = (sdf_n > 0) & (sdf_n < thickness)
            
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
        
        
        # 现在edges_to_interp_sdf有两种情况：第一种 x < 0, 0 < y < thickness, 第二种 0 < x < thickness, y > thickness
        # 只需要对所有x>0且y>0的两条边的sdf值同时减去thickness即可
        edges_to_interp_sdf[(edges_to_interp_sdf[:,0,0] > 0) & (edges_to_interp_sdf[:,1,0] > 0)] -= thickness
        
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
        
        # # 必须在这里去除碎片，以使得uv_idx和去除碎片后的mesh相统一！
        # verts_connected, faces_connected = get_largest_connected_component_triangle(verts, faces)
        
        # # 现在，我们必须更新valid tets, 只留下去除碎片后的tets
        # # 首先找到原有顶点中哪些顶点在去除碎片后仍然存在
        # verts_connect_matches = torch.all(torch.eq(verts.unsqueeze(0), verts_connected.unsqueeze(1)), dim=-1)
        # verts_connect_idx = torch.where(verts_connect_matches)[1] # (n_verts_connected) in range (n_verts)
        
        # # 找到含有剩余顶点的面
        # faces_origin_idx = verts_connect_idx[faces_connected] # 是faces的真子集
        # faces_connect_matches = torch.all(torch.eq(faces.unsqueeze(0), faces_origin_idx.unsqueeze(1)), dim=-1)
        # faces_connect_idx = torch.where(faces_connect_matches)[1] # 每个剩余的面在faces中的下标

        # Get global face index (static, does not depend on topology)
        # 找到所有有效面，用其下标做uv map
        # num_tets = tet_fx4.shape[0]
        # tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets] # 所有有效四面体在所有四面体中的全局编号
        # face_gidx = torch.cat((
        #     tet_gidx[num_triangles == 1]*2,
        #     torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        # ), dim=0) # 由于每个四面体至多有两个有效面，所以直接用四面体的全局编号*2作为有效面的全局编号

        # uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)
        
        # # 为了使得uv map对应上去除碎片后的tet，需要更新uv_idx
        # uv_idx_connected = uv_idx[faces_connect_idx]
        
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

        # return verts_connected, faces_connected, uvs, uv_idx_connected, all_verts_tetmesh, all_tets_tetmesh
        return verts, faces, all_verts_tetmesh, all_tets_tetmesh

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
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        
        # init sdf for loading init mesh
        self.sdf = torch.zeros_like(self.verts[:,0])
        
        # self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        # self.register_parameter('sdf', self.sdf)

        # self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        # self.register_parameter('deform', self.deform)
    
    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getMesh(self, return_triangle=False, thickness_coef=None):
        # Run DM tet to get a base mesh
        # v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        verts, faces, verts_tetmesh, tets_tetmesh = self.marching_tets(self.verts, self.sdf, self.indices, thickness_coef)
        
        # build triangle mesh
        # if material:
            # imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

            # # Run mesh operations to generate tangent space
            # imesh = mesh.auto_normals(imesh)
            # imesh = mesh.compute_tangents(imesh)
            
            # build tetmesh
        #     verts_tetmesh, tets_tetmesh = self.get_largest_connected_component(verts_tetmesh, tets_tetmesh)
        #     sound_obj = DiffSoundObj(verts_tetmesh, tets_tetmesh, mode_num=self.FLAGS.mode_num, order=self.FLAGS.order)

        #     return imesh, sound_obj
        if return_triangle:
            imesh = mesh.Mesh(verts, faces)
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
            return imesh
        else:
            verts_tetmesh, tets_tetmesh = self.get_largest_connected_component(verts_tetmesh, tets_tetmesh)
            sound_obj = DiffSoundObj(verts_tetmesh, tets_tetmesh, mode_num=self.FLAGS.mode_num, order=self.FLAGS.order)
            return sound_obj
        
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

    def tick(self, target, iteration, FLAGS):
        sound_obj = self.getMesh()

        # add audio loss
        sound_obj.eigen_decomposition()
        vals = sound_obj.get_vals()
        audio_loss = ((vals - target) ** 2 / vals**2).mean()
        
        print("thickness",self.marching_tets.thickness_coef().item(), "audio_loss", audio_loss.item())
        
        return audio_loss
    
    def apply_sdf(self, init_mesh_dir, FLAGS):
        mesh = o3d.io.read_triangle_mesh(init_mesh_dir)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        
        query_points = self.verts.cpu().numpy() # (-1.2, 1.2)
        signed_distance = scene.compute_signed_distance(query_points)
        signed_distance = signed_distance.numpy()
        signed_distance = -torch.from_numpy(signed_distance).cuda().reshape(-1)
        self.sdf = signed_distance
        
        self.marching_tets.max_thickness = self.sdf.max()
        
    def parameters(self):
        return self.marching_tets.thickness_coef.parameters()
    
    def get_eigenvalues(self, thickness_coef=None):
        with torch.no_grad():
            sound_obj = self.getMesh(thickness_coef=thickness_coef)
            sound_obj.eigen_decomposition()
            vals = sound_obj.get_vals()
        return vals
        
        
        
