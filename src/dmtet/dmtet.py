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
import torch.nn.functional as F

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        # 按照每个四面体的每个顶点是否在物体内部，将四面体分类为16类，类序号0-15
        # 每类四面体内部待构造的每个三角面的三个顶点的下标（若无需构造某个三角面，其值全为-1）
        # 注意序号规则：四面体四个顶点分别为0,1,2,3,其二进制编码为1,2,4,8,内部顶点编码和为类别序号，六条边的顺序按照[01,02,03,12,13,23]排列，下标为0-5
        # 以下矩阵为每类四面体内待构造的每个三角面，其三个顶点所在边在这个四面体内的下标
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')
        
        # 现在我们来构造四面体分割(以上三角面分割的推广)
        # 每一类分割出的四面体数量, 事实上，内部点数n和四面体数t的关系满足(n,t)=(0,0),(1,1),(2,3),(3,3),(4,1)
        self.num_tets_table = torch.tensor([0,1,1,3,1,3,3,3,1,3,3,3,3,3,3,1], dtype=torch.long, device='cuda')
        
        # 现在，我们将四面体的四个顶点编号为0-3，每条边上的点编号为4-9，给出每一类四面体分割成的四面体的每个顶点的编号的矩阵
        self.tet_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # (0,0,0,0)
            [ 0,  4,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1], # (0,0,0,1)
            [ 1,  4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1], # (0,0,1,0)
            [ 7,  1,  8,  6,  5,  1,  7,  6,  5,  0,  1,  6], # (0,0,1,1)
            [ 2,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1], # (0,1,0,0)
            [ 4,  0,  6,  7,  9,  0,  7,  6,  7,  0,  9,  2], # (0,1,0,1)
            [ 4,  1,  9,  8,  5,  1,  9,  4,  5,  1,  2,  9], # (0,1,1,0)
            [ 6,  0,  1,  2,  8,  6,  1,  2,  9,  6,  8,  2], # (0,1,1,1)
            [ 3,  6,  9,  8, -1, -1, -1, -1, -1, -1, -1, -1], # (1,0,0,0)
            [ 5,  0,  4,  8,  5,  0,  8,  3,  5,  8,  9,  3], # (1,0,0,1)
            [ 1,  4,  7,  3,  4,  7,  6,  3,  9,  6,  7,  3], # (1,0,1,0)
            [ 0,  1,  5,  3,  5,  1,  9,  3,  5,  1,  7,  9], # (1,0,1,1)
            [ 5,  2,  3,  7,  3,  6,  5,  8,  3,  5,  7,  8], # (1,1,0,0)
            [ 0,  4,  7,  8,  0,  3,  8,  7,  0,  3,  7,  2], # (1,1,0,1)
            [ 4,  1,  2,  3,  4,  3,  2,  5,  4,  3,  5,  6], # (1,1,1,0)
            [ 0,  1,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1]  # (1,1,1,1)
        ], dtype=torch.long, device='cuda')
        

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

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4): 
        # pos_nx3: 变形后空间体素每个顶点的坐标
        # sdf_n: 空间体素每个顶点的sdf值
        # tet_fx4: 空间体素每个四面体的顶点索引
        
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) # occ_fx4: 空间体素每个四面体的每个顶点是否在物体内部
            occ_sum = torch.sum(occ_fx4, -1) # occ_sum: 空间体素每个四面体内部顶点的数量
            valid_tets = (occ_sum>0) & (occ_sum<4) # valid_tets(bool): 每个四面体是否有边经过（即需要用其计算）

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2) # 所有的四面体所有的边（有重复）
            all_edges = self.sort_edges(all_edges) # 排序所有的边，使得每个边的第一个顶点索引小于第二个顶点索引（有重复）
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  # unique_edges: 所有的四面体所有的边（无重复），idx_map: 每个边在unique_edges中的索引
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1 # mask_edges: 恰好有一个顶点在物体内部的边（称有效边）
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda") # 将有效边从0开始编号，其他的边编号为-1
            idx_map = mapping[idx_map] # map edges to verts # 初始每条(有重复)边，其中有效边相关编号（目前和顶点没什么关系！）

            interp_v = unique_edges[mask_edges] # 有效边其两端顶点的下标
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3) # 有效边顶点的坐标
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1) # 有效边顶点的sdf值，每条边应当为一正一负
        edges_to_interp_sdf[:,-1] *= -1 # 为便于后续插值计算，对每条边后一顶点sdf值取负，使得两个顶点sdf值同号

        denominator = edges_to_interp_sdf.sum(1,keepdim = True) # 插值分母(每条边两个顶点sdf值(已同号)之和)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1) # 插值求得每条边上sdf值为零(即物体表面与该边交点)的坐标

        idx_map = idx_map.reshape(-1,6) # 每个四面体的每条边的编号（有效边为>=0, 其他边为-1，事实上每个四面体有效边的数量只有0，3，4三种情况）

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda")) # [1, 2, 4, 8] 
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1) # 按照每个四面体的每个顶点是否在物体内部，将四面体分类为16类，类序号0-15
        
        num_triangles = self.num_triangles_table[tetindex] # 直接查表得到，每一类四面体中待构造三角面数量
        
        num_tets = self.num_tets_table[tetindex] # 查表得到每一类四面体中子四面体数量

        # # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0) # 获得每个四面体内部待构造的每个三角面的三个顶点的下标
        # verts和faces能够一一对应的原因是，每个有效边恰好插值出一个顶点，因此顶点的编号就是有效边的编号
        
        # 由于待生成四面体包含顶点和边点，需要将顶点加入边点集idx_map中
        # 所有需要计算的四面体(包含其所有顶点编号)
        valid_tets_vert_idx = tet_fx4[valid_tets]
        
        # 获得原顶点总数
        num_verts = pos_nx3.shape[0]
        # 将边点的序号加上顶点的数量，以使得其直接顺序排列
        idx_map += num_verts
        
        # 同时包含每个有效四面体顶点和边点的集合（顶点在前，边点在后）
        tet_verts_and_edges = torch.cat([valid_tets_vert_idx, idx_map], dim=1)
        
        side_tets = torch.cat((
            torch.gather(input=tet_verts_and_edges[num_tets == 1], dim=1, index=self.tet_table[tetindex[num_tets == 1]][:, :4]).reshape(-1,4),
            torch.gather(input=tet_verts_and_edges[num_tets == 3], dim=1, index=self.tet_table[tetindex[num_tets == 3]][:, :12]).reshape(-1,4),
        ), dim=0) 
        # 现在，side_tets包含了所有原四面体顶点（小数）和新增的边点（大数）,但是其只包含含有有效边的边缘四面体，
        # 其中小数顶点对应pos_nx3中编号，大数边点对应verts中编号
        # 由于以上操作使得边点序号紧跟着顶点序号，因此直接将边点和顶点cat到一起，其序号应当刚好对上
        all_verts = torch.cat([pos_nx3, verts], dim=0)
        
        # 接着，我们还需要加入四个顶点全在内部的内部四面体，此时无需对点集做任何操作
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4) # occ_fx4: 空间体素每个四面体的每个顶点是否在物体内部
            occ_sum = torch.sum(occ_fx4, -1) # occ_sum: 空间体素每个四面体内部顶点的数量
            inner_tets_bool = (occ_sum == 4) # 每个四面体是否在内部
            inner_tets = tet_fx4[inner_tets_bool] # 所有内部四面体中顶点的下标
            
            all_tets = torch.cat([side_tets, inner_tets], dim=0) # 将两类四面体放在一起
        
        # 注意此时的all_verts是包含没有被用上的点的，这些点必须被去掉，否则质量矩阵会非满秩，无法计算特征值, 以下是去除其中没有被all_tets包含的顶点的操作
        all_unique_tets, all_unique_verts_idx_map = torch.unique(all_tets.reshape(-1), return_inverse=True) 
        all_tets_result = all_unique_verts_idx_map.reshape(-1, 4)
        all_verts_result = all_verts[all_unique_tets]
            
        return all_verts_result, all_tets_result

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

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, res, scale = 1.0):
        super(DMTetGeometry, self).__init__()
        self.scale = scale
        self.grid_res = res
        self.sdf_regularizer = 0.2
        self.marching_tets = DMTet()

        tets = np.load('src/dmtet/data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * self.scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

        # Random init
        sdf = torch.rand_like(self.verts[:,0]) - 0.1

        self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    # 从DMTet中获得四面体体素网格
    def getMesh(self):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        verts, tets = self.marching_tets(v_deformed, self.sdf, self.indices)
        return verts, tets

    def reg_loss(self):
        loss = sdf_reg_loss(self.sdf, self.all_edges).mean() 
        print("sdf reg loss: {}".format(loss))
        return loss * self.sdf_regularizer

