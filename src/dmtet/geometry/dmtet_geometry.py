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
        # According to whether each vertex of the tetrahedron is inside the object, 
        # the tetrahedron is classified into 16 categories, and the class number is 0-15.
        # The three vertices of each triangle to be constructed inside each tetrahedron are indexed in this tetrahedron, 
        # and the index of the edge where the three vertices are located.
        # notice the index rule: the four vertices of the tetrahedron are 0,1,2,3,
        # their binary encoding is 1,2,4,8, the sum of the internal vertex encoding is the category number,
        # the order of the six edges is arranged according to [01,02,03,12,13,23], and the index is 0-5.
        # The following matrix is the three vertices of each triangle to be constructed inside each tetrahedron,
        # and the index of the edge where the three vertices are located in this tetrahedron.
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

        # now we construct the tetrahedron segmentation (the extension of the above triangle segmentation)
        # the number of tetrahedra segmented in each class, in fact, the relationship between the number of 
        # internal points n and the number of tetrahedra t satisfies (n, t) = (0,0), (1,1), (2,3), (3,3), (4,1)
        self.num_tets_table = torch.tensor(
            [0, 1, 1, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1],
            dtype=torch.long,
            device="cuda",
        )

        # Now, we number the four vertices of the tetrahedron as 0-3, the points on each edge as 4-9, 
        # and give the number of each vertex of the tetrahedron segmented into tetrahedra in each class.
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
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: the coordinates of each vertex of the space voxel after deformation
        # sdf_n: the sdf value of each vertex of the space voxel
        # tet_fx4: the vertex index of each tetrahedron of the space voxel
        
        # the same as DMTet at first
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(
                -1, 4
            )  
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (
                occ_sum < 4
            ) 

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(
                -1, 2
            ) 
            
            all_edges = self.sort_edges(all_edges) 
            unique_edges, idx_map = torch.unique(
                all_edges, dim=0, return_inverse=True
            )  

            unique_edges = unique_edges.long()
            mask_edges = (
                occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            )  
            mapping = (
                torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda")
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device="cuda"
            )  
            idx_map = mapping[
                idx_map
            ]  

            interp_v = unique_edges[mask_edges] 
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(
            -1, 2, 1
        )  
        edges_to_interp_sdf[:, -1] *= -1  

        denominator = edges_to_interp_sdf.sum(
            1, keepdim=True
        )  

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(
            1
        )  

        idx_map = idx_map.reshape(
            -1, 6
        )  

        v_id = torch.pow(
            2, torch.arange(4, dtype=torch.long, device="cuda")
        )  # [1, 2, 4, 8]
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(
            -1
        ) 

        num_triangles = self.num_triangles_table[tetindex] 

        num_tets = self.num_tets_table[tetindex]

        # # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )  
        # Get the index of the three vertices of each triangle to be constructed inside each tetrahedron.
        # The reason why verts and faces can correspond one-to-one is that each valid edge just interpolates a vertex,
        # so the vertex number is exactly the edge number

        # Now let's generate the tetrahedra mesh
        # Since the tetrahedron to be generated contains vertices and edge points,
        # the vertices need to be added to the edge point set idx_map
        # All tetrahedra to be calculated (including all vertex numbers):
        valid_tets_vert_idx = tet_fx4[valid_tets]

        num_verts = pos_nx3.shape[0]

        # Add the number of vertices to the edge point number so that they are arranged in order    
        idx_map += num_verts

        # The set containing each valid tetrahedron vertex and edge point (vertices in front, edge points in back)
        tet_verts_and_edges = torch.cat([valid_tets_vert_idx, idx_map], dim=1)

        side_tets = torch.cat(
            (
                torch.gather(
                    input=tet_verts_and_edges[num_tets == 1],
                    dim=1,
                    index=self.tet_table[tetindex[num_tets == 1]][:, :4],
                ).reshape(-1, 4),
                torch.gather(
                    input=tet_verts_and_edges[num_tets == 3],
                    dim=1,
                    index=self.tet_table[tetindex[num_tets == 3]][:, :12],
                ).reshape(-1, 4),
            ),
            dim=0,
        )

        # Now, side_tets contains all the original tetrahedron vertices (decimal) and the newly added edge points (large numbers),
        # but it only contains the edge tetrahedra with valid edges,
        # where the decimal vertices correspond to the numbering in pos_nx3, 
        # and the large edge points correspond to the numbering in verts.
        # Since the above operation makes the edge point number follow the vertex number closely, 
        # the edge point and the vertex can be cat together, and their numbers should match exactly
        all_verts = torch.cat([pos_nx3, verts], dim=0)

        # Next, we also need to add the inner tetrahedra with all four vertices inside,
        # and no operation is required on the point set at this time
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(
                -1, 4
            )  # occ_fx4: whether each vertex of each tetrahedron in the space voxel is inside the object
            occ_sum = torch.sum(occ_fx4, -1)  # occ_sum: the number of vertices in each tetrahedron of the space voxel
            inner_tets_bool = occ_sum == 4  # whether each tetrahedron is inside
            inner_tets = tet_fx4[inner_tets_bool]  # the index of the vertices of all internal tetrahedra

            all_tets = torch.cat([side_tets, inner_tets], dim=0)  # Put the two types of tetrahedra together

        # Note that all_verts at this time contains points that have not been used,
        # these points must be removed, otherwise the quality matrix will be non-singular and the eigenvalues cannot be calculated.
        # The following is the operation of removing vertices that are not included in all_tets.
        all_unique_tets, all_unique_verts_idx_map = torch.unique(
            all_tets.reshape(-1), return_inverse=True
        )
        all_tets_result = all_unique_verts_idx_map.reshape(-1, 4)
        all_verts_result = all_verts[all_unique_tets]

        return all_verts_result, all_tets_result


###############################################################################
# Regularizer
###############################################################################


def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    if len(sdf_f1x6x2) == 0:
        return torch.tensor(0.0, device="cuda")
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float()
    )
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
    def __init__(self, res, scale=1.0, freq_num=1):
        super(DMTetGeometry, self).__init__()
        self.scale = scale
        self.grid_res = res
        self.sdf_regularizer = 0.02
        self.marching_tets = DMTet()

        tets = np.load("src/dmtet/data/tets/{}_tets.npz".format(self.grid_res))
        self.base_verts = torch.tensor(
            tets["vertices"], dtype=torch.float32, device="cuda"
        )
        self.verts = self.base_verts * self.scale
        self.indices = torch.tensor(tets["indices"], dtype=torch.long, device="cuda")
        self.generate_edges()

        self.sdf_nerf = NerfWithPositionEncoding(
            freq_num=freq_num, scale=scale, layer_num=3, hidden_dim=512
        )

        self.deform = torch.nn.Parameter(
            torch.zeros_like(self.verts), requires_grad=True
        )
        self.register_parameter("deform", self.deform)

    def mesh_template_loss(self, nodes, signed_distance, margin):
        sdf = self.sdf_nerf(nodes[signed_distance > margin])
        loss = 0
        return_none = True
        if len(sdf[sdf <= margin]) > 0:
            loss += -(sdf[sdf <= margin]).sum() / self.grid_res**3 * 1000
            return_none = False
        sdf = self.sdf_nerf(nodes[signed_distance < -margin])
        if len(sdf[sdf >= margin]) > 0:
            loss += (sdf[sdf >= margin]).sum() / self.grid_res**3 * 1000
            return_none = False
        if return_none:
            return None
        return loss

    @property
    def sdf(self):
        v_deformed = self.verts + self.scale * 1.8 / (self.grid_res * 2) * torch.tanh(
            self.deform
        )
        # print(v_deformed.min(), v_deformed.max())
        # print(self.verts.min(), self.verts.max())
        # print(self.scale)
        sdf = self.sdf_nerf(v_deformed / self.scale)
        return sdf

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda"
            )
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    # get tetrahedra mesh
    def getMesh(self):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + self.scale * 1.8 / (self.grid_res * 2) * torch.tanh(
            self.deform
        )
        verts, tets = self.marching_tets(v_deformed, self.sdf, self.indices)
        return verts, tets

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

    def reg_loss(self):
        loss = sdf_reg_loss(self.sdf, self.all_edges).mean()
        return loss * self.sdf_regularizer
