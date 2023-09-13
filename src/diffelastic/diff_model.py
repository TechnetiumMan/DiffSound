import torch
import torch.nn as nn
import numpy as np
from .material_model import MatSet, Material
from .mesh import TetMesh
from .deform import Deform
from ..lobpcg import lobpcg_func
import scipy
import scipy.sparse
from tqdm import tqdm
from .shape_func import get_shape_function
from .mass_matrix import get_elememt_mass_matrix

batch_trace = torch.vmap(torch.trace)

# linear elastic without any trainable args
class FixedLinear(nn.Module):
    def __init__(self, mat: Material):
        self.youngs = mat.youngs / mat.density
        self.poisson = mat.poisson
        self.mat = mat
        
    def forward(self, F: torch.Tensor):
        """
        Piola stress
        F: deformation gradient with shape (batch_size, node_size, 3, 3)
        return: stress with shape (batch_size, node_size, 3, 3)
        """
        batch_size, node_size, _, _ = F.shape
        F = F.reshape(batch_size * node_size, 3, 3)
        stress = self.get_stress(F)
        return stress.reshape(batch_size, node_size, 3, 3)
    
    def get_stress(self, F):
        lame_lambda = (
            self.youngs
            * self.poisson
            / ((1 + self.poisson) * (1 - 2 * self.poisson))
        )
        lame_mu = self.youngs / (2 * (1 + self.poisson))
        stress = lame_mu * (F + F.transpose(1, 2)) + lame_lambda * batch_trace(
            F
        ).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=F.device)
        return stress
    
    # jacobian for Piola tensor d(stress)/d(F_input)
    def jacobian_F(self):
        inputs = torch.zeros(1, 3, 3).cuda().float()
        mat = torch.autograd.functional.jacobian(self.get_stress, inputs)
        return mat
    

class DiffSoundObj:
    def __init__(
        self,
        vertices,
        tets,
        mode_num=16,
        mat=MatSet.Ceramic,
    ):
        self.tetmesh = TetMesh(vertices, tets)
        self.deform = Deform(self.tetmesh)
        self.material_model = FixedLinear(Material(mat))
        self.mode_num = mode_num

    def update_stiff_matrix(self, assemble_batch_size=20000):
        N = self.deform.num_nodes_per_tet
        batch_size = self.deform.num_tets * self.deform.num_guass_points
        SFDT = self.deform.shape_func_deriv.transpose(1, 2)
        
        # jacobian for Piola tensor d(stress)/d(F_input)
        B = self.material_model.jacobian_F().reshape(1, 9, 9).float()
        
        stress_index = self.deform.stress_index.reshape(batch_size, 3 * N)
        
        # for small batch_size, use assemble_batch_size = 1
        if batch_size < assemble_batch_size:
            assemble_batch_size = batch_size
        
        batch_num = batch_size // assemble_batch_size
        idxs = torch.linspace(0, batch_size, batch_num + 1).long()
        shape = torch.Size(
            [3 * self.tetmesh.vertices.shape[0], 3 * self.tetmesh.vertices.shape[0]])
        self.stiff_matrix = torch.sparse_coo_tensor(
            shape, dtype=torch.float
        ).cuda()
        for i in range(batch_num):
            start = idxs[i]
            end = idxs[i + 1]
            A = torch.zeros(end - start, 9, 3 * N).cuda().float()
            A[:, :3, 0::3] = SFDT[start:end]
            A[:, 3:6, 1::3] = SFDT[start:end]
            A[:, 6:9, 2::3] = SFDT[start:end]
            values = (A.transpose(1, 2) @ B @ A) * self.deform.integration_weights[
                start:end
            ]
            rows = (
                stress_index[start:end].unsqueeze(2).repeat(1, 1, 3 * N).reshape(-1)
            )
            cols = (
                stress_index[start:end].unsqueeze(1).repeat(1, 3 * N, 1).reshape(-1)
            )
            indices = torch.stack([rows, cols], dim=0).long()
            self.stiff_matrix = self.stiff_matrix + torch.sparse_coo_tensor(
                indices, values.reshape(-1), shape
            )
            self.stiff_matrix = self.stiff_matrix.coalesce()

    def update_mass_matrix(self, density):
        '''
        Return the mass matrix of the mesh(as a coo-sparse matrix).
        '''
        msize_list = [12, 30, 60]
        msize = msize_list[self.tetmesh.order - 1]
        values = torch.zeros((msize * msize * self.tetmesh.tets.shape[0]), dtype=torch.float32).cuda()
        rows = torch.zeros_like(values, dtype=torch.int32).cuda()
        cols = torch.zeros_like(values, dtype=torch.int32).cuda()
        vertices_cuda = self.tetmesh.vertices.to(
            torch.float32).reshape(-1).contiguous().cuda()
        tets_cuda = self.tetmesh.tets.to(torch.int32).reshape(-1).contiguous().cuda()
        element_mm = get_elememt_mass_matrix(self.tetmesh.order)
        vnum_list = [4, 10, 20]
        vnum = vnum_list[self.tetmesh.order - 1]
        msize = vnum * 3
        
        idx_num = len(tets_cuda) // vnum
        idx = torch.arange(0, idx_num, dtype=torch.int32).cuda()
        tets_ptr = idx * vnum
        
        x = torch.zeros((idx_num, 4), dtype=torch.float32).cuda()
        y = torch.zeros((idx_num, 4), dtype=torch.float32).cuda()
        z = torch.zeros((idx_num, 4), dtype=torch.float32).cuda()
        if self.tetmesh.order == 1:
            for i in range(4):
                x[:, i] = vertices_cuda[tets_cuda[tets_ptr + i] * 3]
                y[:, i] = vertices_cuda[tets_cuda[tets_ptr + i] * 3 + 1]
                z[:, i] = vertices_cuda[tets_cuda[tets_ptr + i] * 3 + 2]
        else:
            raise NotImplementedError # TODO
        
        V = ((x[:, 1] - x[:, 0]) * ((y[:, 2] - y[:, 0]) * (z[:, 3] - z[:, 0]) - (y[:, 3] - y[:, 0]) * (z[:, 2] - z[:, 0])) +
            (y[:, 1] - y[:, 0]) * ((z[:, 2] - z[:, 0]) * (x[:, 3] - x[:, 0]) - (z[:, 3] - z[:, 0]) * (x[:, 2] - x[:, 0])) +
            (z[:, 1] - z[:, 0]) * ((x[:, 2] - x[:, 0]) * (y[:, 3] - y[:, 0]) - (x[:, 3] - x[:, 0]) * (y[:, 2] - y[:, 0])))
        V = torch.abs(V)
        
        vid = torch.zeros((idx_num, vnum*3), dtype=torch.int32).cuda()
        for i in range(vnum):
            vid[:, i*3] = tets_cuda[tets_ptr + i] * 3
            vid[:, i*3+1] = tets_cuda[tets_ptr + i] * 3 + 1
            vid[:, i*3+2] = tets_cuda[tets_ptr + i] * 3 + 2
            
        # values[offset + i * msize + j] = m[i * msize + j] * d * V
        offset = idx * msize * msize
        for i in range(msize):
            for j in range(msize):
                values[offset + i * msize + j] = element_mm[i * msize + j] * density * V[idx]
                rows[offset + i * msize + j] = vid[idx, i]
                cols[offset + i * msize + j] = vid[idx, j]
            
        indices = torch.stack([rows, cols], dim=0).long()
        shape = torch.Size(
            [3 * self.tetmesh.vertices.shape[0], 3 * self.tetmesh.vertices.shape[0]])
        mass_matrix = torch.sparse_coo_tensor(indices, values, shape)
        self.mass_matrix = mass_matrix.coalesce()
                

    def eigen_decomposition(self):
        self.update_mass_matrix(self.material_model.mat.density)
        self.update_stiff_matrix()
        self.eigen_decomposition_arpack()

    def eigen_decomposition_arpack(self):
        stiff_mat = scipy.sparse.coo_matrix(
            (
                self.stiff_matrix.detach().values().cpu().numpy(),
                (
                    self.stiff_matrix.detach().indices()[0].cpu().numpy(),
                    self.stiff_matrix.detach().indices()[1].cpu().numpy(),
                ),
            )
        ).tocsr()
        stiff_mat.eliminate_zeros()
        mass_mat = scipy.sparse.coo_matrix(
            (
                self.mass_matrix.detach().values().cpu().numpy(),
                (
                    self.mass_matrix.detach().indices()[0].cpu().numpy(),
                    self.mass_matrix.detach().indices()[1].cpu().numpy(),
                ),
            )
        ).tocsr()
        mass_mat.eliminate_zeros()
        S, U_hat_full = scipy.sparse.linalg.eigsh(
            stiff_mat, M=mass_mat, k=self.mode_num + 6, sigma=20000
        )
        # while (S > 20000).sum() < self.mode_num:
        #     S, U_hat_full = scipy.sparse.linalg.eigsh(
        #         stiff_mat, M=mass_mat, k=self.mode_num + (S < 20000).sum(), sigma=20000
        #     )
        # mask = S > 20000

        self.U_hat_full = torch.from_numpy(U_hat_full).cuda().float()
        self.eigenvalues = torch.from_numpy(S).cuda().float()[6:] 
        print(self.eigenvalues)
        self.U_hat = torch.from_numpy(U_hat_full[:, 6:][:, :self.mode_num]).cuda().float()

    def get_vals(self):
        predict = torch.zeros(self.mode_num).cuda()
        predict += self.eigenvalues
        U = self.U_hat
        vals = self.eigenvalues
        add_term = (U.T @ (self.stiff_matrix @ U)).diagonal() - vals * (U.T @ (self.mass_matrix @ U)).diagonal()
        predict += add_term
        return predict.unsqueeze(1)
