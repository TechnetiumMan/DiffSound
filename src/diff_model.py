import torch
import torch.nn as nn
import numpy as np
from .material_model import TinyNN, LinearElastic, MatSet, Material
from .utils import load_audio, dense_to_sparse, LOBPCG_solver_freq, normalize_input
from .mesh import TetMesh
from .deform import Deform
from .lobpcg import lobpcg_func
from .ddsp.oscillator import WeightedParam
import scipy
import scipy.sparse
from tqdm import tqdm
from src.solve import WaveSolver

batch_trace = torch.vmap(torch.trace)

class TrainableLinear(nn.Module):
    def __init__(self, mat: Material, bin_num=16):
        super().__init__()
        self.youngs_list = torch.linspace(
            np.log(mat.youngs / mat.density / 2),
            np.log(mat.youngs / mat.density * 2),
            bin_num,
        )
        self.youngs_list = torch.exp(self.youngs_list)
        baseline = False
        if baseline:
            self.poisson_list = torch.linspace(mat.poisson, mat.poisson, 1)
        else:
            self.poisson_list = torch.linspace(0.01, 0.499, bin_num)
        self.youngs = WeightedParam(self.youngs_list)
        self.poisson = WeightedParam(self.poisson_list)
        self.mat = mat
        
    def get_value(self):
        self.youngs_value = self.youngs()
        self.poisson_value = self.poisson()
        # keep its grad for comparison with adjoint method
        self.youngs_value.retain_grad()
        self.poisson_value.retain_grad()

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
            self.youngs_value
            * self.poisson_value
            / ((1 + self.poisson_value) * (1 - 2 * self.poisson_value))
        )
        lame_mu = self.youngs_value / (2 * (1 + self.poisson_value))
        stress = lame_mu * (F + F.transpose(1, 2)) + lame_lambda * batch_trace(
            F
        ).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=F.device)
        return stress
    
    # jacobian for Piola tensor d(stress)/d(F_input)
    def jacobian_F(self):
        inputs = torch.zeros(1, 3, 3).cuda().double()
        mat = torch.autograd.functional.jacobian(self.get_stress, inputs)
        return mat
    
    # this function is getting B from theta, for getting dB/d(theta)
    def get_stress_theta(self, F):
        # F = torch.zeros(1, 3, 3).cuda().double()
        youngs = self.theta[0]
        poisson = self.theta[1]
        lame_lambda = (
            youngs
            * poisson
            / ((1 + poisson) * (1 - 2 * poisson))
        )
        lame_mu = youngs / (2 * (1 + poisson))
        stress = lame_mu * (F + F.transpose(1, 2)) + lame_lambda * batch_trace(
            F
        ).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=F.device)
        return stress
    
    def get_B_from_theta(self, theta):
        self.theta = theta
        inputs = torch.zeros(1, 3, 3).cuda().double()
        B = torch.autograd.functional.jacobian(self.get_stress_theta, inputs)
        return B
    
    # jacobian for dB/d(theta)
    def jacobian_dB_dtheta(self):
        youngs = self.youngs_value
        poisson = self.poisson_value
        theta = torch.stack([youngs, poisson]).cuda().double()
        mat = torch.autograd.functional.jacobian(self.get_B_from_theta, theta)
        return mat

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
        inputs = torch.zeros(1, 3, 3).cuda().double()
        mat = torch.autograd.functional.jacobian(self.get_stress, inputs)
        return mat
    

class TrainableNeohookean(nn.Module):
    def __init__(self, mat: Material, bin_num=16):
        super().__init__()
        self.youngs_list = torch.linspace(
            np.log(mat.youngs / mat.density / 2),
            np.log(mat.youngs / mat.density * 2),
            bin_num,
        )
        self.youngs_list = torch.exp(self.youngs_list)
        self.poisson_list = torch.linspace(0.01, 0.499, bin_num)
        self.youngs = WeightedParam(self.youngs_list)
        self.poisson = WeightedParam(self.poisson_list)
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
            self.youngs()
            * self.poisson()
            / ((1 + self.poisson()) * (1 - 2 * self.poisson()))
        )
        lame_mu = self.youngs() / (2 * (1 + self.poisson()))
        F = F + torch.eye(3, device=F.device)
        F_inv_t = torch.inverse(F).transpose(1, 2)
        stress1 = lame_mu * (F - F_inv_t)
        stress2 = (
            lame_lambda
            * torch.log(torch.linalg.det(F)).unsqueeze(-1).unsqueeze(-1)
            * F_inv_t
        )
        return stress1 + stress2

    # jacobian for Piola tensor d(stress)/d(F_input)
    def jacobian_F(self):
        inputs = torch.zeros(1, 3, 3).cuda().double()
        mat = torch.autograd.functional.jacobian(self.get_stress, inputs)
        return mat
    
class TrainableScale(nn.Module):
    def __init__(self, scale_range=[0.5, 1.5], bin_num=16):
        super().__init__()
        self.scale_list = torch.linspace(
            np.log(scale_range[0]),
            np.log(scale_range[1]),
            bin_num,
        ).cuda()
        self.scale_list = torch.exp(self.scale_list)
        self.scale_num = 1
        
        if self.scale_num == 3:
            self.scalex = WeightedParam(self.scale_list).cuda()
            self.scaley = WeightedParam(self.scale_list).cuda()
            self.scalez = WeightedParam(self.scale_list).cuda()
        else:
            self.scale = WeightedParam(self.scale_list).cuda()
    
    def forward(self):
        # now x, y, z scale are the same, but can be different in the future
        if self.scale_num == 3:
            scalex = self.scalex()
            scaley = self.scaley()
            scalez = self.scalez()
            return torch.diag(torch.stack([scalex, scaley, scalez])).cuda()
        else:
            scale = self.scale()
            scale_m = torch.diag(torch.stack([scale, scale, scale])).cuda()
            # scale_m[0, 0] = 1
            # scale_m[1, 1] = 1
            return scale_m
        

def build_model(mesh_dir, mode_num, order,
                     mat, task, vertices=None, tets=None, scale_range=None, init_scale=None):
    if task == "material":
        mat_model = TrainableLinear
    elif task == "shape":
        mat_model = LinearElastic
    elif task == "gt":
        mat_model = LinearElastic
    else:
        raise ValueError("task not defined")
        
    model = DiffSoundObj(mesh_dir, mode_num=mode_num, order=order,
                     mat=mat, mat_model=mat_model, task=task, vertices=vertices, tets=tets)
    
    if task == "material":
        model.init_material_coeffs()
    elif task == "shape":
        model.init_shape(scale_range) 
        if init_scale is not None:
            model.init_scale_coeffs(init_scale)
        with torch.no_grad(): # first update should have no grad!
            model.update_shape()
        
    return model
    

class DiffSoundObj:
    def __init__(
        self,
        mesh_dir=None,
        mode_num=16,
        vertices=None,
        tets=None,
        order=1,
        mat=MatSet.Ceramic,
        mat_model=TrainableLinear,
        task="material",
        mass_matrix_grad=False
    ):
        """
        mesh_dir: the directory of the mesh
        audio_dir: the directory of the audio
        mat: the material of the object
        """
        if mesh_dir is not None:
            self.mesh_dir = mesh_dir
            self.tetmesh = TetMesh.from_triangle_mesh(
                mesh_dir + "model.stl"
            ).to_high_order(order)
        else:
            assert vertices is not None and tets is not None
            self.tetmesh = TetMesh(vertices, tets).to_high_order(order)

        self.deform = Deform(self.tetmesh)
        if mat_model:
            self.material_model = mat_model(Material(mat))
        if mass_matrix_grad:
            # for using DMTet to optimize shape, we must keep grad when calculate mass matrix
            self.mass_matrix = self.tetmesh.compute_mass_matrix(density=1.0, grad=True)
        else:
            self.mass_matrix = self.tetmesh.compute_mass_matrix(density=1.0)
        self.mode_num = mode_num
        self.U_hat_full = None
        self.first_epoch = True
        self.task = task
        self.mat = Material(mat)
    
    # return trainable parameters
    def parameters(self, mat_param=None):
        if self.task == "material":
            if mat_param == "youngs":
                # for OT loss, only tran youngs and fix poisson
                return self.material_model.youngs.parameters() 
            else:
                return self.material_model.parameters()
        elif self.task == "shape":
            return self.scale_model.parameters()
        elif self.task == "debug":
            return None

    def init_material_coeffs(self, scale = 1):
        print("pretrain material")
        optimizer = torch.optim.Adam(self.material_model.parameters(), lr=1e-3)
        gt_youngs = (
            self.material_model.mat.youngs / self.material_model.mat.density * scale
        )
        gt_poisson = self.material_model.mat.poisson
        for i in tqdm(range(2000)):
            optimizer.zero_grad()
            loss = (self.material_model.youngs() - gt_youngs) ** 2 / gt_youngs**2 + (
                self.material_model.poisson() - gt_poisson
            ) ** 2 / gt_poisson**2
            loss.backward()
            optimizer.step()
        print(
            "(net) youngs: ",
            self.material_model.youngs() * self.material_model.mat.density,
            "poisson: ",
            self.material_model.poisson(),
        )
        print(
            "(material table) youngs: ",
            self.material_model.mat.youngs,
            "poisson: ",
            self.material_model.mat.poisson,
        )
        self.scale = torch.eye(3, dtype=torch.float64).cuda()
        self.origin_mass_matrix = self.mass_matrix.clone()
        self.material_model.get_value() # new add: get youngs and poisson value with grad!
        
    def init_shape(self, scale_range):
        self.scale_model = TrainableScale(scale_range=scale_range).cuda()
        self.origin_mass_matrix = self.mass_matrix.clone()
        self.origin_transform_matrix = self.tetmesh.transform_matrix.clone()
        self.origin_shape_func_deriv = self.deform.shape_func_deriv.clone()
        self.origin_integration_weights = self.deform.integration_weights.clone()
        
    def init_scale_coeffs(self, gt_scale):
        print('Start pretraining scale model')
        optimizer = torch.optim.Adam(self.scale_model.parameters(), lr=1e-2)
        for i in tqdm(range(2000)):
            optimizer.zero_grad()
            loss = torch.sum((self.scale_model().diag() - gt_scale) ** 2 / gt_scale**2)
            loss.backward()
            optimizer.step()
        print(
            "scale: ",
            self.scale_model().diag()
        )
                
    # for shape task, we need to update mass matrix, transform matrix and shape_func in each step
    # using shape parameters
    def update_shape(self):
        scale = self.scale_model().to(torch.float64) # (3, 3) diag
        self.scale = scale
        if self.scale.requires_grad:
            self.scale.retain_grad()
        # mass matrix = mass_matrix * (a*b*c)
        self.mass_matrix = self.origin_mass_matrix * torch.linalg.det(scale) 
        self.deform._integration_weights = self.origin_integration_weights * torch.linalg.det(scale) 
        scale_vector = scale.diagonal()
        self.deform._shape_func_deriv = self.origin_shape_func_deriv * (1 / scale_vector)
        self.tetmesh._transform_matrix = (self.origin_transform_matrix.transpose(1, 2) @ scale).transpose(1, 2)

    def update_stiff_matrix(self, assemble_batch_size=20000):
        # with torch.no_grad():
            N = self.deform.num_nodes_per_tet
            batch_size = self.deform.num_tets * self.deform.num_guass_points
            SFDT = self.deform.shape_func_deriv.transpose(1, 2)
            
            # jacobian for Piola tensor d(stress)/d(F_input)
            B = self.material_model.jacobian_F().reshape(1, 9, 9).double()
            
            stress_index = self.deform.stress_index.reshape(batch_size, 3 * N)
            
            # for small batch_size, use assemble_batch_size = 1
            if batch_size < assemble_batch_size:
                assemble_batch_size = batch_size
            
            batch_num = batch_size // assemble_batch_size
            idxs = torch.linspace(0, batch_size, batch_num + 1).long()
            shape = self.mass_matrix.shape
            self.stiff_matrix = torch.sparse_coo_tensor(
                shape, dtype=torch.double
            ).cuda()
            for i in range(batch_num):
                start = idxs[i]
                end = idxs[i + 1]
                A = torch.zeros(end - start, 9, 3 * N).cuda().double()
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
                
    # for getting dK/dtheta, we get dB/dtheta and dK/dB
    def update_stiff_matrix_from_B(self, B):
        assemble_batch_size=20000
        with torch.no_grad():
            N = self.deform.num_nodes_per_tet
            batch_size = self.deform.num_tets * self.deform.num_guass_points
            SFDT = self.deform.shape_func_deriv.transpose(1, 2)
            
            # # jacobian for Piola tensor d(stress)/d(F_input)
            # B = self.material_model.jacobian_F().reshape(1, 9, 9).double()
            
            stress_index = self.deform.stress_index.reshape(batch_size, 3 * N)
            batch_num = batch_size // assemble_batch_size
            idxs = torch.linspace(0, batch_size, batch_num + 1).long()
            shape = self.mass_matrix.shape
            self.stiff_matrix = torch.sparse_coo_tensor(
                shape, dtype=torch.double
            ).cuda()
            for i in range(batch_num):
                start = idxs[i]
                end = idxs[i + 1]
                A = torch.zeros(end - start, 9, 3 * N).cuda().double()
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
                
    def jacobian_dK_dtheta(self):
        with torch.no_grad():
            B = self.material_model.jacobian_F().reshape(1, 9, 9).double()
            dK_dB = torch.autograd.functional.jacobian(self.update_stiff_matrix_from_B, B)
            dB_dtheta = self.material_model.jacobian_dB_dtheta()
            dK_dtheta = torch.einsum('ijk,klm->ijlm', dK_dB, dB_dtheta)
            return dK_dtheta

    # def check_stiff_matrix(self):
    #     self.update_stiff_matrix()
    #     u0 = torch.randn(self.mass_matrix.shape[0], 8).cuda().double()
    #     y1 = self.stiff_func(u0)
    #     y2 = self.stiff_matrix @ u0
    #     assert torch.allclose(y1, y2)

    def stiff_func(self, x_in: torch.Tensor):
        # the input may be (point_num*3, modes) or (point_num*3)
        if len(x_in.shape) == 1:
            x = x_in.unsqueeze(1)
        else:
            x = x_in
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], -1, 3)
        F = self.deform.gradient_batch(x)
        stress = self.material_model(F)
        force = self.deform.stress_to_force_batch(stress)  # (modes, point_num*3)
        force = force.transpose(0, 1)
        if len(x_in.shape) == 1:
            force = force.squeeze(1)
        return force

    def eigen_decomposition(self, native=False, load=False, save=False, freq=None, grad=False):
        # with torch.no_grad():
            if native:
                self.eigen_decomposition_native()
                return
            if grad:
                self.update_stiff_matrix()
            else:
                with torch.no_grad():
                    self.update_stiff_matrix()
            
            if load:
                self.U_hat_full, self.U_hat, self.eigenvalues = torch.load("saves/eigen.pt")
            else:
                with torch.no_grad():
                    if self.U_hat_full is None:
                        self.eigen_decomposition_arpack()
                    else:
                        self.eigen_decomposition_torch()
                
                if freq:
                    eigenvalue_limit = (freq * 2 * 3.14159) ** 2
                    mask = self.eigenvalues < eigenvalue_limit
                    self.eigenvalues = self.eigenvalues[mask]
                    self.U_hat_full = self.U_hat_full[:, mask]
                    self.U_hat = self.U_hat_full[:, self.zero_eigens:]
            
            if save:
                torch.save([self.U_hat_full, self.U_hat, self.eigenvalues], "saves/eigen.pt")

    def eigen_decomposition_native(self):
        with torch.no_grad():
            S, U_hat_full = lobpcg_func(
                self.stiff_func,
                self.mass_matrix,
                self.mode_num + self.zero_eigens,
                X=self.U_hat_full,
                largest=False,
                niter=2000,
            )
        self.U_hat_full = U_hat_full
        self.U_hat = U_hat_full[:, self.zero_eigens:]
        self.eigenvalues = S

    def eigen_decomposition_torch(self):
        with torch.no_grad():
            S, U_hat_full = lobpcg_func(
                self.stiff_matrix,
                self.mass_matrix,
                self.mode_num + self.zero_eigens,
                X=self.U_hat_full if self.U_hat_full is not None else None,
                largest=False,
                niter=2000,
            )
        
        # notice that zero eigenvalues may change during SDF training, we need to check if it is changed
        threshold = 1.
        for i in range(len(S)):
            if S[i] > threshold:
                break
        new_zero_eigens = i
        if new_zero_eigens != self.zero_eigens:
            print("re calculate eigen")
            self.eigen_decomposition_arpack() # old U and S can not be used, have to re-calculate eigenvalue from scratch
            # and in arpack, self.zero_eigens will change
            
        else:
            self.U_hat_full = U_hat_full
            self.U_hat = U_hat_full[:, self.zero_eigens:]
            self.eigenvalues = S

    def eigen_decomposition_arpack(self):
        stiff_mat = scipy.sparse.coo_matrix(
            (
                self.stiff_matrix.values().cpu().numpy(),
                (
                    self.stiff_matrix.indices()[0].cpu().numpy(),
                    self.stiff_matrix.indices()[1].cpu().numpy(),
                ),
            )
        ).tocsr()
        stiff_mat.eliminate_zeros()
        mass_mat = scipy.sparse.coo_matrix(
            (
                self.mass_matrix.values().cpu().numpy(),
                (
                    self.mass_matrix.indices()[0].cpu().numpy(),
                    self.mass_matrix.indices()[1].cpu().numpy(),
                ),
            )
        ).tocsr()
        mass_mat.eliminate_zeros()
        S, U_hat_full = scipy.sparse.linalg.eigsh(
            stiff_mat, M=mass_mat, k=self.mode_num + 6, sigma=0
        )
        
        # print("eigenvalues: ", self.eigenvalues)
        
        # notice that zero eigenvalues may be more than 6 when there are some discrete tets,
        # so we have to count how many zero eigenvalues
        threshold = 1.
        for i in range(len(S)):
            if S[i] > threshold:
                break
        if(i == len(S)):
            raise ValueError("all eigenvalues are zero!!!")
        self.zero_eigens = i
        
        # and then, to make sure mode_num is constant, we have to re-calculate eigenvalues use k = self.mode_num + self.zero_eigens
        if self.zero_eigens > 6:
            S, U_hat_full = scipy.sparse.linalg.eigsh(
                stiff_mat, M=mass_mat, k=self.mode_num + self.zero_eigens, sigma=0
            )
            
        self.U_hat_full = torch.from_numpy(U_hat_full).cuda().double()
        self.eigenvalues = torch.from_numpy(S).cuda().double()
        self.U_hat = torch.from_numpy(U_hat_full[:, self.zero_eigens:]).cuda().double()

    def get_undamped_freqs(self, sample_num = 1, matrix_grad=False):
        if self.task == "shape":
            self.update_shape()

        predict = torch.zeros(self.mode_num).cuda()
        predict += self.eigenvalues[self.zero_eigens:] # (mode_num)

        idxs = torch.randperm(self.mode_num)[:sample_num] # (sample_num)
        U = self.U_hat[:, idxs] # (n, sample_num)
        vals = self.eigenvalues[self.zero_eigens:][idxs]
        if self.task == "shape":
            mass_k = torch.linalg.det(self.scale)
        else:
            mass_k = 1
        if self.task != "gt":
            if not matrix_grad:
                add_term = (U.T @ self.stiff_func(U)).diagonal() - vals * mass_k * (U.T @ (self.origin_mass_matrix @ U)).diagonal()
            else: # in dmtet
                add_term = (U.T @ (self.stiff_matrix @ U)).diagonal() - vals * (U.T @ (self.mass_matrix @ U)).diagonal()
            predict[idxs] += add_term
        predict = torch.sqrt(predict) / 2 / np.pi
        return predict.unsqueeze(1)
    

    def set2set_loss(self, kl_matrix, weights, target_importance):
        """
        kl_matrix: rows is predict, cols is target
        """
        # predict to target
        loss1 = (weights * kl_matrix).sum(1) / weights.sum(1)
        # target to predict
        loss2 = (weights * kl_matrix).sum(0) / weights.sum(0)
        target_importance = (
            target_importance / target_importance.sum() * target_importance.shape[0]
        )
        return loss1.mean() + (loss2 * target_importance).mean()

    def match_loss(
        self,
        undamped_freqs,
        damps,
        sample_num=8,
        smooth_value=1,
        eps=1e-2,
        force=1.0,
        force_sample_rate = 48000,
        alpha=2,
        beta = 10,
    ):
        """
        undamped_freqs: (full_sample_num, high_energy_mode_num), sorted
        damps: (1, high_energy_mode_num), sorted by undamped_freqs
        """
        # we have pretrained in eigen_decomp, so we don't need to do it again
        # if self.first_epoch:
        #     base_eigenvalue = self.eigenvalues[6]
        #     base_eigenvalue_from_data = (
        #         undamped_freqs.mean(0)[0].item() * 2 * np.pi
        #     ) ** 2
        #     self.pretrain_material(base_eigenvalue_from_data / base_eigenvalue)
        #     self.first_epoch = False

        mode_amp = (force / force_sample_rate) * torch.abs(self.U_hat).mean(0) / (self.eigenvalues[self.zero_eigens:] ** 0.5)
        while True:
            x = torch.rand(self.mode_num, sample_num, dtype=torch.float64).cuda() * 2 - 1
            x[x >= 0] += eps
            x[x < 0] -= eps
            x = x * mode_amp.unsqueeze(1)
            predict = self.U_hat.T @ self.stiff_func(
                self.U_hat @ x
            )  # (mode_num, sample_num)
            predict = predict / x
            predict = torch.sqrt(predict) / 2 / np.pi
            if not torch.isnan(predict).any():
                break
            else:
                print("resample")
                # clear memory 
                del x
                del predict
                torch.cuda.empty_cache()

        self.predict = predict
        predict_mean = predict.mean(1)
        # if predict is (n, 1), std will be nan!!!
        if predict.shape[1] == 1:
            predict_std = torch.zeros_like(predict_mean) + eps
        else:
            predict_std = predict.std(1) + eps  # avoid the value is 0

        # target = (undamped_freqs * 2 * np.pi)**2
        target = undamped_freqs
        target = target.transpose(0, 1)  # (mode_num, sample_num)
        target_mean = target.mean(1)
        target_std = target.std(1) + eps

        # KL divergence matrix (mode_num, mode_num)
        predict_mean = predict_mean.unsqueeze(1)
        predict_std = predict_std.unsqueeze(1)
        target_mean = target_mean.unsqueeze(0)
        target_std = target_std.unsqueeze(0)

        kl = (
            torch.log(target_std / predict_std)
            + (predict_std**2 + (predict_mean - target_mean) ** 2)
            / (2 * target_std**2 + smooth_value)
            - 0.5
        )

        kl = kl / (target_mean**2) * (2 * target_std.max() ** 2 + smooth_value)
        # loss of foundamental frequency
        loss_base = kl[0, 0]

        # set to set match loss
        weights = 1.0 / (kl * beta + eps)
        # give higher weight for lower damping
        target_importance = 1.0 / (damps + eps)
        return self.set2set_loss(kl, weights, target_importance) + loss_base * alpha
    
    # linear RK4 forward
    def linear_step_init(self, force, dt, audio_num, sr, mode_num):
        self.mode_num = mode_num # update it to real value restricted by freq limit
        def stiff_matvec(x: torch.Tensor): 
            return self.grad_eigenvalues * x
        
        # using adjoint method will save memory and code below can be run. but for test, we don't need it
        # def stiff_func_matvec(x: torch.Tensor): 
        #     result = (self.U_hat.T @ self.stiff_func(self.U_hat @ x.T)).T
        #     return result 
            
        def damping_matvec(v): 
            damping = (self.mat.alpha + self.mat.beta * self.grad_eigenvalues) * v # damping need grad too!
            return damping
        
        def get_force(t):
            t_int = int(t * sr)
            cnt_f = force[:, t_int]
            f = torch.zeros([cnt_f.shape[0], self.U_hat.shape[0]], dtype=torch.float64).cuda() # (sound_num, point_num)
            f[:, 0] = cnt_f 
            # U_hat: (point_num, mode_num)
            f = f @ self.U_hat # (sound_num, mode_num)
            return f

        self.solver = WaveSolver("identity", damping_matvec,
                            stiff_matvec, get_force, dt, batch_size=audio_num)
    
    # nonlinear RK4 forward 
    def nonlinear_step_init(self, force, dt, audio_num, sr, mode_num, freq_nonlinear, damp_nonlinear):
        self.mode_num = mode_num # update it to real value restricted by freq limit
        self.freq_nonlinear = freq_nonlinear
        self.stiffness_net = TinyNN(mode_num, 16, mode_num).cuda()
        # self.damping_net = TinyNN(mode_num, 16, mode_num).cuda()
        
        def stiff_matvec(x: torch.Tensor): 
            # for stiffness, we use eigenvalues and only train network for nonlinear part
            # the input of network is x, and output is in (-1, 1)
            # Question: should we use reduced x for network input instead of x in all nodes?
            x_in = normalize_input(x)
            x_out = self.stiffness_net(x_in).double() # (mode_num) in (-1, 1)
            result = self.grad_eigenvalues * x * (1 + self.freq_nonlinear * x_out)
            return result
        
        # using Piola-stress-based stiffness will run out of memory!!!
        # def stiff_func_matvec(x: torch.Tensor): 
        #     result = (self.U_hat.T @ self.stiff_func(self.U_hat @ x.T)).T
        #     return result 
            
        def damping_matvec(v): 
            damping = (self.mat.alpha + self.mat.beta * self.grad_eigenvalues) * v # damping need grad too!
            # TODO: add a network for nonlinear
            return damping
        
        def get_force(t):
            t_int = int(t * sr)
            cnt_f = force[:, t_int]
            f = torch.zeros([cnt_f.shape[0], self.U_hat.shape[0]], dtype=torch.float64).cuda() # (sound_num, point_num)
            f[:, 0] = cnt_f # (sound_num, point_num), the force only give to point 0
            # U_hat: (point_num, mode_num)
            f = f @ self.U_hat # (sound_num, mode_num)
            return f

        self.solver = WaveSolver("identity", damping_matvec,
                            stiff_matvec, get_force, dt, batch_size=audio_num)
        
    
    # for our new step model, RK4 forward is needed
    def forward(self, step_num):
        x = self.solver.solve(step_num)
        x = x.transpose(0, 1).transpose(1, 2)
        x = self.U_hat @ x
        return x
    
    # eigenvalue with grad for step
    def get_grad_eigenvalues(self, sample_num = None):
        
        predict = torch.zeros(self.mode_num).cuda()
        predict += self.eigenvalues[self.zero_eigens:] # (mode_num)
        
        if sample_num is None: # use all modes for sample
            sample_num = self.mode_num
            U = self.U_hat
            vals = self.eigenvalues[self.zero_eigens:]
            add_term = (U.T @ self.stiff_func(U)).diagonal() - vals * (U.T @ (self.origin_mass_matrix @ U)).diagonal()
            predict += add_term
        else: # random choose some modes for sample
            idxs = torch.randperm(self.mode_num)[:sample_num] # (sample_num)
            U = self.U_hat[:, idxs] # (n, sample_num)
            vals = self.eigenvalues[self.zero_eigens:][idxs]
            add_term = (U.T @ self.stiff_func(U)).diagonal() - vals * (U.T @ (self.origin_mass_matrix @ U)).diagonal()
            predict[idxs] += add_term
            
        # predict = torch.sqrt(predict) / 2 / np.pi
        # return predict.unsqueeze(1)
        
        self.grad_eigenvalues = predict
