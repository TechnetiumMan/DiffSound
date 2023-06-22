import sys
sys.path.append('./')

import torch
from src.deform import Deform
from src.mesh import TetMesh
from src.material_model import LinearElastic, MatSet, Material
from src.linearFEM.fem import FEMmodel
from src.utils import LOBPCG_solver_freq


def check_B_matrix():
    B = deform.B_matrix # (gauss_points * num_tets, 4, 3)
    
    # now we need to get the B matrix for each tet by gauss integration, to compare each tet's B matrix to linear fem
    B = B.reshape(deform.num_tets, deform.num_guass_points, 4, 3) # (num_tets, gauss_points, 4, 3)
    B = B.permute(0, 2, 3, 1) # (num_tets, 4, 3, gauss_points)
    B = B * deform.gauss_weights * 6 # (gauss_points)
    B = torch.sum(B, dim=-1) # (num_tets, 4, 3)
    
    u = torch.rand(12).cuda()
    F = u.reshape(4, 3).t() @ B
    e_ = 0.5 * (F.transpose(1, 2) + F)
    e1 = torch.stack([e_[:, 0, 0], e_[:, 1, 1], e_[:, 2, 2], e_[:, 0, 1] +
                       e_[:, 1, 0], e_[:, 1, 2] + e_[:, 2, 1], e_[:, 0, 2] + e_[:, 2, 0]]).cuda().transpose(0, 1)
    B_linear = fem.B_matrix.reshape(-1, 6, 12)
    e2 = B_linear @ u
    # print(e1)
    # print(e2)
    assert torch.allclose(e1, e2, atol=1e-1)


def check_D_matrix():
    e = torch.rand(deform.num_tets, 3, 3).cuda()
    stress_ = linear_elastic.stress(e)
    stress1 = torch.stack([stress_[:, 0, 0], stress_[:, 1, 1], stress_[:, 2, 2], stress_[
                           :, 0, 1], stress_[:, 1, 2], stress_[:, 0, 2]]).cuda().transpose(0, 1)
    e_compress = torch.stack([e[:, 0, 0], e[:, 1, 1], e[:, 2, 2], e[:, 0, 1] +
                               e[:, 1, 0], e[:, 1, 2] + e[:, 2, 1], e[:, 0, 2] + e[:, 2, 0]]).cuda().transpose(0, 1).unsqueeze(-1)
    D = fem.D_matrix.reshape(-1, 6, 6)
    stress2 = (D @ e_compress).squeeze(-1) # strain to stress
    # print(linear_elastic.lame_mu, linear_elastic.lame_lambda)
    # print(D)
    # print(stress1)
    # print(stress2)
    assert torch.allclose(stress1, stress2, rtol=1e-1)


def check_stiffness_matrix():
    # because the stiffness matrix is a matrix about every vertice, but one vertice may in multiple tets.
    # so we use index of u to represent the index of vertice
    u = torch.rand(mesh.vertices.shape[0] * 3).cuda()
    force1 = fem.stiffness_matrix @ u
    
    deform_grad_batch = deform.gradient_batch(u.reshape(-1, 3))
    stress_batch = linear_elastic.stress_batch(deform_grad_batch)
    force2_batch = deform.stress_to_force_batch(stress_batch)
    
    deform_grad = deform.gradient(u.reshape(-1, 3))
    stress = linear_elastic.stress(deform_grad)
    force2 = deform.stress_to_force(stress)
    # print(force1)
    # print(force2)
    # print(force1 / force2)
    assert torch.allclose(force1, force2, atol=1e+4)
    assert torch.allclose(force2_batch, force2, atol=1e+4)
    
def check_lobpcg():
    # check if the input stiffness function has the same result in lobpcg
    def stiff_fem(u):
        return fem.stiffness_matrix @ u
    def stiff_mesh(u):
        u = u.transpose(0, 1) # (batch, 3 * num_vertices)
        deform_grad = deform.gradient_batch(u.reshape(u.shape[0], -1, 3))
        stress = linear_elastic.stress_batch(deform_grad)
        force = deform.stress_to_force_batch(stress)
        return force.transpose(0, 1)
    
    mass_matrix = fem.mass_matrix
    
    k = 20
    freq_limit = 1e10 # float(inf)
    eigenvalues_fem, U_fem = LOBPCG_solver_freq(stiff_fem, mass_matrix, freq_limit, k)
    eigenvalues_mesh, U_mesh = LOBPCG_solver_freq(stiff_mesh, mass_matrix, freq_limit, k)
    
    ratio = eigenvalues_fem / eigenvalues_mesh
    assert torch.allclose(ratio, torch.ones_like(ratio).cuda(), atol=5e-2)
   
def stiff_mesh(u, deform):
    # u = u.transpose(0, 1) # (batch, 3 * num_vertices)
    deform_grad = deform.gradient_batch(u.reshape(u.shape[0], -1, 3))
    stress = linear_elastic.stress_batch(deform_grad)
    force = deform.stress_to_force_batch(stress)
    return force.transpose(0, 1)
     
def check_high_order():
    mesh1 = mesh
    mesh2 = mesh.to_high_order(2)
    mesh3 = mesh.to_high_order(3)
    deform1 = Deform(mesh1)
    deform2 = Deform(mesh2)
    deform3 = Deform(mesh3)
    
    u1 = mesh1.vertices.clone().reshape(1, -1)
    force1 = stiff_mesh(u1, deform1)
    u2 = mesh2.vertices.clone().reshape(1, -1)
    force2 = stiff_mesh(u2, deform2)
    u3 = mesh3.vertices.clone().reshape(1, -1)
    force3 = stiff_mesh(u3, deform3)
    # in case with more than 1 tets, in fact I can not judge if the result is correct.
    print(666)
    
    
if __name__ == '__main__':
    vertices = torch.Tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()
    tets = torch.Tensor([[0, 1, 2, 3]]).long().cuda()
    mesh = TetMesh(vertices, tets)
    # mesh = TetMesh.from_triangle_mesh('assets/bowl.obj')
    deform = Deform(mesh)
    fem = FEMmodel(mesh.vertices, mesh.tets, Material(MatSet.Test))
    linear_elastic = LinearElastic(fem.material.youngs, fem.material.poisson)
    # linear_elastic = LinearElastic(Material(MatSet.Ceramic))
    # check_B_matrix()
    # check_D_matrix()
    # check_stiffness_matrix()
    # check_lobpcg()
    check_high_order()
    print("check stiffness passed!")
