import sys
sys.path.append('./')

import torch
import numpy as np
from src.cuda_module import CUDA_MODULE
from src.mesh import TetMesh
from src.mass_matrix import get_elememt_mass_matrix
from src.utils import calculate_volume
from src.linearFEM.fem import FEMmodel
from src.cuda_module import mass_matrix_assembler


def check_mass_matrix(mesh):
    mass_matrix = mesh.compute_mass_matrix(1.0)

    # when we convert the mass matrix to dense matrix, the size should be vnum * 3
    mm_dense = mass_matrix.to_dense()
    assert (mm_dense.shape[0] == 3 * mesh.vertices.shape[0])

    # the matrix should be symmetric
    assert (torch.allclose(mm_dense, mm_dense.t(), atol=1e-5))

    # the sum of mass matrix should be the volume of the mesh * density * 3
    volume = torch.sum(calculate_volume(mesh))
    mm_sum = torch.sum(mm_dense)
    assert(torch.allclose(mm_sum, volume * 3, atol=1e-5))
    
    # for high order mass matrix check, we should give each vertice a random force,
    # and the acceleration of each vertice should be the same.
    # but when we add more vertices, what force should be given to them?
    # F = Ma! so a = M^{-1}F
    # Oh no! Github tells me sparse matrix inverse is not implemented in pytorch!!!
    # mm_inv = torch.inverse(mass_matrix)
    # rdf = torch.randn((3)).cuda()
    # rdf1 = rdf.repeat((mass_matrix.shape[0] // 3))
    # result1 = mm_inv @ rdf1
    

    # the same as 2nd and 3rd order mass matrix
    mesh2 = mesh.to_high_order(2)
    # mass_matrix2 = mesh2.compute_mass_matrix(1.0)
    # rdf2 = rdf.repeat((mass_matrix2.shape[0] // 3))
    
    # # stop here. we only give force to vertices in 1st order!!!
    # # the force in additional vertices should be 0!!!
    # middle_idx = [1, 3, 5, 6, 7, 8]
    # middle_points = mesh2.tets[:, middle_idx]
    # rdf2[middle_points * 3] = 0
    # rdf2[middle_points * 3 + 1] = 0
    # rdf2[middle_points * 3 + 2] = 0
        
    # mm_inv2 = torch.inverse(mass_matrix2)
    # result_2nd = mm_inv2 @ rdf2
    
    # mm_dense = mass_matrix.to_dense()
    # assert(mm_dense.shape[0] == 3 * mesh2.vertices.shape[0])
    # assert(torch.allclose(mm_dense, mm_dense.t(), atol=1e-5))
    
    # volume = torch.sum(calculate_volume(mesh))
    # mm_sum = torch.sum(mm_dense)
    # assert(torch.allclose(mm_sum, volume * 3, atol=1e-5))
    
    # torch.save(mass_matrix, 'saves/mass_matrix2.pt')
    
    # 3rd order
    mesh3 = mesh.to_high_order(3)
    mass_matrix = mesh3.compute_mass_matrix(1.0)

    # too big to convert to dense!!!

    # mm_dense = mass_matrix.to_dense()
    # assert(mm_dense.shape[0] == 3 * mesh3.vertices.shape[0])
    # assert(torch.allclose(mm_dense, mm_dense.t(), atol=1e-5))
    # volume = torch.sum(calculate_volume(mesh))
    # mm_sum = torch.sum(mass_matrix)
    # assert(torch.allclose(mm_sum, volume * 3, atol=1e-5))
    
    # torch.save(mass_matrix, 'saves/mass_matrix3.pt')
    
def check_linear(mesh):       
    fem_mesh = FEMmodel(mesh.vertices, mesh.tets)
    fem_mass_matrix = fem_mesh.mass_matrix

    mass_matrix = mesh.compute_mass_matrix(2700)

    fem_mm_dense = fem_mass_matrix.to_dense()
    mm_dense = mass_matrix.to_dense()
    assert(torch.allclose(mm_dense, fem_mm_dense, atol=1e-9))
    
    
if __name__ == '__main__':
    mesh = TetMesh.from_triangle_mesh('assets/bowl.obj')
    check_mass_matrix(mesh)
    check_linear(mesh)
    print("check_mass_martix passed!")
