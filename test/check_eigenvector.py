# this file is for check the generalized eigenvalue of mass matrix and stiffness matrix
# we want to do a model reduction, to remove high freq modes and keep low freq modes (to avoid unstable step simulation)
# so we find generalized eigenvalue of mass matrix and stiffness matrix, and only keep the low freq modes

import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from src.material_model import TinyNN, LinearElastic, MatSet, Material
from src.utils import load_audio, dense_to_sparse, LOBPCG_solver_freq, comsol_mesh_loader
from src.mesh import TetMesh
from src.deform import Deform
from src.solve import WaveSolver
from src.spectrogram import MelSpectrogram, plot_spectrogram
from src.linearFEM.fem import FEMmodel
import torchaudio.transforms as T
from src.lobpcg import lobpcg_func
from src.visualize import viewer
import numpy as np

def LOBPCG_solver_test(stiff_matrix, mass_matrix, k):
    vals, vecs = lobpcg_func(
        stiff_matrix, mass_matrix, k, tracker=None, largest=False)

    return vals, vecs

class TestObj():
    def __init__(self, mesh_dir=None, order=1, mat=MatSet.Ceramic, MatModel=LinearElastic, vertices=None, tets=None):
        '''
        mesh_dir: the directory of the mesh
        audio_dir: the directory of the audio
        mat: the material of the object
        '''
        if(mesh_dir):
            self.mesh_dir = mesh_dir
            self.tetmesh = TetMesh.from_triangle_mesh(mesh_dir + 'model.stl').to_high_order(order)
        else:
            self.tetmesh = TetMesh(vertices, tets).to_high_order(order)
            
        self.deform = Deform(self.tetmesh)
        self.mat = Material(mat)
        self.mat_model = MatModel(self.mat.youngs, self.mat.poisson)
        self.mass_matrix = self.tetmesh.compute_mass_matrix(self.mat.density)
        self.mat_weights = None
    
    def stiff_func(self, x_in: torch.Tensor):
        # the input may be (point_num*3, modes) or (point_num*3)
        if (len(x_in.shape) == 1):
            x = x_in.unsqueeze(1)
        else:
            x = x_in
        x = x.transpose(0, 1)
        x = x.reshape(x.shape[0], -1, 3)
        F = self.deform.gradient_batch(x)
        stress = self.mat_model.stress_batch(F, self.mat_weights)
        force = self.deform.stress_to_force_batch(
            stress)  # (modes, point_num*3)
        force = force.transpose(0, 1)
        if (len(x_in.shape) == 1):
            force = force.squeeze(1)
        return force

    def check_eigenvector(self, k):
        # fem = FEMmodel(self.tetmesh.vertices, self.tetmesh.tets, Material(MatSet.Ceramic))
        # self.eigenvalues, self.eigenvectors = LOBPCG_solver_test(fem.stiffness_matrix , self.mass_matrix, k=k) 
        self.eigenvalues, self.eigenvectors = LOBPCG_solver_test(self.stiff_func , self.mass_matrix, k=k) 
        U = self.eigenvectors
        S = self.eigenvalues
        print(S)
        eigenfreq = torch.sqrt(S) / (2 * np.pi)
        return eigenfreq


torch.set_printoptions(precision=8)
if __name__ == '__main__':
#     mesh_dir = '/data/xcx/mesh_data/full/2/'
    # audio_dir = '/data/xcx/audio_data/2/audio'
    # vertices = torch.Tensor(
    #     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()
    # tets = torch.Tensor([[0, 1, 2, 3]]).long().cuda()
    filename = "assets/spoon_smaller.txt"
    vertices, tets = comsol_mesh_loader(filename)
    # triangles = torch.Tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]).long()
    # viewer(vertices.cpu(), triangles).show()

    obj1 = TestObj(vertices=vertices, tets=tets, order=1)
    obj2 = TestObj(vertices=vertices, tets=tets, order=2)
    # obj3 = TestObj(vertices=vertices, tets=tets, order=3)
    # obj1 = TestObj(mesh_dir, order=1)
    # obj2 = TestObj(mesh_dir, order=2)
    # obj3 = TestObj(mesh_dir, order=3)
    eigenfreq1 = obj1.check_eigenvector(20)
    print(eigenfreq1)
    eigenfreq2 = obj2.check_eigenvector(20)
    print(eigenfreq2)
    # eigenfreq3 = obj3.check_eigenvector(20)
    # print(eigenfreq3)
    # U2, S2 = obj2.check_eigenvector(4)
    # U3, S3 = obj3.check_eigenvector(4)
    