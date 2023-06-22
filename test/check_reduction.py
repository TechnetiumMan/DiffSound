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
from src.utils import load_audio, dense_to_sparse, LOBPCG_solver_freq
from src.mesh import TetMesh
from src.deform import Deform
from src.solve import WaveSolver
from src.spectrogram import MelSpectrogram, plot_spectrogram
from src.linearFEM.fem import FEMmodel
import torchaudio.transforms as T

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

class DiffSoundObj():
    def __init__(self, mesh_dir, audio_dir):
        '''
        mesh_dir: the directory of the mesh
        audio_dir: the directory of the audio
        mat: the material of the object
        '''
        self.mesh_dir = mesh_dir
        self.audio_dir = audio_dir
        self.audios, self.forces, self.sr = load_audio(audio_dir)
        self.tetmesh = TetMesh.from_triangle_mesh(mesh_dir + 'model.stl')
        self.femmodel = FEMmodel(
            self.tetmesh.vertices, self.tetmesh.tets, Material(MatSet.Steel))

    def reduction_solve(self, freq_limit):
        '''
        extract modes freq < freq_limit, and solve the vibration
        '''

        # def stiff_matvec(x: torch.Tensor):
        #     return self.femmodel.stiffness_matrix @ x
        # we assume that the stiffness can be represented as a matrix, and extract low freq modes
        self.stiffness_matrix = self.femmodel.stiffness_matrix 
        self.mass_matrix = self.femmodel.mass_matrix
        
        # for a test of callable stiffness, we define a function
        def stiff_func(x):
            return self.stiffness_matrix @ x
        
        self.eigenvalues, self.eigenvectors = LOBPCG_solver_freq(stiff_func, self.mass_matrix, freq_limit) # 10 for a test
        # now eigenvector matrix U and diag eigenvalue matrix S satisfy: KU=MUS 
        # but only keep k=10 modes.
        # since u=Uq, the vibration can be: q'' + Dq' + Sq = (U^T)f 
        
        U = self.eigenvectors
        S = torch.diag(self.eigenvalues)

        # when damping is 0, it don't need extra calculation in model reduction, so just return 0
        def damping_matvec(x): return torch.zeros_like(x) 
        
        # and stiffness matrix is: S = U^T * K * U
        def stiff_matvec(x): return S @ x
        
        force_signal = self.forces[0]  # only use the first force
        force_node_idx = 0
        print(force_signal.shape)

        def get_force(t):
            t_idx = int(t * self.sr * 10)
            force_all_nodes = torch.zeros(
                self.femmodel.mass_matrix.shape[0]).cuda()
            if t_idx < len(force_signal):
                force_all_nodes[force_node_idx] = force_signal[t_idx]
                
            # now in model reduction, the force is: f = U^T * force_all_nodes
            f = U.t() @ force_all_nodes
            return f

        # now mass matrix is a identity matrix, stiffness matrix is S, force is f
        mass_matrix = torch.eye(self.eigenvalues.shape[0]).cuda().to_sparse()
        
        dt = 1e-1 / self.sr # use 10x sr to avoid inaccuracy in RK4 step simulation
        solver = WaveSolver(mass_matrix, damping_matvec,
                            stiff_matvec, get_force, dt)

        x = solver.solve(5000)
        x = x.transpose(0, 1)
        # u=Uq
        x = U @ x
        
        # resample x (in 10x sr) to sr
        x = x[:, ::10]
        
        # now x is the vibration of every single point of the object
        # just add them up
        x = torch.sum(x, dim=0)
        
        # print(x.shape)
        # print(x)
        return x


if __name__ == '__main__':
    mesh_dir = '/data/xcx/mesh_data/full/2/'
    audio_dir = '/data/xcx/audio_data/2/audio'

    obj = DiffSoundObj(mesh_dir, audio_dir)
    signal_low = obj.reduction_solve(3000)
    signal_high = obj.reduction_solve(20000)
    
    # this 2 signal should be the same in the freq below 3000 Hz
    n_fft = 256
    Spec = T.Spectrogram(n_fft=n_fft,
            center=True,
            pad_mode="reflect",
            power=2).cuda()
    low_spec = Spec(signal_low)
    low_spec = low_spec / (low_spec**2).mean()**0.5
    high_spec = Spec(signal_high)
    high_spec = high_spec / (high_spec**2).mean()**0.5
    writer.add_image('low_spec', low_spec.unsqueeze(0))
    writer.add_image('high_spec', high_spec.unsqueeze(0))
    
    eps = 1e-2
    low_log = torch.log10(low_spec + eps)
    high_log = torch.log10(high_spec + eps)
    assert(torch.allclose(low_log[3:30, 2], high_log[3:30, 2], atol=2e-1))
    print("reduction check pass!")
    
    
    
