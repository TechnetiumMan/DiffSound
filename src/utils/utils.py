import torch
import numpy as np
import os
import torchaudio
import yaml
from glob import glob
from src.lobpcg import lobpcg_func
import matplotlib.pyplot as plt


def remove_duplicate_vertices(vertices, tets):
    '''
    Remove duplicate vertices from a tetrahedral mesh.
    :param vertices: tensor of shape (num_vertices, 3) containing the vertex positions
    :param tets: tensor of shape (num_tets, N) containing the tetrahedra
    :return: a tuple (new_vertices, new_tets) containing the updated vertex and tetrahedra tensors
    '''
    # Calculate unique vertices
    unique_vertices, sort_indices = torch.unique(
        vertices, dim=0, return_inverse=True)
    # Create new tets tensor with updated vertex indices
    new_tets = sort_indices[tets]
    return unique_vertices, new_tets


def load_audio(audio_dir):
    subdirs = glob(audio_dir + "/*")
    audios = []
    forces = []
    for sspath in subdirs:
        print(sspath)
        files = os.listdir(sspath)
        for filename in files:
            filedir = sspath + "/" + filename
            if "mic" in filename:
                audio, sr = torchaudio.load(filedir)
            if "Force" in filename:
                force, sr = torchaudio.load(filedir)
            if "metadata" in filename:
                f = open(filedir)
                yaml_data = yaml.safe_load(f)
                gain = yaml_data.get("gain")
                pad = yaml_data.get("pad")
        force = torchaudio.functional.gain(force, gain[0])
        audio = torchaudio.functional.gain(audio, gain[1])
        force = force[:, pad[0] * sr:]
        audio = audio[:, pad[1] * sr:]
        audios.append(audio[0])  # only use the first channel
        forces.append(force[0])  # only use the first channel
    return audios, forces, sr


# calculate the volume of the mesh
def calculate_volume(mesh):
    p1 = mesh.vertices[mesh.tets[:, 0]]
    p2 = mesh.vertices[mesh.tets[:, 1]]
    p3 = mesh.vertices[mesh.tets[:, 2]]
    p4 = mesh.vertices[mesh.tets[:, 3]]
    V = torch.abs(torch.einsum('ij,ij->i', p1 - p4,
                  torch.cross(p2 - p4, p3 - p4))) / 6
    return V


def dense_to_sparse(A):
    A_sparse = torch.sparse_coo_tensor(
        indices=torch.nonzero(A).t(), values=A[A != 0], size=A.shape)
    A_sparse = A_sparse.coalesce()
    return A_sparse


def LSDloss(spec, spec_gt, eps=1e-7):
    spec = torch.log10(torch.abs(spec) + eps)
    spec_gt = torch.log10(torch.abs(spec_gt) + eps)
    loss_squared = ((spec - spec_gt) ** 2)
    target_lsd = torch.mean(torch.sqrt(torch.mean(loss_squared)))
    return target_lsd


def LOBPCG_solver_freq(stiff_matrix, mass_matrix, niter=1000, freq_limit=None, k=100):
    vals, vecs = lobpcg_func(
        stiff_matrix, mass_matrix, k + 6, niter=niter, tracker=None, largest=False)

    # find the last eigenvalue lower than eigenvalue_limit (notice that vals are sorted in ascending order)
    if freq_limit:
        eigenvalue_limit = (freq_limit * 2 * 3.14159) ** 2
        mask = vals < eigenvalue_limit
        vals = vals[mask]
        vecs = vecs[:, mask]
    return vals[6:], vecs[:, 6:]


def mel_scale(freq):
    if isinstance(freq, torch.Tensor):
        return 2595 * torch.log10(1 + freq / 700)
    return 2595 * np.log10(1 + freq / 700)


def inv_mel_scale(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def mode_loss(pred, gt):
    R = (pred.unsqueeze(1) - gt) ** 2
    err, indices = R.min(dim=0)
    err = torch.sqrt(err) / gt
    err = err.mean()
    # fundamental_freq
    err += torch.abs(pred[0] - gt[0]) / gt[0]
    return err

# load a tet mesh export from comsol in .txt format, return vertices and tets
def comsol_mesh_loader(filename):
    f = open(filename,'r')
    vertices = []
    tets = []
    
    # the .txt file has such format:
    # first some comment lines begin with '%'
    # then lines about vertices, each line contains 3 float number representing the vertice's coordinate
    # then one line begins with '%'
    # then lines about tets, each line contains 4 int number representing the index of vertices of a tet
    # the index is based on the order of lines about vertices, starting with 1.
    
    # skip comment lines
    line = f.readline()
    while line.startswith('%'):
        line = f.readline()
    # read vertices
    while not line.startswith('%'):
        vertices.append([float(coord) for coord in line.split()])
        line = f.readline()
    # skip comment lines
    while line.startswith('%'):
        line = f.readline()
    # read tets
    while line:
        tets.append([int(index)-1 for index in line.split()])
        line = f.readline()
        
    f.close()
    vertices = torch.tensor(vertices).cuda()
    tets = torch.tensor(tets).cuda()
    return vertices, tets

def reconstruct_signal(undamped_freq, damp, sample_num, sample_rate):
    '''
    undamped_freq: (mode_num)
    damp: (mode_num)
    '''
    damped_freq = ((undamped_freq*2*np.pi)**2 - damp**2)**0.5 / (2*np.pi)
    damped_freq = damped_freq.unsqueeze(-1)
    t = torch.arange(sample_num).cuda()
    t = t / sample_rate
    t = t.unsqueeze(0)
    t = t.repeat(len(damped_freq), 1)
    signal = torch.sin(2*np.pi*damped_freq*t)
    signal = signal.sum(0)
    return signal

def plot_spec(spec_gt, spec_predict):
    fig = plt.figure(figsize=(10, 5))
    img = torch.cat([spec_gt, spec_predict], dim=1)
    ax = plt.imshow(img.detach().cpu().numpy(),
                  origin="lower", aspect="auto", cmap='magma')
    # ax.set_title('gt-predict')
    # ax.set_xticks([])
    # ax.set_yticks([])
    fig.tight_layout(pad=0)
    return fig


def plot_signal(siganl):
    fig, ax = plt.subplots(1, 1)
    ax.plot(siganl.detach().cpu().numpy())
    fig.tight_layout(pad=0)
    return fig
    