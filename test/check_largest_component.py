import sys

sys.path.append("./")
import torch
import numpy as np
from src.dmtet.dmtet import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh

res = 8
scale = 0.2
DMTet = DMTetGeometry(res=res, scale=scale, freq_num=3).cuda()

unit_vert = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32
)
verts = torch.zeros((0, 3), dtype=torch.float32).cuda()
tets = torch.zeros((0, 4), dtype=torch.int32).cuda()
vert_size = scale / res
# add random noise tetrahedron
for i in range(5):
    rand_pos = torch.rand(3) * scale - scale / 2
    vert = unit_vert * vert_size + rand_pos
    offset = verts.shape[0]
    verts = torch.cat([verts, vert.cuda()], dim=0)
    tets = torch.cat([tets, torch.tensor([[0, 1, 2, 3]]).cuda() + offset], dim=0)

vert_last = vert[0] + torch.tensor([0, 0, 2], dtype=torch.float32) * vert_size
offset = verts.shape[0]
verts = torch.cat([verts, vert_last.cuda().unsqueeze(0)], dim=0)
tets = torch.cat([tets, torch.tensor([[-4, -3, -2, 0]]).cuda() + offset], dim=0)
print(verts)
print(tets)
TetMesh(vertices=verts, tets=tets).export("output/dmtet_noise.msh")
verts, tets = DMTet.get_largest_connected_component(verts, tets)
TetMesh(vertices=verts, tets=tets).export("output/dmtet_noise_lcc.msh")
print(verts)
print(tets)
