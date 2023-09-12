import sys
sys.path.append('./')
import torch
import numpy as np
from src.dmtet.dmtet import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh

res = 8
num_iter = 100
DMTet = DMTetGeometry(res=res, scale = 0.5)
optimizer = torch.optim.Adam(DMTet.parameters(), lr=0.03)
TetMesh(vertices=DMTet.verts, tets=DMTet.indices).export("output/dmtet_base.msh")
verts, tets = DMTet.getMesh()

for i in range(num_iter):
    verts, tets = DMTet.getMesh()
    TetMesh(vertices=verts, tets=tets).export("output/dmtet.msh")
    loss = DMTet.reg_loss()
    obj = DiffSoundObj(verts, tets)
    obj.eigen_decomposition()
    freqs = obj.get_undamped_freqs()
    loss += torch.sum((freqs[0] - 200) ** 2)
    print(freqs)
    print("iter:{}, reg_loss: {}".format(i, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    


    


