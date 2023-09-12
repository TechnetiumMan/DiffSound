import sys
sys.path.append('./')
import torch
import numpy as np
from src.dmtet.dmtet import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh

res = 8
num_iter = 1000
DMTet = DMTetGeometry(res=res)
optimizer = torch.optim.Adam(DMTet.parameters(), lr=0.03)

for i in range(num_iter):
    verts, tets = DMTet.getMesh()
    loss = DMTet.reg_loss()
    print("iter:{}, reg_loss: {}".format(i, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
TetMesh(vertices=verts, tets=tets).export("dmtet.msh")
    


