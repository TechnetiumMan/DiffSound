import sys
sys.path.append('./')
import torch
import numpy as np
from src.dmtet.dmtet import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh

import meshio
mesh = meshio.read("data/model.msh")
verts = torch.tensor(mesh.points).cuda().float()
print(verts.min(), verts.max())
tets = torch.tensor(mesh.cells_dict["tetra"]).cuda().long()
obj_gt = DiffSoundObj(verts, tets)
obj_gt.eigen_decomposition()
vals_gt = obj_gt.get_vals()
print(vals_gt)

res = 32
num_iter = 500
DMTet = DMTetGeometry(res=res, scale = 0.2, radius = 0.3, fill_rate=1.0).cuda()
optimizer = torch.optim.Adam(DMTet.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
TetMesh(vertices=DMTet.verts, tets=DMTet.indices).export("output/dmtet_base.msh")
verts, tets = DMTet.getMesh()



for i in range(num_iter):
    verts, tets = DMTet.getMesh()
    print(verts.shape)
    TetMesh(vertices=verts, tets=tets).export("output/dmtet.msh")
    loss = DMTet.reg_loss()
    obj = DiffSoundObj(verts, tets)
    obj.eigen_decomposition()
    vals = obj.get_vals()
    loss += torch.nn.MSELoss()(vals, vals_gt)
    print(vals, vals_gt)
    print("iter:{}, reg_loss: {}".format(i, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    


    


