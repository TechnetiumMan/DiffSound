import sys
sys.path.append('./')
import torch
import numpy as np
from src.dmtet.dmtet_test import DMTetGeometry
from src.mesh import TetMesh
from src.deform import Deform
from src.visualize import viewer

res = 32
num_iter = 10000
DMTet = DMTetGeometry(res=res, iter=num_iter)
verts, tets = DMTet.getMesh()
tet_mesh = TetMesh(vertices=verts, tets=tets) 

# 这里假设gt audio是1000Hz振幅=1的正弦波
num_step = 1000
freq = 1000
sr = 44100
freq = torch.tensor([2 * np.pi * freq]).repeat(num_step)
freq = torch.cumsum(freq / sr, dim=0)
gt_audio = torch.sin(freq).unsqueeze(0).cuda()
loss_func = torch.nn.MSELoss()

optimizer = torch.optim.Adam(DMTet.parameters(), lr=0.03)

for i in range(num_iter):
    audio_loss, reg_loss = DMTet.tick(gt_audio, loss_func, i, sr)
    print("iter:{}, audio_loss: {}, reg_loss: {}".format(i, audio_loss, reg_loss))
    
    total_loss = audio_loss + reg_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
# viewer
viewer(verts.cpu().detach(), tets.cpu().detach(), draw_tet=True).show()
    
    
    


