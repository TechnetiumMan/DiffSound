import sys
sys.path.append("./")
import torch
import numpy as np
from src.dmtet.dmtet import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh
import open3d as o3d
from tqdm import tqdm
import meshio

# generate ground truth audio(eigenvalues)
# assume that the gt object are scaled, and we want to train the scale rate
gt_scale = 0.8
mesh = meshio.read("data/sphere.msh")
verts = torch.Tensor(mesh.points).cuda() * gt_scale
tets = torch.Tensor(mesh.cells[0].data).long().cuda()
print('Load tetramesh with ', len(verts),
        ' vertices & ', len(tets), ' tets')

gt_obj = DiffSoundObj(verts, tets, mode_num=32, order=1)
gt_obj.eigen_decomposition()
gt_vals = gt_obj.get_vals()
print(gt_vals)

# load initial mesh
# armadillo_data = o3d.data.ArmadilloMesh()
mesh = o3d.io.read_triangle_mesh("data/sphere.ply")
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)
min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()

# add margin
size = max_bound - min_bound
min_bound -= size * 0.1
max_bound += size * 0.1

xyz_range = np.linspace(min_bound, max_bound, num=32)
query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32) # (-1.2, 1.2)
signed_distance = scene.compute_signed_distance(query_points)

scale = 0.2 # (-2, 2)
scale_rate = 2.4 / scale # scale the mesh from DMTet to original scale
# radius = 0.8
query_points = (query_points - min_bound) / (max_bound - min_bound) - 0.5 # (-0.5, 0.5)
query_points *= scale # (-0.1, 0.1)
query_points = torch.from_numpy(query_points).cuda().reshape(-1, 3)
signed_distance = -torch.from_numpy(signed_distance.numpy()).cuda().reshape(-1)
# SDF in this implementation is positive inside the mesh

res = 32
pre_iter = 500
num_iter = 1000
DMTet = DMTetGeometry(res=res, scale=scale, freq_num=3).cuda()
optimizer = torch.optim.Adam(DMTet.parameters(), lr=2e-5)
TetMesh(vertices=DMTet.verts, tets=DMTet.indices).export("output/dmtet_base.msh")

margin = scale * 0.02
for i in tqdm(range(pre_iter)):
    loss = DMTet.mesh_template_loss(query_points, signed_distance, margin) # query points:(-1.2, 1.2) -> (-0.1, 0.1) scale:1/12
    optimizer.zero_grad()
    if loss is not None:
        loss.backward()
    else:
        break
    optimizer.step()
    if i % 20 == 0:
        print("pre_iter:{}, loss: {}".format(i, loss))
print("pre_iter:{}, loss: {}".format(pre_iter, loss))

verts, tets = DMTet.getMesh()
print(verts.shape, tets.shape)
TetMesh(vertices=verts, tets=tets).export("output/dmtet_pre.msh")

optimizer = torch.optim.Adam(DMTet.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

for i in range(num_iter):
    verts, tets = DMTet.getMesh()
    verts, tets = DMTet.get_largest_connected_component(verts, tets)
    verts = verts * scale_rate # scale the mesh back
    if i % 10 == 0:
        torch.save([verts, tets], "output/dmtet_{}.pth".format(i))
    print(verts.shape, tets.shape)
    loss1 = DMTet.mesh_template_loss(query_points, signed_distance, margin)
    if loss1 is None:
        loss1 = torch.tensor(0.0).cuda()
    print("iter:{}, loss1: {}".format(i, loss1))
    obj = DiffSoundObj(verts, tets, mode_num=32, order=1)
    obj.eigen_decomposition()
    vals = obj.get_vals()
    print(vals)
    loss2 = ((vals - gt_vals) ** 2 / vals**2).mean()
    # loss2 = ((vals[0] - 40000) ** 2 / vals[0]**2)
    print("iter:{}, loss_2: {}".format(i, loss2))
    loss = loss1 + loss2 * 1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    TetMesh(vertices=verts, tets=tets).export("output/dmtet.msh")
    if i % 10 == 0:
        torch.save([verts, tets], "output/dmtet_{}.pth".format(i))
        
