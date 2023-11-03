import torch
from tqdm import tqdm
import open3d as o3d
import numpy as np
import sys
import os
sys.path.append("./")
from src.diffelastic.diff_model import TetMesh

def train_sdfnerf(geometry, init_mesh, FLAGS):
    mesh = o3d.io.read_triangle_mesh(init_mesh)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # min_bound = mesh.vertex.positions.min(0).numpy()
    # max_bound = mesh.vertex.positions.max(0).numpy()

    # # add margin
    # size = max_bound - min_bound
    # min_bound -= size * 0.1
    # max_bound += size * 0.1

    # xyz_range = np.linspace(min_bound, max_bound, num=32)
    xyz_range = np.linspace([-1, -1, -1], [1, 1, 1], num=32)
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32) # (-1.2, 1.2)
    
    signed_distance = scene.compute_signed_distance(query_points)

    # scale = FLAGS.sdf_scale # (-2, 2)
    # radius = 0.8
    # query_points = (query_points - min_bound) / (max_bound - min_bound) - 0.5 # (-0.5, 0.5)
    # query_points *= scale # (-0.1, 0.1)
    query_points = torch.from_numpy(query_points).cuda().reshape(-1, 3)
    signed_distance = -torch.from_numpy(signed_distance.numpy()).cuda().reshape(-1)
    # margin = scale * 0.02
    margin = 0.01
    
    optimizer = torch.optim.Adam(geometry.parameters(), lr=2e-5)
    TetMesh(vertices=geometry.verts, tets=geometry.indices).export(os.path.join(FLAGS.out_dir, "dmtet_base.msh"))
    
    for i in tqdm(range(FLAGS.pre_iter)):
        loss = geometry.mesh_template_loss(query_points, signed_distance, margin) # query points:(-1.2, 1.2) -> (-0.1, 0.1) scale:1/12
        optimizer.zero_grad()
        if loss is not None:
            loss.backward()
        else:
            break
        optimizer.step()
        if i % 20 == 0:
            print("pre_iter:{}, loss: {}".format(i, loss))
    print("pre_iter:{}, loss: {}".format(FLAGS.pre_iter, loss))
    
    verts, tets = geometry.getMesh(None)
    print(verts.shape, tets.shape)
    TetMesh(vertices=verts, tets=tets).export(os.path.join(FLAGS.out_dir, "dmtet_pre.msh"))
    
    return geometry