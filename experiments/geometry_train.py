# Geometric Shape Estimation experiment
# This experiment sets the implicit neural SDF as learnable parameters, 
# and uses a coarse voxel as constraint, 
# aiming to restore a more detailed shape from its mode eigenvalues.

import sys
sys.path.append("./")
import torch
import numpy as np
from src.dmtet.geometry.dmtet_geometry import DMTetGeometry
from src.diffelastic.diff_model import DiffSoundObj, TetMesh
import open3d as o3d
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import meshio
import torch
from numba import njit
import argparse
import json

cube_vertex = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


cube_normal = np.array(
    [
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0],
    ]
)
cube_faces = (
    np.array(
        [
            [1, 7, 5],
            [1, 3, 7],
            [1, 4, 3],
            [1, 2, 4],
            [3, 8, 7],
            [3, 4, 8],
            [5, 7, 8],
            [5, 8, 6],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 8],
            [2, 8, 4],
        ]
    )
    - 1
)
cube_faces_normal_index = np.array([2, 2, 6, 6, 3, 3, 5, 5, 4, 4, 1, 1]) - 1


@njit()
def voxel2boundary(coords, resolution=32):
    ds = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    coords = coords + 1
    res = resolution + 2
    voxel = np.zeros((res, res, res))
    voxel[0, 0, 0] = 1
    coords_n = len(coords)
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0], c[1], c[2]] = 2
    points = [np.array([0, 0, 0])]
    while len(points) > 0:
        p = points.pop()
        for d_idx in range(6):
            d = ds[d_idx]
            p_ = p + d
            if (p_ >= 0).all() and (p_ < res).all() and voxel[p_[0], p_[1], p_[2]] != 1:
                if voxel[p_[0], p_[1], p_[2]] == 0:
                    points.append(p_)
                voxel[p_[0], p_[1], p_[2]] = 1
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0], c[1], c[2]] = 2
    vertex_flag = np.zeros((res, res, res)) - 1

    vertices = []
    elements = []
    feats_index = []
    for i in range(len(coords)):
        for j in range(len(cube_faces)):
            c = coords[i]
            face = cube_faces[j]
            normal = cube_normal[cube_faces_normal_index[j]]
            c_ = normal + c
            if voxel[c_[0], c_[1], c_[2]] == 1:
                element = []
                for vertex_idx in face:
                    v = cube_vertex[vertex_idx] + c
                    if vertex_flag[v[0], v[1], v[2]] == -1:
                        vertex_flag[v[0], v[1], v[2]] = len(vertices)
                        vertices.append(v - 1)
                    element.append(vertex_flag[v[0], v[1], v[2]])
                elements.append(element)
                feats_index.append(i)
    return vertices, elements, feats_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    FLAGS = parser.parse_args()

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]
            
    print("Config / Flags:")
    print("---------")
    for key in FLAGS.__dict__.keys():
        print(key, FLAGS.__dict__[key])
    print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    
    num_iter = FLAGS.iter
    grid_res = FLAGS.grid_res
    out_dir_base = FLAGS.out_dir
    freq_num = FLAGS.freq_num
    best_loss_dict = {}
    for voxel_num in FLAGS.voxel_num_list:
        out_dir = f"{out_dir_base}/{voxel_num}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        writer = SummaryWriter(out_dir)
        for model_name in FLAGS.mesh_name_list:
            # load ground truth
            mesh = TetMesh().import_from_file(f"{FLAGS.init_mesh_dir}{model_name}.msh")
            # print(mesh.vertices.shape, mesh.tets.shape)
            soundObj = DiffSoundObj(mesh.vertices, mesh.tets, mode_num=64)
            soundObj.eigen_decomposition()
            gt_vals = soundObj.get_vals()

            # load voxelized mesh
            mesh = o3d.io.read_triangle_mesh(f"{FLAGS.init_mesh_dir}{model_name}_surf.obj")
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh)
            min_bound = mesh.vertex.positions.min(0).numpy()
            max_bound = mesh.vertex.positions.max(0).numpy()
            center = (min_bound + max_bound) / 2
            size = (max_bound - min_bound).max()
            min_bound = center - size / 2 * 1.05
            max_bound = center + size / 2 * 1.05
            size = (max_bound - min_bound).max()
            mesh = mesh.translate(-center)
            mesh = mesh.to_legacy()
            o3d.io.write_triangle_mesh(f"{out_dir}/{model_name}_surf_aligned.obj", mesh)
            xyz_range = np.linspace(min_bound, max_bound, num=voxel_num)
            query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
            signed_distance = scene.compute_signed_distance(query_points)
            query_points = (query_points - min_bound) / size - 0.5
            voxel_points = query_points[signed_distance.numpy() < 0]
            voxel_points = ((voxel_points + 0.5) * voxel_num).astype(np.int32)
            voxel_verts, voxel_faces, _ = voxel2boundary(voxel_points, resolution=voxel_num)
            voxel_verts = (
                np.array(voxel_verts).astype(np.float32) / voxel_num * size
                + min_bound
                - center
            )
            voxel_faces = np.array(voxel_faces).astype(np.int32)
            # write voxel points to obj
            meshio_mesh = meshio.Mesh(voxel_verts, [("triangle", voxel_faces)])
            meshio.write(f"{out_dir}/{model_name}_voxel.obj", meshio_mesh)

            query_points = torch.from_numpy(query_points).cuda().reshape(-1, 3)
            signed_distance = -torch.from_numpy(signed_distance.numpy()).cuda().reshape(-1)
            # SDF in this implementation is positive inside the mesh

            pre_iter = 2000
            DMTet = DMTetGeometry(res=grid_res, scale=size, freq_num=freq_num).cuda()
            optimizer = torch.optim.Adam(DMTet.parameters(), lr=0.0001)

            margin = 0
            for i in tqdm(range(pre_iter)):
                loss = DMTet.mesh_template_loss(query_points, signed_distance, margin)
                optimizer.zero_grad()
                if loss is not None:
                    loss.backward()
                else:
                    break
                optimizer.step()
            print("pre_iter:{}, loss: {}".format(pre_iter, loss))

            verts, tets = DMTet.getMesh()
            print(verts.shape, tets.shape)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            TetMesh(vertices=verts, tets=tets).export(f"{out_dir}/{model_name}_before.msh")
            torch.save(DMTet.state_dict(), f"{out_dir}/{model_name}.pt")

            for check_mode_num in FLAGS.mode_num_list:
                DMTet.load_state_dict(torch.load(f"{out_dir}/{model_name}.pt"))
                optimizer = torch.optim.Adam(DMTet.parameters(), lr=FLAGS.learning_rate)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=100, gamma=0.8
                )
                best_loss = 100000
                try:
                    for i in tqdm(range(num_iter)):
                        verts, tets = DMTet.getMesh()
                        verts, tets = DMTet.get_largest_connected_component(verts, tets)
                        tetmesh = TetMesh(vertices=verts, tets=tets)
                        vols = torch.abs(torch.det(tetmesh.transform_matrix))
                        tets = tets[vols > 0]
                        # print(verts.shape, tets.shape)
                        loss1 = DMTet.mesh_template_loss(
                            query_points, signed_distance, margin
                        )
                        if loss1 is None:
                            loss1 = torch.tensor(0.0).cuda()
                        print("iter:{}, loss1: {}".format(i, loss1))
                        obj = DiffSoundObj(verts, tets, mode_num=check_mode_num)
                        obj.eigen_decomposition()
                        vals = obj.get_vals()
                        # print(gt_vals)
                        # print(vals)
                        loss2 = (
                            ((vals - gt_vals[:check_mode_num]) ** 2)
                            / gt_vals[:check_mode_num] ** 2
                        ).mean() ** 0.5
                        print("iter:{}, loss_2: {}".format(i, loss2.item()))
                        writer.add_scalar(f"{model_name}_{check_mode_num}", loss2, i)
                        loss = loss1 + loss2 * 0.0002
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            TetMesh(vertices=verts, tets=tets).export(
                                f"{out_dir}/{model_name}_{check_mode_num}.msh"
                            )
                            best_loss_dict[f"{model_name}_{check_mode_num}"] = loss2.item()
                            torch.save(best_loss_dict, f"{out_dir}/best_loss.pt")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                except Exception as e:
                    print(e)
        writer.close()
