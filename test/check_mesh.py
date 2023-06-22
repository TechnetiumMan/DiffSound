import sys
sys.path.append('./')

from src.mesh import TetMesh
from src.shape_func import get_shape_function, get_shape_function_grad
from src.transform import compute_transform_coord, compute_inv_transform_coord
import torch
import numpy as np


def test_high_order(mesh=None):
    # Generate a simple tetrahedral mesh with 5 vertices and 2 tetrahedra
    if (mesh is None):
        vertices = torch.Tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        tets = torch.Tensor([[0, 1, 2, 3], [1, 4, 2, 3]]).long()

        # Create a TetMesh object
        mesh = TetMesh(vertices, tets)

        # Check that the mesh has the correct properties
        assert mesh.vertices.shape == (5, 3)
        assert mesh.tets.shape == (2, 4)
        assert mesh.order == 1

    # Convert to higher order mesh
    mesh2 = mesh.to_high_order(order=2)

    # Check that the new mesh has the correct properties
    # assert mesh2.vertices.shape == (14, 3)
    # assert mesh2.tets.shape == (2, 10)
    # assert mesh2.order == 2

    mesh3 = mesh.to_high_order(order=3)
    # there are 7 duplicated vertices: 6 edges and 1 face
    # assert mesh3.vertices.shape == (30, 3)
    # assert mesh3.tets.shape == (2, 20)
    # assert mesh3.order == 3

    # Check shape function of each order
    test_shape_function(mesh)
    test_shape_function(mesh2)
    test_shape_function(mesh3)

    # Check the transform function of each order
    test_transform(mesh)
    test_transform(mesh2)
    test_transform(mesh3)


def test_shape_function(mesh):
    # we use mesh generated in test_high_order() to test shape function:
    # sum of shape function in any point in any tets should be 1
    # we generate a random point in each tet and test the sum of shape function

    # the coord of vertice is in the transform_matrix of each tet
    # so we just need to generate a random point in a tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # for each tet, we generate 4 random number which sum is 1.
    # and the random number generated above is exactly the volume coord!

    vol_rd = torch.rand(mesh.tets.shape[0], 4)
    vol_rd = vol_rd / vol_rd.sum(dim=1, keepdim=True)  # (tets.shape[0], 4)

    # then get the shape function based on its order
    shape_func_rd = get_shape_function(vol_rd, mesh.order)

    # sum of shape function in any point in any tets should be 1
    sum_shape_func = shape_func_rd.sum(dim=1)
    assert (torch.allclose(sum_shape_func, torch.ones(sum_shape_func.shape)))

# TODO: check grad of shape function: dl


def test_transform(mesh):
    # we use mesh generated in test_high_order() to test transform function:
    # 1. for any point in any tets, its transformed coord should be the same as first 3 element of its volume coord
    # 2. its coord should be recovered by inverse transform
    # 3. when the tet translates as a whole, the point's coord should be translated as well

    # for 1, we generate a random point in each tet and get its transformed coord
    # so we need to get the vertex of each tet
    if mesh.order == 1:
        v1 = mesh.vertices[mesh.tets[:, 0]]  # (tets.shape[0], 3)
        v2 = mesh.vertices[mesh.tets[:, 1]]
        v3 = mesh.vertices[mesh.tets[:, 2]]
        v4 = mesh.vertices[mesh.tets[:, 3]]
    elif mesh.order == 2:
        v1 = mesh.vertices[mesh.tets[:, 0]]
        v2 = mesh.vertices[mesh.tets[:, 2]]
        v3 = mesh.vertices[mesh.tets[:, 4]]
        v4 = mesh.vertices[mesh.tets[:, 9]]
    elif mesh.order == 3:
        v1 = mesh.vertices[mesh.tets[:, 0]]
        v2 = mesh.vertices[mesh.tets[:, 3]]
        v3 = mesh.vertices[mesh.tets[:, 6]]
        v4 = mesh.vertices[mesh.tets[:, 16]]

    # then generate a random point in each tet, using the same method as above
    rd = torch.rand(mesh.tets.shape[0], 4, 1, device=mesh.vertices.device)
    rd = rd / rd.sum(dim=1, keepdim=True)  # (tets.shape[0], 4)
    rdv = rd[:, 0] * v1 + rd[:, 1] * v2 + rd[:, 2] * \
        v3 + rd[:, 3] * v4  # (tets.shape[0], 3)

    # get the transform matrix of each tet
    A = mesh.transform_matrix  # (tets.shape[0], 3, 3)

    # transform the random point
    rdv_hat = compute_transform_coord(rdv, A, v4)  # (tets.shape[0], 3)

    # calculate the volume coord
    # first, calculate the volume of the whole tet use its 4 vertices
    # (tets.shape[0],)
    V = torch.abs(torch.det(torch.stack([v1 - v4, v2 - v4, v3 - v4], dim=1)))
    # then calculate the volume of each sub tet
    V1 = torch.abs(torch.det(torch.stack([v2 - v4, v3 - v4, rdv - v4], dim=1)))
    V2 = torch.abs(torch.det(torch.stack([v1 - v4, v3 - v4, rdv - v4], dim=1)))
    V3 = torch.abs(torch.det(torch.stack([v1 - v4, v2 - v4, rdv - v4], dim=1)))
    # we only need first 3 element of vol_coord
    vol_coord = torch.stack([V1 / V, V2 / V, V3 / V], dim=1)

    # the transformed coord should be the same as first 3 element of its volume coord
    assert (torch.allclose(rdv_hat, vol_coord, atol=1e-5))

    # for 2, we inverse transform the transformed coord
    rdv_inv = compute_inv_transform_coord(rdv_hat, A, v4)  # (tets.shape[0], 3)
    assert (torch.allclose(rdv_inv, rdv, atol=1e-5))

    # # for 3, we translate the tet as a whole
    # translate_vec = torch.randn(3, device=rd.device) # (3,)
    # v1, v2, v3, v4 = v1 + translate_vec, v2 + translate_vec, v3 + translate_vec, v4 + translate_vec
    # rdv_trans = rd[:,0] * v1 + rd[:,1] * v2 + rd[:,2] * v3 + rd[:,3] * v4
    # rdv_hat_trans = compute_transform_coord(rdv_trans, A, v4)

    # # it should be the same as the transformed coord of the translated point
    # assert(torch.allclose(rdv_hat_trans, rdv_hat))


def test_wildtet():
    mesh = TetMesh.from_triangle_mesh('assets/sphere.obj')
    assert mesh.vertices.shape[1] == 3
    assert mesh.tets.shape[1] == 4
    assert mesh.order == 1
    test_high_order(mesh)


if __name__ == '__main__':
    test_high_order()
    test_wildtet()
    print('check_mesh passed!')
