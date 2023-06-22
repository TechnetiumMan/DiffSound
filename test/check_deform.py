import sys
sys.path.append('./')
import torch
from src.shape_func import get_shape_function, get_shape_function_grad
from src.mesh import TetMesh
from src.deform import Deform


def test_shape_func():
    vertices = torch.Tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).cuda()
    tets = torch.Tensor([[0, 1, 2, 3]]).long().cuda()
    mesh = TetMesh(vertices, tets)
    for order in range(1, 4):
        mesh_ = mesh.to_high_order(order)
        vs = mesh_.vertices[mesh_.tets[0]]
        L = torch.zeros(vs.shape[0], 4)
        L[:, :3] = vs
        L[:, 3] = 1 - L[:, 0] - L[:, 1] - L[:, 2]
        shape_func = get_shape_function(L, order)
        assert torch.allclose(shape_func, torch.eye(vs.shape[0]), atol=1e-3)


def test_shape_func_grad():
    for order in range(1, 4):
        L = torch.abs(torch.rand(10, 4)) / 4
        L[:, 0] = 1 - L[:, 1] - L[:, 2] - L[:, 3]
        shape_func = get_shape_function(L, order)
        EPS = 1e-4
        for i in range(4):
            L_ = L.clone()
            L_[:, i] += EPS
            shape_func_ = get_shape_function(L_, order)
            grad = (shape_func_ - shape_func) / EPS
            grad_ = get_shape_function_grad(L, order)[..., i]
            # print('grad', grad)
            # print('grad_', grad_)
            assert torch.allclose(grad, grad_, atol=1e-2)


def test_deform(mesh=None):
    if (mesh is None):
        vertices = torch.Tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]]).cuda()
        tets = torch.Tensor([[0, 1, 2, 3], [4, 5, 6, 7]]).long().cuda()
        # vertices = torch.Tensor(
        #     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()
        # tets = torch.Tensor([[0, 1, 2, 3]]).long().cuda()
        mesh = TetMesh(vertices, tets)

    for order in range(1, 4):
        mesh_ = mesh.to_high_order(order)
        deform = Deform(mesh_)
        deform_matrix = torch.rand(3, 3).cuda()
        u = mesh_.vertices @ deform_matrix + torch.rand(3).cuda()
        grad_u = deform.gradient(u)
        grad_gt = deform_matrix.T.unsqueeze(0).repeat(grad_u.shape[0], 1, 1)
        assert torch.allclose(grad_u, grad_gt, atol=1e-2)


if __name__ == '__main__':
    test_shape_func()
    test_shape_func_grad()
    mesh = TetMesh.from_triangle_mesh('assets/bowl.obj')
    test_deform(mesh)
    print('check_deform passed')
