import torch
import meshio
import subprocess
import os
from glob import glob
import torch
from .gauss import *
# from ..utils.grad import *
from torch_scatter import scatter


class TetMesh:
    '''
    A class to represent a tetrahedral mesh.
    '''

    def __init__(self, vertices: torch.Tensor, tets: torch.Tensor, order=1):
        '''
        Create a tetrahedral mesh.
        :param vertices: tensor of shape (num_vertices, 3) containing the vertex positions
        :param tets: tensor of shape (num_tets, 4) containing the tetrahedra when order=1;
        if order==2, the shape of tets should be (num_tets, 10), and order==3, tets shape should be (num_tets, 20)
        '''
        self.vertices = vertices
        self.tets = tets
        self.device = vertices.device
        self.order = order

    def __repr__(self):
        return 'TetMesh(vertices={}, tets={}, order={})'.format(self.vertices.shape, self.tets.shape, self.order)

    @property
    def transform_matrix(self):
        '''
        Return the transformation matrices for each tetrahedron.
        shape: (num_tets, 3, 3)
        '''
        if not hasattr(self, '_transform_matrix'):
            self._transform_matrix = self._compute_transform_matrix()
        return self._transform_matrix


    def _compute_transform_matrix(self):
        '''
        Compute the transformation matrices for each tetrahedron.
        '''
        A = torch.zeros(
            (self.tets.shape[0], 3, 3), dtype=torch.float32, device=self.vertices.device)
        if self.order == 1:
            v1 = self.vertices[self.tets[:, 0]]
            v2 = self.vertices[self.tets[:, 1]]
            v3 = self.vertices[self.tets[:, 2]]
            v4 = self.vertices[self.tets[:, 3]]
        elif self.order == 2:
            v1 = self.vertices[self.tets[:, 0]]
            v2 = self.vertices[self.tets[:, 2]]
            v3 = self.vertices[self.tets[:, 4]]
            v4 = self.vertices[self.tets[:, 9]]
        elif self.order == 3:
            v1 = self.vertices[self.tets[:, 0]]
            v2 = self.vertices[self.tets[:, 3]]
            v3 = self.vertices[self.tets[:, 6]]
            v4 = self.vertices[self.tets[:, 16]]
        A[:, 0, 0] = v1[:, 0] - v4[:, 0]
        A[:, 0, 1] = v2[:, 0] - v4[:, 0]
        A[:, 0, 2] = v3[:, 0] - v4[:, 0]
        A[:, 1, 0] = v1[:, 1] - v4[:, 1]
        A[:, 1, 1] = v2[:, 1] - v4[:, 1]
        A[:, 1, 2] = v3[:, 1] - v4[:, 1]
        A[:, 2, 0] = v1[:, 2] - v4[:, 2]
        A[:, 2, 1] = v2[:, 2] - v4[:, 2]
        A[:, 2, 2] = v3[:, 2] - v4[:, 2]
        return A
    
    def to_high_order(self, order):
        '''
        Convert the mesh to a higher order mesh.
        :param order: the order to convert to
        :return: a new TetMesh object
        '''

        # only support order=2,3 from order=1
        assert (self.order == 1)
        assert (order in [1, 2, 3])
        if (order == 1):
            return TetMesh(self.vertices, self.tets, order=1)
        num_tets = self.tets.shape[0]
        num_verts = self.vertices.shape[0]

        if (order == 2):
            # for each tet, add 6 new vertices, and add these vertices to the vertice list
            # tets shape should be (num_tets, 10)
            new_vertices = torch.zeros(
                (num_tets * 6 + num_verts, 3), dtype=self.vertices.dtype, device=self.device)
            new_vertices[:num_verts] = self.vertices
            new_tets = torch.zeros(
                (num_tets, 10), dtype=self.tets.dtype, device=self.device)

            vertices_full = self.vertices[self.tets]  # (num_tets, 4, 3)
            v1 = vertices_full[:, 0]  # (num_tets, 3)
            v3 = vertices_full[:, 1]  # (num_tets, 3)
            v5 = vertices_full[:, 2]  # (num_tets, 3)
            v10 = vertices_full[:, 3]  # (num_tets, 3)
            v2 = (v1 + v3) / 2
            v4 = (v3 + v5) / 2
            v6 = (v1 + v5) / 2
            v7 = (v1 + v10) / 2
            v8 = (v3 + v10) / 2
            v9 = (v5 + v10) / 2

            new_vertices[num_verts:] = torch.cat(
                [v2, v4, v6, v7, v8, v9], dim=0)
            new_tets[:, 0] = self.tets[:, 0]
            new_tets[:, 1] = torch.arange(
                num_verts, num_verts + num_tets, 1, dtype=torch.int)
            new_tets[:, 2] = self.tets[:, 1]
            new_tets[:, 3] = torch.arange(
                num_verts + num_tets, num_verts + 2 * num_tets, 1, dtype=torch.int)
            new_tets[:, 4] = self.tets[:, 2]
            new_tets[:, 5] = torch.arange(
                num_verts + 2 * num_tets, num_verts + 3 * num_tets, 1, dtype=torch.int)
            new_tets[:, 6] = torch.arange(
                num_verts + 3 * num_tets, num_verts + 4 * num_tets, 1, dtype=torch.int)
            new_tets[:, 7] = torch.arange(
                num_verts + 4 * num_tets, num_verts + 5 * num_tets, 1, dtype=torch.int)
            new_tets[:, 8] = torch.arange(
                num_verts + 5 * num_tets, num_verts + 6 * num_tets, 1, dtype=torch.int)
            new_tets[:, 9] = self.tets[:, 3]

            new_mesh = TetMesh(new_vertices, new_tets, order=2)

        # remove duplicate vertices
        new_mesh.remove_duplicate_vertices()
        return new_mesh

    def remove_duplicate_vertices(self):
        '''
        Remove duplicate vertices from a tetrahedral mesh.
        :param vertices: tensor of shape (num_vertices, 3) containing the vertex positions
        :param tets: tensor of shape (num_tets, N) containing the tetrahedra
        :return: a tuple (new_vertices, new_tets) containing the updated vertex and tetrahedra tensors
        '''
        # Calculate unique vertices
        # unique_vertices, sort_indices = torch.unique(
        #     self.vertices, dim=0, return_inverse=True)
        # unique_vertices, sort_indices = custom_unique(self.vertices)
        # Create new tets tensor with updated vertex indices
        unique_vertices, sort_indices = torch.unique(self.vertices, dim=0, return_inverse=True)
        origin_indices = torch.arange(self.vertices.shape[0], device=self.device)
        inverse_indices = scatter(origin_indices, sort_indices, dim=0, reduce="min")
        new_tets = sort_indices[self.tets]
        self.vertices = self.vertices[inverse_indices]
        self.tets = new_tets

    def export(self, filename):
        '''
        Export the tetrahedral mesh to a file format supported by meshio.
        :param filename: the filename to save the mesh to
        '''
        tets = self.tets.detach().cpu().numpy()
        vertices = self.vertices.detach().cpu().numpy()

        # Create a meshio mesh
        if self.order == 1:
            cells = [("tetra", tets)]
        elif self.order == 2:
            cells = [("tetra10", tets)]
        elif self.order == 3:
            cells = [("tetra20", tets)]

        meshio_mesh = meshio.Mesh(vertices, cells)

        # Save the mesh to a file
        meshio.write(filename, meshio_mesh, file_format="gmsh")

        print(f"Mesh saved to file {filename}")
