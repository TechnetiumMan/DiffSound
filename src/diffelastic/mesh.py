import torch
import meshio
import subprocess
import os
from glob import glob
import torch
from .gauss import *


class TetMesh:
    '''
    A class to represent a tetrahedral mesh.
    '''

    def __init__(self, vertices: torch.Tensor, tets: torch.Tensor):
        '''
        Create a tetrahedral mesh.
        :param vertices: tensor of shape (num_vertices, 3) containing the vertex positions
        :param tets: tensor of shape (num_tets, 4) containing the tetrahedra when order=1;
        if order==2, the shape of tets should be (num_tets, 10), and order==3, tets shape should be (num_tets, 20)
        '''
        self.vertices = vertices
        self.tets = tets
        self.device = vertices.device
        self.order = 1

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

    def remove_duplicate_vertices(self):
        '''
        Remove duplicate vertices from a tetrahedral mesh.
        :param vertices: tensor of shape (num_vertices, 3) containing the vertex positions
        :param tets: tensor of shape (num_tets, N) containing the tetrahedra
        :return: a tuple (new_vertices, new_tets) containing the updated vertex and tetrahedra tensors
        '''
        # Calculate unique vertices
        unique_vertices, sort_indices = torch.unique(
            self.vertices, dim=0, return_inverse=True)
        # Create new tets tensor with updated vertex indices
        new_tets = sort_indices[self.tets]
        self.vertices = unique_vertices
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
