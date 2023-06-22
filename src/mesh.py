import torch
import meshio
import subprocess
import os
from glob import glob


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
        self.order = order
        self.device = vertices.device

    def __repr__(self):
        return 'TetMesh(vertices={}, tets={}, order={})'.format(self.vertices.shape, self.tets.shape, self.order)

    def from_triangle_mesh(filename, log=False):
        '''
        Create a tetrahedral mesh from a triangle mesh.
        '''
        if not os.path.exists(filename + "_.msh"):
            mesh = meshio.read(filename)
            # check if the mesh is a triangle mesh
            assert (mesh.cells[0].type == 'triangle')

            # convert the triangle mesh to a tetrahedral mesh
            # need to install FloatTetWild first
            result = subprocess.run(
                ["FloatTetwild_bin", "-i", filename, "--max-threads", "8", "--stop-energy", "90.0", "--coarsen"], \
                    capture_output=True, text=True)
            if log:
                print(result.stdout, result.stderr)

            # if FloatTetWild not installed:
            # vertices = mesh.points
            # faces = mesh.cells["triangle"]
            # tet_mesh = meshio.Mesh()
            # tet_mesh.points, tet_mesh.cells = meshio.tetgen.build(
            #     dict(vertices=vertices, faces=faces)
            # )
        mesh = meshio.read(filename + "_.msh")
        # remove the intermediate file
        vertices = torch.Tensor(mesh.points).cuda()
        tets = torch.Tensor(mesh.cells[0].data).long().cuda()
        print('Load tetramesh with ', len(vertices),
              ' vertices & ', len(tets), ' tets')
        return TetMesh(vertices, tets)

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

        elif (order == 3):
            new_vertices = torch.zeros(
                (num_tets * 16 + num_verts, 3), dtype=self.vertices.dtype, device=self.device)
            new_vertices[:num_verts] = self.vertices
            new_tets = torch.zeros(
                (num_tets, 20), dtype=self.tets.dtype, device=self.device)

            vertices_full = self.vertices[self.tets]  # (num_tets, 4, 3)
            v1 = vertices_full[:, 0]  # (num_tets, 3)
            v4 = vertices_full[:, 1]  # (num_tets, 3)
            v7 = vertices_full[:, 2]  # (num_tets, 3)
            v17 = vertices_full[:, 3]  # (num_tets, 3)

            v2 = (v1 * 2 + v4) / 3
            v3 = (v1 + v4 * 2) / 3
            v5 = (v4 * 2 + v7) / 3
            v6 = (v4 + v7 * 2) / 3
            v8 = (v7 * 2 + v1) / 3
            v9 = (v7 + v1 * 2) / 3
            v10 = (v1 + v4 + v7) / 3
            v11 = (v1 * 2 + v17) / 3
            v12 = (v4 * 2 + v17) / 3
            v13 = (v7 * 2 + v17) / 3
            v14 = (v1 + v17 * 2) / 3
            v15 = (v4 + v17 * 2) / 3
            v16 = (v7 + v17 * 2) / 3
            v18 = (v4 + v7 + v17) / 3
            v19 = (v1 + v7 + v17) / 3
            v20 = (v1 + v4 + v17) / 3

            new_vertices[num_verts:] = torch.cat([v2, v3, v5, v6, v8, v9, v10, v11,
                                                  v12, v13, v14, v15, v16, v18, v19, v20], dim=0)
            new_tets[:, 0] = self.tets[:, 0]
            new_tets[:, 1] = torch.arange(
                num_verts, num_verts + num_tets, 1, dtype=torch.int)
            new_tets[:, 2] = torch.arange(
                num_verts + num_tets, num_verts + 2 * num_tets, 1, dtype=torch.int)
            new_tets[:, 3] = self.tets[:, 1]
            new_tets[:, 4] = torch.arange(
                num_verts + 2 * num_tets, num_verts + 3 * num_tets, 1, dtype=torch.int)
            new_tets[:, 5] = torch.arange(
                num_verts + 3 * num_tets, num_verts + 4 * num_tets, 1, dtype=torch.int)
            new_tets[:, 6] = self.tets[:, 2]
            for i in range(7, 16):
                new_tets[:, i] = torch.arange(
                    num_verts + (i - 3) * num_tets, num_verts + (i - 2) * num_tets, 1, dtype=torch.int)
            new_tets[:, 16] = self.tets[:, 3]
            for i in range(17, 20):
                new_tets[:, i] = torch.arange(
                    num_verts + (i - 4) * num_tets, num_verts + (i - 3) * num_tets, 1, dtype=torch.int)

            new_mesh = TetMesh(new_vertices, new_tets, order=3)

        # remove duplicate vertices
        new_mesh.remove_duplicate_vertices()
        return new_mesh

    @property
    def transform_matrix(self):
        '''
        Return the transformation matrices for each tetrahedron.
        shape: (num_tets, 3, 3)
        '''
        if not hasattr(self, '_transform_matrix'):
            self._transform_matrix = self._compute_transform_matrix()
        return self._transform_matrix

    def compute_mass_matrix(self, density):
        '''
        Return the mass matrix of the mesh(as a coo-sparse matrix).
        '''

        from src.mass_matrix import get_elememt_mass_matrix
        from src.cuda_module import mass_matrix_assembler

        msize_list = [12, 30, 60]
        msize = msize_list[self.order - 1]
        values = torch.zeros((msize * msize * self.tets.shape[0]), dtype=torch.float64).cuda()
        rows = torch.zeros_like(values, dtype=torch.int32).cuda()
        cols = torch.zeros_like(values, dtype=torch.int32).cuda()
        vertices_cuda = self.vertices.to(
            torch.float64).reshape(-1).contiguous().cuda()
        tets_cuda = self.tets.to(torch.int32).reshape(-1).contiguous().cuda()
        element_mm = get_elememt_mass_matrix(self.order)
        mass_matrix_assembler(vertices_cuda, tets_cuda, values,
                              rows, cols, element_mm, density, self.order)

        # debug: change data type to float64
        # values = values.double()

        indices = torch.stack([rows, cols], dim=0).long()
        shape = torch.Size(
            [3 * self.vertices.shape[0], 3 * self.vertices.shape[0]])
        mass_matrix = torch.sparse_coo_tensor(indices, values, shape)
        return mass_matrix.coalesce()

    def _compute_transform_matrix(self):
        '''
        Compute the transformation matrices for each tetrahedron.
        '''
        A = torch.zeros(
            (self.tets.shape[0], 3, 3), dtype=torch.float64, device=self.vertices.device)
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
