import torch
from .mesh import TetMesh
from .gauss import generate_gauss_points_weights
from .shape_func import get_shape_function, get_shape_function_grad
from torch_scatter import scatter


class Deform():
    def __init__(self, tetmesh: TetMesh):
        self.tetmesh = tetmesh
        self.device = tetmesh.device
        self.gauss_points, self.gauss_weights = generate_gauss_points_weights(
            tetmesh.order + 2)
        self.gauss_points = torch.tensor(self.gauss_points, dtype=torch.float32,
                                         device=tetmesh.vertices.device).to(self.device)  # (num_guass_points, 4)
        self.gauss_weights = torch.tensor(self.gauss_weights, dtype=torch.float32,
                                          device=tetmesh.vertices.device).to(self.device)  # (num_guass_points)
        self.num_guass_points = self.gauss_points.shape[0]
        self.num_nodes_per_tet = tetmesh.tets.shape[1]
        self.num_tets = tetmesh.tets.shape[0]

    @property
    def B_matrix(self):
        return self.shape_func_deriv

    @property
    def shape_func_deriv(self):
        '''
        shape_func_deriv:  (num_tets*num_guass_points, num_nodes_per_tet, 3)
        '''
        if not hasattr(self, '_shape_func_deriv'):
            self.precompute_shape_func_deriv()
        return self._shape_func_deriv

    def precompute_shape_func_deriv(self):
        '''
        precompute the shape function derivative of each tet
        _shape_func_deriv:  (num_tets*num_guass_points, num_nodes_per_tet, 3)
        '''
        A = self.tetmesh.transform_matrix  # (num_tets, 3, 3)
        A_inv = torch.inverse(A).unsqueeze(1).repeat(
            1, self.num_guass_points, 1, 1)  # (num_tets, num_guass_points, 3, 3)

        A_inv = A_inv.reshape(self.num_tets *
                              self.num_guass_points, 3, 3)  # (num_tets*num_guass_points, 3, 3)

        dL_dx = torch.tensor([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [-1, -1, -1]],
                             dtype=torch.float32,
                             device=self.device
                             ).unsqueeze(0).repeat(self.num_tets * self.num_guass_points, 1, 1
                                                   )  # (num_tets*num_guass_points, 4, 3)
        # (num_guass_points, num_nodes_per_tet, 4)
        dN_dL = get_shape_function_grad(self.gauss_points, self.tetmesh.order)

        # debug: dN_dL is based on the position of gauss points.
        # but the displacement in all the gauss point in a tet is the same!
        # should we

        dN_dL = dN_dL.unsqueeze(0).repeat(self.num_tets, 1, 1, 1).reshape(
            self.num_tets * self.num_guass_points, self.num_nodes_per_tet, 4
        )  # (num_tets*num_guass_points, num_nodes_per_tet, 4)

        # (num_tets*num_guass_points, num_nodes_per_tet, 3)
        B = (dN_dL @ dL_dx) @ A_inv
        self._shape_func_deriv = B

    def gradient_batch(self, u: torch.Tensor):
        '''
        u: (batch_num, num_nodes, 3), deformation or deformation speed of each node
        return: (batch_num, num_tets*num_guass_points, 3, 3)
        '''
        assert u.device == self.device

        # add: u maybe (batch_num, num_nodes, 3)
        if (len(u.shape) == 2):
            u = u.unsqueeze(0)  # (batch_num, num_nodes, 3)
        u = u[:, self.tetmesh.tets].transpose(
            2, 3)  # batch_num, num_tets, 3, num_nodes_per_tet)

        u = u.unsqueeze(2).repeat(1, 1, self.num_guass_points, 1, 1).reshape(
            -1, self.num_tets * self.num_guass_points, 3, self.num_nodes_per_tet)  # (batch_num, num_tets*num_guass_points, 3, num_nodes_per_tet
        # (batch_num, num_tets*num_guass_points, 3, 3)
        F = u @ self.shape_func_deriv
        return F

    # debug
    def gradient(self, u: torch.Tensor):
        '''
        u: (num_nodes, 3), deformation or deformation speed of each node
        return: (num_tets*num_guass_points, 3, 3)
        '''
        assert u.device == self.device
        u = u[self.tetmesh.tets].transpose(
            1, 2).float()  # (num_tets, 3, num_nodes_per_tet)
        # (num_tets*num_guass_points, 3, num_nodes_per_tet)
        u = u.unsqueeze(1).repeat(1, self.num_guass_points, 1, 1).reshape(
            self.num_tets * self.num_guass_points, 3, self.num_nodes_per_tet)  # (num_tets*num_guass_points, 3, num_nodes_per_tet
        F = u @ self.shape_func_deriv  # (num_tets*num_guass_points, 3, 3)
        return F

    @property
    def stress_index(self):
        '''
        stress_index: (num_tets*num_guass_points*num_nodes_per_tet*3)
        '''
        if not hasattr(self, '_stress_index'):
            self.precompute_stress_index()
        return self._stress_index

    def precompute_stress_index(self):
        '''
        precompute the index for stress computation
        _stress_index: (num_tets*num_guass_points*num_nodes_per_tet*3)
        '''
        index_base = self.tetmesh.tets.unsqueeze(1).repeat(
            1, self.num_guass_points, 1)  # (num_tets, num_guass_points, num_nodes_per_tet)
        index = torch.zeros(self.num_tets, self.num_guass_points, self.num_nodes_per_tet,
                            3, dtype=torch.long, device=self.device)
        index[:, :, :, 0] = index_base * 3
        index[:, :, :, 1] = index_base * 3 + 1
        index[:, :, :, 2] = index_base * 3 + 2
        self._stress_index = index.reshape(-1)

    @property
    def integration_weights(self):
        '''
        integration_weights: (num_tets*num_guass_points, 1, 1)
        '''
        if not hasattr(self, '_integration_weights'):
            self.precompute_integration_weights()
        return self._integration_weights

    def precompute_integration_weights(self):
        '''
        precompute the integration weights
        _integration_weights: (num_tets*num_guass_points, 1, 1)
        '''
        guass_weights = self.gauss_weights.unsqueeze(
            0)  # (1, num_guass_points)
        tets_volumes_r = torch.abs(torch.det(
            self.tetmesh.transform_matrix)).unsqueeze(1)  # (num_tets, 1)
        integration_weights = (
            guass_weights * tets_volumes_r).reshape(-1).unsqueeze(-1).unsqueeze(-1)  # (num_tets*num_guass_points, 1, 1)
        self._integration_weights = integration_weights

    def stress_to_force_batch(self, stress):
        '''
        stress: (batch_num, num_tets*num_guass_points, 3, 3)
        return: (batch_num, num_nodes*3)
        '''
        assert stress.device == self.device
        # (num_modes, num_tets*num_guass_points, 1, 1)
        weights = self.integration_weights
        # (num_modes, num_tets*num_guass_points, 3, num_nodes_per_tet)
        force = (stress @ self.shape_func_deriv.transpose(1, 2)) * weights
        force = force.transpose(2, 3).reshape(force.shape[0], -1)
        index = self.stress_index 
        # index = index.unsqueeze(0).repeat(force.shape[0], 1)
        # result = torch.zeros(force.shape[0], self.tetmesh.vertices.shape[0] * 3, device=self.device)
        # for i in range(force.shape[0]):
        #     result[i] = scatter(force[i], index, dim=0, reduce='sum')
        return scatter(force, index, dim=1, reduce='sum', dim_size=self.tetmesh.vertices.shape[0] * 3)
        # return result

    # debug
    def stress_to_force(self, stress):
        '''
        stress: (num_tets*num_guass_points, 3, 3)
        return: (num_nodes*3)
        '''
        assert stress.device == self.device
        weights = self.integration_weights  # (num_tets*num_guass_points, 1, 1)
        # (num_tets*num_guass_points, 3, num_nodes_per_tet)
        force = (stress @ self.shape_func_deriv.transpose(1, 2)) * weights
        force = force.transpose(1, 2).reshape(-1)
        index = self.stress_index
        return scatter(force, index, dim=0, reduce='sum')
