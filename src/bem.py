import bempp.api.linalg
from bempp.api.operators import potential, boundary
from bempp.api import GridFunction, export, function_space
import numpy as np


def obj_to_grid(vertices, elements):
    import bempp.api
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    return bempp.api.Grid(vertices.T.astype(np.float64),
                          elements.T.astype(np.uint32))


class BEMModel():

    def __init__(self, vertices, elements):
        '''
        vertices: (n, 3) array
        elements: (m, 3) array
        '''
        self.grid = obj_to_grid(vertices, elements)
        self.dp0_space = function_space(self.grid, "DP", 0)
        self.dirichlet_fun = None
        self.neumann_fun = None

    def boundary_equation_solve(self, neumann_coeff, wave_number):
        '''
        neumann_coeff: (m) array
        wave_number: float
        '''
        self.k = wave_number
        neumann_fun = GridFunction(
            self.dp0_space, coefficients=np.asarray(neumann_coeff))
        self.neumann_fun = neumann_fun
        M = boundary.sparse.identity(
            self.dp0_space, self.dp0_space, self.dp0_space, precision="single", device_interface="numba")
        K = boundary.helmholtz.double_layer(
            self.dp0_space, self.dp0_space, self.dp0_space, self.k, precision="single", device_interface="numba")
        V = boundary.helmholtz.single_layer(
            self.dp0_space, self.dp0_space, self.dp0_space, self.k, precision="single", device_interface="numba")
        left_side = -0.5 * M + K
        right_side = V*self.neumann_fun
        dirichlet_fun, info, res = bempp.api.linalg.gmres(
            left_side, right_side, tol=1e-6, return_residuals=True)
        self.dirichlet_fun = dirichlet_fun

    def potential_solve(self, points):
        '''
        points: (n, 3) array
        '''
        potential_single = potential.helmholtz.single_layer(
            self.dp0_space, points.T, self.k, precision="single", device_interface="numba")
        potential_double = potential.helmholtz.double_layer(
            self.dp0_space, points.T, self.k, precision="single", device_interface="numba")
        dirichlet = - potential_single * self.neumann_fun + \
            potential_double * self.dirichlet_fun
        return dirichlet.reshape(-1)

    def export_neumann(self, filename):
        export(filename, grid_function=self.neumann_fun)

    def export_dirichlet(self, filename):
        export(filename, grid_function=self.dirichlet_fun)