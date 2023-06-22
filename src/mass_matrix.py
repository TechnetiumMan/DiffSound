import sys
sys.path.append('./')
import numpy as np
from numpy.polynomial.legendre import Legendre, legroots
from src.shape_func import get_shape_function
import torch
from .gauss import *

def calculate_element_mass_matrix(fem_order, gauss_order):
    points, weights = generate_gauss_points_weights(gauss_order)
    points = torch.from_numpy(points).cuda()
    weights = torch.from_numpy(weights).cuda()
    
    shape_func = get_shape_function(points, fem_order) # (4/10/20, gauss_points)
    vnum = shape_func.shape[1] 
    M = torch.zeros((vnum, vnum), dtype=torch.float64).cuda() # element mass matrix
    for v1 in range(vnum):
        for v2 in range(vnum):
            N1 = shape_func[:, v1]
            N2 = shape_func[:, v2]
            sum = torch.sum(N1 * N2 * weights)
            M[v1, v2] = sum
    return M

def get_elememt_mass_matrix(fem_order):
    M = calculate_element_mass_matrix(fem_order, fem_order + 2)
    M = M[:, :, None, None] * torch.eye(3).cuda()
    M = M.transpose(1, 2)
    # M = M.reshape(12, 12)
    M = M.reshape(-1)
    return M
                
if __name__ == "__main__":
    for order in range(1, 4):
        M = get_elememt_mass_matrix(order).cpu()
        print(M)

    