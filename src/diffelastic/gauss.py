import numpy as np
from numpy.polynomial.legendre import Legendre, legroots

def calculate_legendre_roots_weights(order): 
    coeffs = np.zeros(order+1, dtype=np.float64) # initialize coefficients to zero
    coeffs[-1] = 1 # set last coefficient to 1
    
    Pn = Legendre(coeffs)
    roots = legroots(coeffs)
    
    Pn_deriv = Pn.deriv()
    Pn_deriv_val = Pn_deriv(roots)
    weights = 2 / ((1 - roots**2) * Pn_deriv_val**2)

    return roots, weights

def generate_gauss_points_weights(order):
    roots, weights = calculate_legendre_roots_weights(order) # [-1, 1]
    roots = (roots + 1) / 2 # [0, 1]
    
    # generate 3D points in a tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    x = np.zeros(order**3, dtype=np.float64)
    y = np.zeros(order**3, dtype=np.float64)
    z = np.zeros(order**3, dtype=np.float64)
    w = np.zeros(order**3, dtype=np.float64)
    weight_p = np.zeros(order**3, dtype=np.float64) # weight for every point
    for i in range(order):
        for j in range(order):
            for k in range(order):
                idx = i*(order**2) + j*order + k
                w[idx] = roots[i]
                z[idx] = roots[j] * (1-w[idx])
                y[idx] = roots[k] * (1-w[idx]-z[idx])
                x[idx] = 1 - w[idx] - z[idx] - y[idx]
                weight_p[idx] = weights[i] * weights[j] * weights[k] * (1-w[idx]) * (1-w[idx]-z[idx]) / 8
                
    points = np.stack([x, y, z, w], axis=1)
    return points, weight_p