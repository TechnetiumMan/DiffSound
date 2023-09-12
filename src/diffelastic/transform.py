import torch
def compute_transform_coord(p, A, b):
    '''
    compute the transform coordinates of points p in tetrahedrons which transformation matrix is A
    :param p: tensor of shape (num_points, 3) containing the coordinates of points
    :param A: tensor of shape (num_points, 3, 3) containing the transformation matrix of each tetrahedron
    :param b: tensor of shape (num_points, 3) which is coord of v4 of each tetrahedron
    notice that the tet of each index of A contains the point p of the same index
    :return: tensor of shape (num_points, 3) containing the transformed coordinates of points p
    '''
    
    A_inv = torch.inverse(A)
    p_hat = torch.bmm(A_inv, (p - b).unsqueeze(-1)).squeeze(-1)
    return p_hat
    
def compute_inv_transform_coord(p_hat, A, b):
    '''
    compute the inverse transform coordinates of points p in tetrahedrons which transformation matrix is A
    :param p: tensor of shape (num_points, 3) containing the coordinates of points
    :param A: tensor of shape (num_points, 3, 3) containing the transformation matrix of each tetrahedron
    :param b: tensor of shape (num_points, 3) which is coord of v4 of each tetrahedron
    notice that the tet of each index of A contains the point p of the same index
    :return: tensor of shape (num_points, 3) containing the inverse transform coordinates of points p
    '''
    
    p = torch.bmm(A, p_hat.unsqueeze(-1)).squeeze(-1) + b
    return p