import torch

def get_shape_function(L, order=1):
    '''
    get the shape function of a set of volume coord according to the order of the mesh
    :param L: tensor of shape (n,4) containing the volume coord
    :return: if order==1, return shape function(n,4); 
    if order==2, return shape function(n,10); 
    if order==3, return shape function(n,20)
    '''
    L1, L2, L3, L4 = L[:,0], L[:,1], L[:,2], L[:,3]
    if(order == 1):
        return L
    elif(order == 2):
        sf = torch.stack([L1*(2*L1-1),
                        4*L1*L2,
                        L2*(2*L2-1),
                        4*L2*L3,
                        L3*(2*L3-1), # 5
                        4*L3*L1,
                        4*L1*L4,
                        4*L2*L4,
                        4*L3*L4,
                        L4*(2*L4-1) # 10
                        ])
        return sf.transpose(0,1)
    elif(order == 3):
        sf = torch.stack([1/2*(3*L1-1)*(3*L1-2)*L1,
                        9/2*L1*L2*(3*L1-1),
                        9/2*L1*L2*(3*L2-1),
                        1/2*(3*L2-1)*(3*L2-2)*L2,
                        9/2*L2*L3*(3*L2-1), # 5
                        9/2*L2*L3*(3*L3-1),
                        1/2*(3*L3-1)*(3*L3-2)*L3,
                        9/2*L3*L1*(3*L3-1),
                        9/2*L3*L1*(3*L1-1),
                        27*L1*L2*L3, # 10
                        9/2*L1*L4*(3*L1-1),
                        9/2*L2*L4*(3*L2-1),
                        9/2*L3*L4*(3*L3-1),
                        9/2*L1*L4*(3*L4-1),
                        9/2*L2*L4*(3*L4-1), # 15
                        9/2*L3*L4*(3*L4-1),
                        1/2*(3*L4-1)*(3*L4-2)*L4,
                        27*L2*L3*L4,
                        27*L1*L3*L4,
                        27*L1*L2*L4 # 20
                        ])
        return sf.transpose(0,1)
        
def get_shape_function_grad(L, order=1):
    '''
    get the gradient of the shape function of each volume coord
    :param L: tensor of shape (n,4) containing the volume coord
    :return: if order==1, return shape function(n,4); 
    if order==2, return shape function(n,10); 
    if order==3, return shape function(n,20)
    '''
    
    # IT IS REALLY TEDIOUS!!!
    L1, L2, L3, L4 = L[:,0], L[:,1], L[:,2], L[:,3]
    one = torch.ones_like(L1)
    zero = torch.zeros_like(L1)
    
    if(order == 1):
        return torch.stack([
                            torch.stack([one, zero, zero, zero]),
                            torch.stack([zero, one, zero, zero]),
                            torch.stack([zero, zero, one, zero]),
                            torch.stack([zero, zero, zero, one])
                            ]).permute(2,0,1)
    elif(order == 2):
        return torch.stack([    
                            torch.stack([4*L1-one, zero, zero, zero]),
                            torch.stack([4*L2, 4*L1, zero, zero]),
                            torch.stack([zero, 4*L2-one, zero, zero]),
                            torch.stack([zero, 4*L3, 4*L2, zero]),
                            torch.stack([zero, zero, 4*L3-one, zero]), # 5
                            torch.stack([4*L3, zero, 4*L1, zero]),
                            torch.stack([4*L4, zero, zero, 4*L1]),
                            torch.stack([zero, 4*L4, zero, 4*L2]),
                            torch.stack([zero, zero, 4*L4, 4*L3]),
                            torch.stack([zero, zero, zero, 4*L4-one]) # 1zero
                            ]).permute(2,0,1)
        
    elif(order == 3):
        return torch.stack([
                            torch.stack([27/2*L1*L1-9*L1+one, zero, zero, zero]),
                            torch.stack([(27*L1-9/2)*L2, 9/2*L1*(3*L1-one), zero, zero]),
                            torch.stack([9/2*L2*(3*L2-one), (27*L2-9/2)*L1, zero, zero]),
                            torch.stack([zero, 27/2*L2*L2-9*L2+one, zero, zero]),
                            torch.stack([zero, (27*L2-9/2)*L3, 9/2*L2*(3*L2-one), zero]), # 5
                            torch.stack([zero, 9/2*L3*(3*L3-one), (27*L3-9/2)*L2, zero]),
                            torch.stack([zero, zero, 27/2*L3*L3-9*L3+one, zero]),
                            torch.stack([9/2*L3*(3*L3-one), zero, (27*L3-9/2)*L1, zero]),
                            torch.stack([(27*L1-9/2)*L3, zero, 9/2*L1*(3*L1-one), zero]),
                            torch.stack([27*L2*L3, 27*L1*L3, 27*L1*L2, zero]), # 1zero
                            torch.stack([(27*L1-9/2)*L4, zero, zero, 9/2*L1*(3*L1-one)]),
                            torch.stack([zero, (27*L2-9/2)*L4, zero, 9/2*L2*(3*L2-one)]),
                            torch.stack([zero, zero, (27*L3-9/2)*L4, 9/2*L3*(3*L3-one)]),
                            torch.stack([9/2*L4*(3*L4-one), zero, zero, (27*L4-9/2)*L1]),
                            torch.stack([zero, 9/2*L4*(3*L4-one), zero, (27*L4-9/2)*L2]), # 15
                            torch.stack([zero, zero, 9/2*L4*(3*L4-one), (27*L4-9/2)*L3]), 
                            torch.stack([zero, zero, zero, 27/2*L4*L4-9*L4+one]),
                            torch.stack([zero, 27*L3*L4, 27*L2*L4, 27*L2*L3]),
                            torch.stack([27*L3*L4, zero, 27*L1*L4, 27*L1*L3]),
                            torch.stack([27*L2*L4, 27*L1*L4, zero, 27*L1*L2]) # 2zero
                        ]).permute(2,0,1)
    

