
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import numpy as np


class MatSet():
    Ceramic = 2700, 7.2E10, 0.19, 6, 1E-7
    Glass = 2600, 6.2E10, 0.20, 1, 1E-7
    Wood = 750, 1.1E10, 0.25, 60, 2E-6
    Plastic = 1070, 1.4E9, 0.35, 30, 1E-6
    Iron = 8000, 2.1E11, 0.28, 10, 1e-7
    Polycarbonate = 1190, 2.4E9, 0.37, 0.5, 4E-7
    Steel = 7850, 2.0E11, 0.29, 20, 3E-8
    Tin = 7265, 5e10, 0.325, 2, 3E-8
    Test = 1, 1, 0, 0, 0
    # gt9 = 2700, 7.055E10, 0.1892, 15, 1E-7
    gt2 = 2700, 7.231E10, 0.1958, 15, 2.0E-8

class Material(object):
    def __init__(self, material):
        self.density, self.youngs, self.poisson, self.alpha, self.beta = material


class TinyNN(Module):
    def __init__(self, in_dim, mid_dim, out_dim, non_linear=True):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.non_linear = non_linear
        self.layer1 = Linear(in_dim, mid_dim)
        nn.init.normal_(self.layer1.weight, 0, np.sqrt(2 / in_dim))
        self.layer2 = Linear(mid_dim, mid_dim)
        nn.init.normal_(self.layer2.weight, 0, np.sqrt(2 / mid_dim))
        self.layer3 = Linear(mid_dim, out_dim)
        nn.init.normal_(self.layer3.weight, 0, np.sqrt(1 / mid_dim))
        if self.non_linear:
            self.relu = torch.nn.ReLU()
            
        self.last_layer = torch.nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        if self.non_linear:
            x = self.relu(x)
        x = self.layer2(x)
        if self.non_linear:
            x = self.relu(x)
        x = self.layer3(x)
        # new
        x = self.last_layer(x)
        return x


batch_trace = torch.vmap(torch.trace)


class LinearElastic():
    def __init__(self, mat: Material):
        self.youngs_modulus = mat.youngs / mat.density
        self.poisson_ratio = mat.poisson
        self.lame_lambda = self.youngs_modulus * self.poisson_ratio / \
            ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        self.lame_mu = self.youngs_modulus / (2 * (1 + self.poisson_ratio))

    # def __init__(self, youngs_modulus, poisson_ratio):
    #     self.youngs_modulus = youngs_modulus
    #     self.poisson_ratio = poisson_ratio
    #     self.lame_lambda = youngs_modulus * poisson_ratio / \
    #         ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    #     self.lame_mu = youngs_modulus / (2 * (1 + poisson_ratio))
        
    def __call__(self, F: torch.Tensor):
        batch_size, node_size, _, _ = F.shape
        F = F.reshape(batch_size * node_size, 3, 3)
        stress = self.lame_mu * (F + F.transpose(1, 2)) + \
            self.lame_lambda * batch_trace(F).unsqueeze(-1).unsqueeze(-1) * \
            torch.eye(3, device=F.device)
        return stress.reshape(batch_size, node_size, 3, 3)
    
    # make sure it is the same as TrainableLinear
    def get_stress(self, F):
        lame_lambda = (
            self.youngs_modulus
            * self.poisson_ratio
            / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        )
        lame_mu = self.youngs_modulus / (2 * (1 + self.poisson_ratio))
        stress = lame_mu * (F + F.transpose(1, 2)) + lame_lambda * batch_trace(
            F
        ).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=F.device)
        return stress

    def jacobian(self):
        inputs = torch.zeros(1, 3, 3).cuda().double()
        mat = torch.autograd.functional.jacobian(self.get_stress, inputs)
        return mat

    def stress_batch(self, F: torch.Tensor, weight: torch.Tensor = None):
        '''
        Piola stress
        F: deformation gradient with shape (batch_size, 3, 3)
        weight: weight for each sample, with shape (batch_size, 2)
        return: stress with shape (batch_size, 3, 3)
        '''
        if len(F.shape) == 2:
            F = F.unsqueeze(0)
        if len(F.shape) == 4:  # (mode_num, batch_size, 3, 3)
            shape0 = F.shape[0]
            F = F.reshape(F.shape[0] * F.shape[1], F.shape[2], F.shape[3])

        result = self.stress(F, weight)

        if shape0 is not None:
            result = result.reshape(
                shape0, result.shape[0] // shape0, result.shape[1], result.shape[2])

        return result

    # debug
    def stress(self, F: torch.Tensor, weight: torch.Tensor = None):
        '''
        Piola stress
        F: deformation gradient with shape (batch_size, 3, 3)
        weight: weight for each sample, with shape (batch_size, 2)
        return: stress with shape (batch_size, 3, 3)
        '''
        if len(F.shape) == 2:
            F = F.unsqueeze(0)
        if weight is None:
            weight = torch.ones(F.shape[0], 2, device=F.device)
        else:
            weight = weight(F.reshape(F.shape[0], -1))

        w1 = weight[:, 0].unsqueeze(-1).unsqueeze(-1)
        w2 = weight[:, 1].unsqueeze(-1).unsqueeze(-1)
        return self.lame_mu * (F + F.transpose(1, 2)) * w1 + \
            self.lame_lambda * batch_trace(F).unsqueeze(-1).unsqueeze(-1) * \
            torch.eye(3, device=F.device) * w2
