import torch.nn.functional as F
import torch
import numpy as np


def modifed_sigmoid(x):
    x = F.sigmoid(x)
    x = 2 * (x**2.3) + 1e-6
    return x
