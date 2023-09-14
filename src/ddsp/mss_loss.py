"""
Implementation of Multi-Scale Spectral Loss as described in DDSP, 
which is originally suggested in NSF (Wang et al., 2019)
"""

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from geomloss import SamplesLoss
import numpy as np


def clip_spec(x, scale):
    freq_length = x.shape[-2]
    return x[..., :int(freq_length * scale), :]  # (batch, freq, time)

def spec2point(x, freq = None, sample_rate = None):
    '''
    x : (batch, freq, time)
    '''
    freq_res = x.shape[-2]
    x = x.detach()
    feature_num = 3
    points = torch.zeros(x.shape[0], x.shape[1], feature_num + 1, device=x.device)
    points[:, :, :feature_num] = torch.nn.functional.interpolate(x, size=(feature_num), mode='linear')
    pos = torch.arange(freq_res, dtype=torch.float32, device=x.device) / freq_res
    pos = pos.unsqueeze(0)
    pos = pos.repeat(x.shape[0], 1)
    points[:, :, feature_num] = pos
    if freq is not None:
        pos = freq_res / (sample_rate // 2) * freq
        pos_width = 2
        for w in range(pos_width, -1, -1):
            x_pos_current = pos - w
            x_mask = x_pos_current.long()
            in_range = (x_mask >= 0) & (x_mask < freq_res)
            x_mask = x_mask[in_range]
            x_pos_current = x_pos_current[in_range]
            points[:, x_mask, feature_num] = x_pos_current / freq_res
            x_pos_current = pos + w
            x_mask = x_pos_current.long()
            in_range = (x_mask >= 0) & (x_mask < freq_res)
            x_mask = x_mask[in_range]
            x_pos_current = x_pos_current[in_range]
            points[:, x_mask, feature_num] = x_pos_current / freq_res
    return points

def weighted_l1_loss(x_pred, x_true):
    """
    Weighted L1 loss. 
    
    output(loss) : torch.tensor(scalar)
    """
    time_length = x_pred.shape[-1]
    weight = 1 - torch.linspace(1.0, 0.9, time_length).to(x_pred.device)
    weight = weight / weight.sum() * time_length
    weight = weight.unsqueeze(0).unsqueeze(1)
    x_pred = x_pred[:,1:,:] * weight # remove DC component
    x_true = x_true[:,1:,:] * weight
    return F.l1_loss(x_pred, x_true)


def normlize(x):
    x_ = x.detach()
    max_value = x_.max(-1)[0]
    return x / (max_value.unsqueeze(-1) + 1e-7)

class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft, sample_rate, alpha=1.0, overlap=0.75, eps=1e-7, type = 'geomloss'):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length)
        self.geomloss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.loss_type = type
        self.sample_rate = sample_rate

    def log_func(self, x):
        return ((x + self.eps).log2() - np.log2(self.eps))
    

    def log_spec(self, x, scale = 1.0):
        S = self.spec(x)
        S = clip_spec(S, scale)
        return self.log_func(S)

    def forward(self, x_pred, x_true, freq = None, scale = 1.0):
        if self.loss_type == 'l1_loss':
            linear_true = self.spec(x_true)
            linear_pred = self.spec(x_pred)
            log_true = (linear_true + self.eps).log2()
            log_pred = (linear_pred + self.eps).log2()
            loss = self.alpha * weighted_l1_loss(log_pred, log_true) + weighted_l1_loss(linear_pred, linear_true)
        elif self.loss_type == "geomloss":
            x_true = normlize(x_true)
            x_pred = normlize(x_pred)
            linear_true = self.spec(x_true)
            linear_pred = self.spec(x_pred)
            log_true = self.log_spec(x_true, scale) / 40
            log_pred = self.log_spec(x_pred, scale) / 40
            points_log_true = spec2point(log_true)
            points_log_pred = spec2point(log_pred, freq, self.sample_rate)
            points_linear_true = spec2point(linear_true)
            points_linear_pred = spec2point(linear_pred, freq, self.sample_rate)
            loss_linear = self.geomloss(points_linear_pred, points_linear_true)
            loss_log = self.geomloss(points_log_pred, points_log_true) 
            loss = self.alpha * loss_log + loss_linear
        elif self.loss_type == 'rmse_loss':
            log_true = self.log_spec(x_true, scale)
            log_pred = self.log_spec(x_pred, scale)
            loss = torch.sqrt(self.MSEloss(log_pred, log_true))
        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.

    Usage ::

    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)

    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    """

    def __init__(self, n_ffts: list, sample_rate, alpha=1.0, overlap=0.75, eps=1e-7, type = 'geomloss'):
        super().__init__()
        self.n_ffts = n_ffts
        self.losses = nn.ModuleList(
            [SSSLoss(n_fft, sample_rate, alpha, overlap, eps, type) for n_fft in n_ffts])
        

    def forward(self, x_pred, x_true, freq = None, scale = 1.0):
        losses = [loss(x_pred, x_true, freq, scale) for loss in self.losses]
        return sum(losses).sum()
