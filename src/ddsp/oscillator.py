import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .utils import modifed_sigmoid
from .filtered_noise import FilteredNoise
from ..material_model import MatSet, Material

class WeightedParam(nn.Module):
    def __init__(self, values_list: torch.Tensor):
        super(WeightedParam, self).__init__()
        self.values_list = values_list
        self.probablity = nn.Parameter(torch.zeros(len(values_list)))
        self.probablity.data.uniform_(-1, 1)

    def forward(self):
        probablity = F.softplus(self.probablity)
        probablity = probablity / probablity.sum()
        value = (self.values_list * probablity).sum()
        return value
    
class WeightedSum(nn.Module):
    def __init__(self, dims: list, vlist: list):
        super(WeightedSum, self).__init__()
        self.values_list = torch.tensor(
            vlist, dtype=torch.float32).cuda()
        self.params = nn.Parameter(torch.zeros(*dims, len(self.values_list)))
        self.params.data.uniform_(-4, 4)

    def forward(self):
        x = F.softplus(self.params)
        x = x / x.sum(dim=-1).unsqueeze(-1)
        x = (self.values_list * x).sum(dim=-1)
        return x


class DirectValue(nn.Module):
    def __init__(self, dims: list):
        super(DirectValue, self).__init__()
        self.value = nn.Parameter(torch.zeros(*dims))
        # self.value.data.uniform_(0, 4)
        self.value.data.uniform_(0, 0.04)

    def forward(self):
        return modifed_sigmoid(self.value)


class DampedOscillator(nn.Module):
    def __init__(self, forces, audio_num, mode_num, sample_num, sr, f_range:list, mat:Material):
        super(DampedOscillator, self).__init__()
        self.audio_num = audio_num
        self.sr = sr
        self.sample_num = sample_num
        self.mode_num = mode_num
        # self.freq_nonlinear = WeightedSum(
        #     [audio_num, mode_num, sample_num], f_range) 
        self.freq_nonlinear = WeightedSum(
            [audio_num, mode_num, 8000], f_range) # not used
        bin_num = 64
        self.alpha_list = torch.linspace(
            np.log(mat.alpha / 10),
            np.log(mat.alpha * 10),
            bin_num,
        )
        self.alpha_list = torch.exp(self.alpha_list)
        self.alpha = WeightedSum([1, mode_num, 1], list(self.alpha_list))
        self.beta_list = torch.linspace(
            np.log(mat.beta / 10),
            np.log(mat.beta * 10),
            bin_num,
        )
        self.mat = mat
        self.beta_list = torch.exp(self.beta_list)
        self.beta = WeightedSum([1, mode_num, 1], list(self.beta_list))
        self.amp = DirectValue([audio_num, mode_num, 1])
        
        # self.noise = FilteredNoise(audio_num, sample_num)
        self.noise = FilteredNoise(audio_num, 8000) # not used, just for load
        
        self.forces = forces.reshape(audio_num, 1, -1)
        self.forces = torch.flip(self.forces, [-1])
        self.force_frame_num = forces.shape[-1]
    
    def early(self, freq_linear, damping_curve):
        freq = freq_linear.detach().cpu().numpy().reshape(-1)
        damp = torch.zeros(self.audio_num, self.mode_num, self.sample_num).cuda()
        damp_= torch.zeros(1, self.mode_num, 1).cuda()
        for i in range(len(freq)):
            damp[:, i, :] = torch.tensor(damping_curve(freq[i]))
            damp_[:, i, :] = torch.tensor(damping_curve(freq[i]))
        undamped_freq = freq_linear #  + 0.0 * self.freq_nonlinear()
        lbd_linear = (freq_linear * 2 * np.pi)**2
        self.damped_freq = (lbd_linear - damp_**2)**0.5 / (2 * np.pi)
        lbd = (undamped_freq * 2 * np.pi)**2
        freq = (lbd - damp**2)**0.5 / (2 * np.pi)
        damp = torch.cumsum(damp / self.sr, dim=2)
        freq = torch.cumsum(freq / self.sr, dim=2)
        damp_part = torch.exp(-damp)
        freq_part = torch.sin(2 * np.pi * freq)
        signal = (damp_part * freq_part)

        signal = signal.sum(1)
        signal = signal.unsqueeze(0)
        signal = F.conv1d(signal, self.forces, groups=self.audio_num,
                          padding=self.force_frame_num - 1)
        signal = signal.squeeze(0)
        signal = signal[:, :self.sample_num]
        return signal



    def forward(self, freq_linear, non_linear_rate=0.0, noise_rate=0.0):
        amp = self.amp()
        
        freq_linear = torch.reshape(freq_linear, (1, self.mode_num, 1))
        freq_linear = freq_linear.repeat((self.audio_num, 1, self.sample_num))
        
        undamped_freq = freq_linear # + non_linear_rate * self.freq_nonlinear()
        lbd_linear = (freq_linear * 2 * np.pi)**2
        damp_linear = 0.5 * (self.alpha() + self.beta() * lbd_linear)
        self.damped_freq = (lbd_linear - damp_linear**2)**0.5 / (2 * np.pi)
        lbd = (undamped_freq * 2 * np.pi)**2
        damp = 0.5 * (self.alpha() + self.beta() * lbd)
        freq = (lbd - damp**2)**0.5 / (2 * np.pi)  
        # noise = self.noise()

        damp = torch.cumsum(damp / self.sr, dim=2)
        freq = torch.cumsum(freq / self.sr, dim=2)

        damp_part = torch.exp(-damp)
        freq_part = torch.sin(2 * np.pi * freq)
        signal = (amp * damp_part * freq_part)

        signal = signal.sum(1)
        signal = signal.unsqueeze(0)
        signal = F.conv1d(signal, self.forces, groups=self.audio_num,
                          padding=self.force_frame_num - 1)
        signal = signal.squeeze(0)
        signal = signal[:, :self.sample_num]
        return signal # + noise * noise_rate

class GTDampedOscillator(nn.Module):
    def __init__(self, forces, audio_num, mode_num, sample_num, sr, f_range:list, mat:Material):
        super(GTDampedOscillator, self).__init__()
        self.audio_num = audio_num
        self.sr = sr
        self.sample_num = sample_num
        self.mode_num = mode_num
        self.freq_linear = WeightedSum(
            [1, mode_num, 1], f_range)
        self.freq_nonlinear = WeightedSum(
            [audio_num, mode_num, sample_num], f_range)
        bin_num = 64
        self.alpha_list = torch.linspace(
            np.log(mat.alpha / 10),
            np.log(mat.alpha * 100),
            bin_num,
        )
        self.alpha_list = torch.exp(self.alpha_list)
        self.alpha = WeightedSum([1, mode_num, 1], list(self.alpha_list))
        self.beta_list = torch.linspace(
            np.log(mat.beta / 10),
            np.log(mat.beta * 100),
            bin_num,
        )
        self.mat = mat
        self.beta_list = torch.exp(self.beta_list)
        self.beta = WeightedSum([1, mode_num, 1], list(self.beta_list))
        self.amp = DirectValue([audio_num, mode_num, 1])
        self.noise = FilteredNoise(audio_num, sample_num)
        self.forces = forces.reshape(audio_num, 1, -1)
        self.forces = torch.flip(self.forces, [-1])
        self.force_frame_num = forces.shape[-1]

    def damping(self):
        lbd_linear = (self.freq_linear() * 2 * np.pi)**2
        damp_linear = 0.5 * (self.alpha() + self.beta() * lbd_linear)
        return damp_linear


    def forward(self, non_linear_rate=0.0, noise_rate=0.0):
        amp = self.amp()
        freq_linear = self.freq_linear()
        undamped_freq = freq_linear + non_linear_rate * self.freq_nonlinear()
        lbd = (undamped_freq * 2 * np.pi)**2
        damp = 0.5 * (self.alpha() + self.beta() * lbd)
        # self.damped_freq = (lbd - damp**2)**0.5 / (2 * np.pi)
        
        freq = (lbd - damp**2)**0.5 / (2 * np.pi)  
        noise = self.noise()
        self.undamped_freq = ((2 * np.pi * freq)**2 + damp**2)**0.5 / \
            (2 * np.pi)  # linear approximation

        damp = torch.cumsum(damp / self.sr, dim=2)
        freq = torch.cumsum(freq / self.sr, dim=2)

        damp_part = torch.exp(-damp)
        freq_part = torch.sin(2 * np.pi * freq)
        signal = (amp * damp_part * freq_part)

        signal = signal.sum(1)
        signal = signal.unsqueeze(0)
        signal = F.conv1d(signal, self.forces, groups=self.audio_num,
                          padding=self.force_frame_num - 1)
        signal = signal.squeeze(0)
        signal = signal[:, :self.sample_num]
        return signal + noise * noise_rate
        
    
class TraditionalDampedOscillator(nn.Module):
    def __init__(self, forces, audio_num, mode_num, sample_num, sr, mat:Material):
        super(TraditionalDampedOscillator, self).__init__()
        self.audio_num = audio_num
        self.sr = sr
        self.sample_num = sample_num
        self.mode_num = mode_num
        # self.freq_linear = WeightedSum(
        #     [1, mode_num, 1], f_range)
        # self.freq_nonlinear = WeightedSum(
        #     [audio_num, mode_num, sample_num], f_range)
        # bin_num = 64
        # self.alpha_list = torch.linspace(
        #     np.log(mat.alpha / 10),
        #     np.log(mat.alpha * 10),
        #     bin_num,
        # )
        # self.alpha_list = torch.exp(self.alpha_list)
        # self.alpha = WeightedSum([1, mode_num, 1], list(self.alpha_list))
        self.alpha = mat.alpha
        self.beta = mat.beta
        # self.beta_list = torch.linspace(
        #     np.log(mat.beta / 10),
        #     np.log(mat.beta * 10),
        #     bin_num,
        # )
        self.mat = mat
        # self.beta_list = torch.exp(self.beta_list)
        # self.beta = WeightedSum([1, mode_num, 1], list(self.beta_list))
        # self.amp = DirectValue([audio_num, mode_num, 1])
        # self.amp = 1
        # self.noise = FilteredNoise(audio_num, sample_num)
        self.forces = forces.reshape(audio_num, 1, -1)
        self.forces = torch.flip(self.forces, [-1])
        self.force_frame_num = forces.shape[-1]

    def forward(self, freq_linear):
        freq_linear = torch.reshape(freq_linear, (1, self.mode_num, 1))
        freq_linear = freq_linear.repeat((self.audio_num, 1, self.sample_num))
        # amp = self.amp()

        undamped_freq = freq_linear # + non_linear_rate * self.freq_nonlinear()
        lbd = (undamped_freq * 2 * np.pi)**2
        damp = 0.5 * (self.alpha + self.beta * lbd)
        self.damped_freq = (lbd - damp**2)**0.5 / (2 * np.pi)
        
        freq = (lbd - damp**2)**0.5 / (2 * np.pi)  
        # noise = self.noise()
        self.undamped_freq = ((2 * np.pi * freq)**2 + damp**2)**0.5 / \
            (2 * np.pi)  # linear approximation

        damp = torch.cumsum(damp / self.sr, dim=2)
        freq = torch.cumsum(freq / self.sr, dim=2)

        damp_part = torch.exp(-damp)
        freq_part = torch.sin(2 * np.pi * freq)
        signal = (damp_part * freq_part)

        signal = signal.sum(1)
        signal = signal.unsqueeze(0)
        signal = F.conv1d(signal, self.forces, groups=self.audio_num,
                          padding=self.force_frame_num - 1)
        signal = signal.squeeze(0)
        signal = signal[:, :self.sample_num]
        return signal # + noise * noise_rate
    
    

def init_damps(osc):
    print('Start pretraining for alpha and beta')
    optimizer = torch.optim.Adam(list(osc.alpha.parameters()) + list(osc.beta.parameters()), lr=0.01)
    for i in tqdm(range(2000)):
        optimizer.zero_grad()
        loss = (osc.alpha() - osc.mat.alpha)**2/ osc.mat.alpha**2 + (osc.beta() - osc.mat.beta)**2 / osc.mat.beta**2
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    print('Pretraining finished')
    # print('alpha ( net ):', osc.alpha(), 'beta:', osc.beta())
    # print('alpha (table):', osc.mat.alpha, 'beta:', osc.mat.beta)