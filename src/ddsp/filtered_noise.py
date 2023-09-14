import numpy as np
import torch
import torch.nn as nn
from .utils import modifed_sigmoid


class FilteredNoise(nn.Module):
    def __init__(self, noise_num, sample_num, filter_coeff_length=65, frame_length=64, attenuate_gain=1.0, device='cuda'):
        super(FilteredNoise, self).__init__()
        self.frame_length = frame_length
        self.filter_coeff_length = filter_coeff_length
        self.noise_num = noise_num
        self.sample_num = sample_num
        self.device = device
        self.attenuate_gain = attenuate_gain
        self.coefficient_bank = nn.Parameter(torch.zeros(
            noise_num, sample_num // frame_length + 1, filter_coeff_length))
        self.coefficient_bank.data.uniform_(-1, 1)

    def forward(self):
        x = self.coefficient_bank
        x = modifed_sigmoid(x)
        batch_num, frame_num, filter_coeff_length = x.shape
        self.filter_window = nn.Parameter(torch.hann_window(
            filter_coeff_length * 2 - 1, dtype=torch.float32), requires_grad=False).to(self.device)
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response.
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd,
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = torch.complex(
            x, torch.zeros_like(x)).view(-1, filter_coeff_length)
        zero_phase_ir_bank = torch.fft.irfft(
            ZERO_PHASE_FR_BANK, n=filter_coeff_length * 2 - 1)

        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(
            filter_coeff_length - 1, 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * \
            self.filter_window.view(1, -1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(
            windowed_linear_phase_ir_bank, (0, self.frame_length - 1))
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.fft.rfft(
            zero_paded_windowed_linear_phase_ir_bank)

        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length,
                           dtype=torch.float32).view(-1, self.frame_length).to(self.device) * 2 - 1
        zero_paded_noise = nn.functional.pad(
            noise, (0, filter_coeff_length * 2 - 2))
        ZERO_PADED_NOISE = torch.fft.rfft(zero_paded_noise)

        # Convolve & IDFT to make filtered noise frame, for each frame, noise band, and batch.
        FILTERED_NOISE = ZERO_PADED_NOISE * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK
        filtered_noise = torch.fft.irfft(FILTERED_NOISE).view(
            batch_num, frame_num, -1) * self.attenuate_gain

        # Overlap-add to build time-varying filtered noise.
        overlap_add_filter = torch.eye(
            filtered_noise.shape[-1], requires_grad=False).unsqueeze(1).to(self.device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2),
                                                       overlap_add_filter,
                                                       stride=self.frame_length,
                                                       padding=0).squeeze(1)

        return output_signal[:, :self.sample_num]
