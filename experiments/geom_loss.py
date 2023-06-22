import sys
sys.path.append('./')
from src.ddsp.mss_loss import SSSLoss, MSSLoss
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_imag(tensor1, tensor2, tensor3):
    tensor1 = tensor1[0].detach().cpu().numpy()
    tensor2 = tensor2[0].detach().cpu().numpy()
    tensor3 = tensor3[0].detach().cpu().numpy()
    fig, axis = plt.subplots(1, 3)
    fig.colorbar(axis[0].imshow(tensor1, origin="lower", aspect="auto", cmap='magma'), ax=axis[0])
    fig.colorbar(axis[1].imshow(tensor2, origin="lower", aspect="auto", cmap='magma'), ax=axis[1])
    fig.colorbar(axis[2].imshow(tensor3, origin="lower", aspect="auto", cmap='magma'), ax=axis[2])
    # title
    axis[0].set_title('x_pred')
    axis[1].set_title('x_pred.grad')
    axis[2].set_title('x_gt')
    plt.show()
    

scale = torch.tensor([1.5]).cuda().requires_grad_(True)
gt_freq = torch.tensor([440, 880, 1000]).cuda().unsqueeze(-1)

sample_rate = 16000
frame_num = 8000
early_loss_func = MSSLoss([2048, 1024], sample_rate, type='geomloss').cuda()
late_loss_func = MSSLoss([1024, 512, 256, 128, 64], sample_rate, type='l1_loss').cuda()

t = torch.arange(frame_num).cuda() / sample_rate
gt_signal = torch.sin(2 * np.pi * gt_freq * t)
gt_signal = gt_signal.sum(0).unsqueeze(0)

optimizer = torch.optim.Adam([scale], lr=0.001)
max_epoch = 2000
for epoch in tqdm(range(max_epoch)):

    freq = gt_freq * scale
    signal = torch.sin(2 * np.pi * freq * t)
    signal = signal.sum(0).unsqueeze(0)

    if epoch < max_epoch * 0.4:
        loss = early_loss_func(signal, gt_signal, freq)
    else:
        loss = late_loss_func(signal, gt_signal, freq)

    optimizer.zero_grad()
    loss.backward()
    print(scale.item(), loss.item(), scale.grad)
    optimizer.step()
    
    
