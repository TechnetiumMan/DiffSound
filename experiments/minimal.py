import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from src.material_model import TinyNN
from src.utils import dense_to_sparse, LSDloss
from src.solve import WaveSolver
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)
import torch._dynamo

writer = SummaryWriter()


def mel_spectrogram(waveform, sample_rate, n_fft, n_mel=128):
    """Compute mel spectrogram from waveform"""
    t = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mel, normalized=True).cuda()
    return t(waveform)


def add_spectrogram(spec, n_iter):
    writer.add_image('spectrogram', spec.unsqueeze(0), n_iter)

def add_gt_spectrogram(spec):
    writer.add_image('gt_spectrogram', spec.unsqueeze(0))

def add_loss(loss, n_iter):
    writer.add_scalar("loss", loss, n_iter)


def add_waveform(waveform, n_iter, name="waveform"):
    plt.figure()
    plt.plot(waveform.detach().cpu().numpy())
    writer.add_figure(name, plt.gcf(), n_iter)


def get_gt_signal(dt, num_steps):
    t = torch.arange(0, 1, dt)[:num_steps]
    gt_signal = torch.sin(2 * 3.1415926 * 100 * t) \
    + torch.sin(2 * 3.1415926 * 200 * t) 
    #   + \  torch.sin(2 * 3.1415926 * 1600 * t)
    return gt_signal


def get_force():
    def force(t): return torch.Tensor(
        [100, 100, 100]).cuda() if t == 0 else torch.zeros(3).cuda()
    return force

def normalize_input(x):
    # the network input may vary from a very large range.
    # so we need to normalize it to a smaller range, (-1, 1) better.
    # for a simple test, we scale (1e-12, 1) to (0, 1), and The negative case is symmetrical 
    
    # Oh no! this way can not work! now we assume that the positive and negative displacement is symmetial!
    # so we just scale abs(x) from (1e-8, 1e-4) to (-1, 1)
    x_abs = torch.abs(x)
    x_abs = torch.where(x_abs < 1e-8, 1e-8, x_abs)
        
    # x_scale = torch.log10(x_abs * 1e12) / 12
    x_scale = torch.log10(x_abs * 1e6) / 2
    # x_out = torch.where(x < 0, -x_scale, x_scale)
    
    return x_scale
    

def post_activation(x_out, x_in):
    # max_bound = (2 * 3.1415926 * 2000)**2 
    # x = torch.tanh(x) 
    # return x * max_bound
    
    # in fact, the network will not know what is "elastic" in a random init, 
    # which cause that the sign of output elastic force is not relevant to the sign of input displacement,
    # and it is really hard to optimize from a wave that is actually not "vibrating".
    # but we know that the elastic force is in the opposite direction of the displacement.
    # so the output should be corrected to be negative of input, and its value is a function of network output.
    # we use tanh(x) ∈ (-1, 1) map to freq ∈ (20, 20000)  (force = (2*pi*freq)**2)
    # for easier test, we assume freq ∈ (50, 200) and gt freq is 100
    
    # now gt signal is not linear, 
    # but first 2 dim of the freq here should be close to 100 and be the same, and the last is 100.
    
    # Oh no! it didn't work!
    # I'm trying to use 2 separate mode, which freq is 100, 200, and the output scale to (50, 400).
    
    # didn't work too!
    # now scale dim0 to (75, 150), dim1 to (150, 300)
    x_out = torch.tanh(x_out) 
    x_out_scale = x_out.clone() # to prevent inplace modify
    # x_out_scale[:, 1] = x_out[:, 1] + 2
    freq = 75 * torch.exp2((x_out_scale + 1) * 0.5)
    force = ((2 * 3.1415926 * freq) ** 2) * (x_in)
    return force
    

def main():
    torch._dynamo.config.verbose = True
    dt = 1 / 4000
    num_steps = 2048
    n_fft = 512
    n_mel = 128
    
    # gt_signal = get_gt_signal(dt, num_steps).cuda()
    # gt_spec = mel_spectrogram(gt_signal, int(1 / dt), n_fft)
    # gt_spec = gt_spec / (gt_spec**2).mean()**0.5
    # add_gt_spectrogram(gt_spec)
    

    mass_matrix = dense_to_sparse(torch.diag(torch.ones(3)).cuda())
    damping_matrix = dense_to_sparse(torch.zeros(3, 3).cuda())
    def damping_matvec(x): return damping_matrix @ x
    force = get_force()

    stiffness_net = TinyNN(3, 16, 3).cuda()
    # stiffness_net = torch.compile(stiffness_net)
    
    def gt_stiffness_matvec(x):
        freq = 100
        force_scale = (2 * 3.1415926 * freq) ** 2
        non_diag_scale = force_scale
        # we assume the stiffness matrix is not a diag matrix (but symmetric for a easier test)
        stiff_matrix = torch.tensor([[force_scale, non_diag_scale, 0],
                                    [non_diag_scale, force_scale, 0],
                                    [0, 0, force_scale]]).cuda()
        elastic_force = stiff_matrix @ x
        return elastic_force

    def stiffness_matvec(x):
        x_in = x.unsqueeze(0)
        x_in_norm = normalize_input(x_in)
        x_out = stiffness_net(x_in_norm)
        x = post_activation(x_out, x_in)
        x = x.squeeze(0)
        return x
    
    gt_solver = WaveSolver(mass_matrix, damping_matvec,
                        gt_stiffness_matvec, force, dt)
    solver = WaveSolver(mass_matrix, damping_matvec,
                        stiffness_matvec, force, dt)
    
    with torch.no_grad():
        gt_signal = gt_solver.solve(num_steps)
        gt_signal = gt_signal.sum(dim=1)
        add_waveform(gt_signal, 0, "gt_signal")
        gt_spec = mel_spectrogram(gt_signal, int(1 / dt), n_fft, n_mel)
        gt_spec = gt_spec / (gt_spec**2).mean()**0.5
        add_gt_spectrogram(gt_spec)
    
    
    optimizer = Adam(stiffness_net.parameters(), lr=1e-3)

    for i in range(1000):
        x = solver.solve(num_steps)
        # print(x.shape)
        x = x.sum(dim=1)
        spec = mel_spectrogram(x, int(1 / dt), n_fft, n_mel)
        spec = spec / (spec**2).mean()**0.5
        add_waveform(x, i)
        add_spectrogram(spec, i)
        # loss = F.mse_loss(spec[:,1:-1], gt_spec[:,1:-1]) 
        # loss_crit = torch.nn.L1Loss() 
        # loss = loss_crit(spec[:,1:-1], gt_spec[:,1:-1]) # begin and end column of a spec is not accurate, so remove them
        loss = LSDloss(spec[:,1:-1], gt_spec[:,1:-1], eps=1e-3)
        add_loss(loss, i)
        optimizer.zero_grad()
        loss.backward()
        # for param in stiffness_net.parameters():
        #     print(param.grad)
        optimizer.step()
        print(loss.item())


if __name__ == '__main__':
    main()
    writer.close()
