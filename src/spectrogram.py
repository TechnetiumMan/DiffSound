import torchaudio.transforms as T


class MelSpectrogram():
    def __init__(self, sample_rate, n_fft, n_mels):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.t = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, f_min=20).cuda()
        self.t2 = T.AmplitudeToDB().cuda()

    def __call__(self, waveform):
        return self.t2(self.t(waveform))


def resample(waveform, sample_rate, new_sample_rate):
    t = T.Resample(sample_rate, new_sample_rate).cuda()
    return t(waveform)


import matplotlib.pyplot as plt


def plot_spectrogram(spec):
    plt.figure(figsize=(10, 6))
    plt.imshow(spec.detach().cpu().numpy())


def plot_spectrograms(specs, n_row=3, n_col=10):
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n_row * n_col):
        # add subplot
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(specs[i].detach().cpu().numpy())
