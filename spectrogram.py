import numpy as np
from scipy.signal import spectrogram


def iq_to_spectrogram(iq, fs):
    f, t, sxx = spectrogram(iq, fs=fs, window="hann", nperseg=3200, noverlap=2390, return_onesided=False)
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)

    # Keep only the middle 48 MHz (the edges are guard bands)
    keep = np.abs(f) <= 24e6
    f, sxx = f[keep], sxx[keep]

    # Average every 4 frames: 1600 frames -> 400 rows, like WASD
    sxx = sxx[:, :1600].reshape(len(f), 400, 4).mean(axis=2)
    sxx_db = 10 * np.log10(sxx).T  # (time, frequency)
    return sxx_db, t[1599] * 1e3
