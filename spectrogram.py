import numpy as np
from scipy.signal import spectrogram


def power_map(iq, fs):
    """WASD-style spectrogram in linear power, shape (400 time, frequency), and its duration in ms."""
    f, t, sxx = spectrogram(iq, fs=fs, window="hann", nperseg=3200, noverlap=2390, return_onesided=False)
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)

    # Keep only the middle 48 MHz (the edges are guard bands)
    keep = np.abs(f) <= 24e6
    f, sxx = f[keep], sxx[keep]

    # Average every 4 frames: 1600 frames -> 400 rows, like WASD
    sxx = sxx[:, :1600].reshape(len(f), 400, 4).mean(axis=2)
    return sxx.T, t[1599] * 1e3  # (time, frequency)


def iq_to_spectrogram(iq, fs):
    """The same spectrogram in dB."""
    sxx, duration_ms = power_map(iq, fs)
    return 10 * np.log10(sxx), duration_ms
