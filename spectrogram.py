import numpy as np
from scipy.signal import spectrogram

# STFT grid. Chosen so that both axes shrink to 400 pixels by averaging whole blocks:
#   frequency: 4096 bins of 15 kHz -> central 3200 bins (48 MHz) -> blocks of 8 -> 400 x 120 kHz
#   time:      hop 809 samples     -> first 1600 frames          -> blocks of 4 -> 400 x 52.7 us
NFFT = 4096
HOP = 809
FREQ_BINS_KEPT = 3200    # the middle 48 MHz; the edges are guard bands
FREQ_BLOCK = 8
TIME_FRAMES_KEPT = 1600
TIME_BLOCK = 4
SIDE = 400


def power_map(iq, fs):
    """Spectrogram in linear power, shape (400 time, 400 frequency), and its duration in ms."""
    # detrend=False: scipy's default removes the mean of every window, which would also delete
    # any real signal sitting exactly at 0 Hz
    f, t, sxx = spectrogram(iq, fs=fs, window="hann", nperseg=NFFT, noverlap=NFFT - HOP,
                            detrend=False, return_onesided=False)
    sxx = np.fft.fftshift(sxx, axes=0)

    # Keep only the middle 48 MHz
    center = NFFT // 2
    sxx = sxx[center - FREQ_BINS_KEPT // 2: center + FREQ_BINS_KEPT // 2, :TIME_FRAMES_KEPT]

    # Average blocks of bins and frames. Averaging is done on linear power, never on dB
    sxx = sxx.reshape(SIDE, FREQ_BLOCK, SIDE, TIME_BLOCK).mean(axis=(1, 3))
    return sxx.T, t[TIME_FRAMES_KEPT - 1] * 1e3  # (time, frequency)


def iq_to_spectrogram(iq, fs):
    """The same spectrogram in dB."""
    sxx, duration_ms = power_map(iq, fs)
    return 10 * np.log10(sxx), duration_ms
