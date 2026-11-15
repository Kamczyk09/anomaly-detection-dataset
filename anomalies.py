"""Simple synthetic anomalies (tone, chirp, pulse) added to a complex I/Q signal.

All functions share the same arguments:
    iq          complex I/Q samples
    fs          sampling frequency [Hz]
    start, stop time window of the anomaly [s]
    isr         anomaly power relative to the average power of the recording [dB]
    ramp        smooth on/off time [s] (an abrupt switch smears over all frequencies)
"""
import numpy as np


def _amplitude(iq, isr):
    """Amplitude giving power = average power of iq * 10^(isr/10)."""
    power = np.mean(np.abs(iq) ** 2)
    return np.sqrt(power * 10 ** (isr / 10))


def _envelope(t, start, stop, ramp):
    """1 inside [start, stop], 0 outside, with rounded (raised-cosine) edges."""
    env = np.clip(np.minimum(t - start, stop - t) / ramp, 0, 1)  # 0 -> 1 -> 0, linear edges
    return np.sin(np.pi / 2 * env) ** 2


def add_tone(iq, fs, freq, start, stop, isr, ramp=0.2e-3):
    """Constant tone at frequency freq [Hz]. Spectrogram: vertical line."""
    t = np.arange(len(iq)) / fs
    tone = np.exp(2j * np.pi * freq * t)
    return iq + _amplitude(iq, isr) * tone * _envelope(t, start, stop, ramp)


def add_chirp(iq, fs, f_start, f_stop, start, stop, isr, ramp=0.2e-3):
    """Linear sweep from f_start to f_stop [Hz]. Spectrogram: slanted line."""
    t = np.arange(len(iq)) / fs
    tau = t - start                                   # time since the chirp began
    k = (f_stop - f_start) / (stop - start)           # sweep rate [Hz/s]
    chirp = np.exp(2j * np.pi * (f_start * tau + 0.5 * k * tau ** 2))
    return iq + _amplitude(iq, isr) * chirp * _envelope(t, start, stop, ramp)


def add_pulse(iq, fs, freq, start, stop, isr, period=2e-3, width=0.2e-6):
    """Train of short Gaussian pulses at freq [Hz], one every `period` [s].

    Spectrogram: short horizontal dashes. A shorter `width` [s] gives wider dashes.
    `isr` sets the peak power of a pulse.
    """
    t = np.arange(len(iq)) / fs
    train = np.zeros(len(t))
    for center in np.arange(start, stop, period):
        train += np.exp(-0.5 * ((t - center) / width) ** 2)
    carrier = np.exp(2j * np.pi * freq * t)
    return iq + _amplitude(iq, isr) * train * carrier
