"""Simple synthetic anomalies (tone, chirp, pulse, comb, OFDM, FSK) added to a complex I/Q signal.

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


def add_comb(iq, fs, f_first, spacing, n_tones, start, stop, isr, ramp=0.2e-3):
    """n_tones equally spaced tones: f_first, f_first + spacing, ... [Hz].

    Spectrogram: several parallel vertical lines. `isr` sets the power of all tones together.
    """
    t = np.arange(len(iq)) / fs
    comb = np.zeros(len(t), dtype=complex)
    for k in range(n_tones):
        comb += np.exp(2j * np.pi * (f_first + k * spacing) * t)
    comb /= np.sqrt(n_tones)                          # unit power, whatever the number of tones
    return iq + _amplitude(iq, isr) * comb * _envelope(t, start, stop, ramp)


def add_ofdm(iq, fs, freq, bandwidth, start, stop, isr, rng, subcarrier_spacing=15e3,
             ramp=0.2e-3):
    """OFDM carrier: many subcarriers with random QPSK symbols, centred at freq [Hz].

    Spectrogram: a filled rectangle `bandwidth` [Hz] wide.
    rng: np.random.Generator for the random symbols.
    """
    t = np.arange(len(iq)) / fs
    symbol_len = int(round(fs / subcarrier_spacing))  # samples per OFDM symbol (4096 at 61.44 MHz)
    n_symbols = len(iq) // symbol_len + 1

    # Random QPSK symbol on each subcarrier inside the bandwidth, zero elsewhere
    bins = np.fft.fftfreq(symbol_len, d=1 / fs)
    used = np.abs(bins) <= bandwidth / 2
    qpsk = (rng.choice([-1, 1], size=(n_symbols, used.sum()))
            + 1j * rng.choice([-1, 1], size=(n_symbols, used.sum())))
    grid = np.zeros((n_symbols, symbol_len), dtype=complex)
    grid[:, used] = qpsk

    # One inverse FFT per symbol turns the subcarriers into a time signal
    ofdm = np.fft.ifft(grid, axis=1).ravel()[:len(iq)]
    ofdm /= np.sqrt(np.mean(np.abs(ofdm) ** 2))       # unit power
    carrier = np.exp(2j * np.pi * freq * t)
    return iq + _amplitude(iq, isr) * ofdm * carrier * _envelope(t, start, stop, ramp)


def add_fsk(iq, fs, freq, deviation, symbol_duration, start, stop, isr, rng, ramp=0.1e-3):
    """2-FSK: a tone jumping at random between freq - deviation and freq + deviation [Hz].

    Each symbol lasts `symbol_duration` [s]. Neighbouring symbols with the same bit form one
    burst, and every burst gets its own smooth on/off, so the jumps do not smear over all
    frequencies and the tone never fades between equal symbols. `ramp` should be at most
    symbol_duration / 2. Spectrogram: dashes alternating between two frequencies.
    rng: np.random.Generator for the random bits.
    """
    t = np.arange(len(iq)) / fs
    symbol_starts = np.arange(start, stop, symbol_duration)
    bits = rng.choice([-1, 1], size=len(symbol_starts))

    # Indices of the first symbol of each burst: wherever the bit changes
    burst_starts = [0] + [k for k in range(1, len(bits)) if bits[k] != bits[k - 1]]
    burst_ends = burst_starts[1:] + [len(bits)]

    fsk = np.zeros(len(t), dtype=complex)
    for k0, k1 in zip(burst_starts, burst_ends):
        s0 = symbol_starts[k0]
        s1 = min(start + k1 * symbol_duration, stop)
        tone = np.exp(2j * np.pi * (freq + deviation * bits[k0]) * t)
        fsk += tone * _envelope(t, s0, s1, ramp)
    return iq + _amplitude(iq, isr) * fsk
