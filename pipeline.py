"""Random anomaly pipeline: one spectrogram with random anomalies, driven by a TOML config.

Usage:
    cfg = load_config("configs/v1.toml")
    rng = np.random.default_rng(seed)
    sample = generate_spectrogram("data/iq/IQ_....bin", cfg, rng)

`sample` holds the image, the semantic mask and one dict per inserted anomaly (its type, all
its parameters and its labels).
"""
import tomllib

import numpy as np

from anomalies import add_tone, add_chirp, add_pulse, add_comb, add_ofdm, add_fsk
from labels import anomaly_labels, semantic_mask
from spectrogram import iq_to_spectrogram, power_map


def load_config(path):
    """Read a TOML config file into a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_capture(path, num_samples):
    """Complex I/Q from a .bin file (interleaved int32: I, Q, I, Q, ...), first num_samples."""
    raw = np.fromfile(path, dtype=np.int32, count=2 * num_samples)
    if len(raw) < 2 * num_samples:
        raise ValueError(f"{path} has fewer than {num_samples} samples")
    return raw[0::2] + 1j * raw[1::2]


def draw(rng, spec):
    """One value from a config entry: a plain number, or {dist = "uniform"} / {dist = "choice"}."""
    if not isinstance(spec, dict):
        return spec
    if spec["dist"] == "uniform":
        return rng.uniform(spec["min"], spec["max"])
    if spec["dist"] == "choice":
        weights = spec.get("weights", [1] * len(spec["values"]))
        p = np.array(weights) / np.sum(weights)
        return rng.choice(spec["values"], p=p).item()
    raise ValueError(f"unknown distribution: {spec['dist']}")


def _window_and_power(rng, cfg, kind, duration):
    """Start, stop [s] and isr [dB] of one anomaly; every class draws them the same way."""
    min_duration = cfg["scene"]["min_duration_s"]
    start = rng.uniform(0, duration - min_duration)
    stop = rng.uniform(start + min_duration, duration)
    isr_spec = cfg["classes"][kind].get("isr_db", cfg["power"]["isr_db"])
    return {"start": start, "stop": stop, "isr": draw(rng, isr_spec)}


# One function per anomaly class: draws its parameters (as the add_* function expects them).

def sample_tone(rng, cfg, duration):
    c, f_max = cfg["classes"]["tone"], cfg["scene"]["f_max_hz"]
    return {"freq": rng.uniform(-f_max, f_max), "ramp": c["ramp_s"],
            **_window_and_power(rng, cfg, "tone", duration)}


def sample_chirp(rng, cfg, duration):
    c, f_max = cfg["classes"]["chirp"], cfg["scene"]["f_max_hz"]
    return {"f_start": rng.uniform(-f_max, f_max), "f_stop": rng.uniform(-f_max, f_max),
            "ramp": c["ramp_s"], **_window_and_power(rng, cfg, "chirp", duration)}


def sample_pulse(rng, cfg, duration):
    c, f_max = cfg["classes"]["pulse"], cfg["scene"]["f_max_hz"]
    return {"freq": rng.uniform(-f_max, f_max),
            "period": draw(rng, c["period_s"]), "width": draw(rng, c["width_s"]),
            **_window_and_power(rng, cfg, "pulse", duration)}


def sample_comb(rng, cfg, duration):
    c, f_max = cfg["classes"]["comb"], cfg["scene"]["f_max_hz"]
    n_tones = draw(rng, c["n_tones"])
    spacing = draw(rng, c["spacing_hz"])
    f_first = rng.uniform(-f_max, f_max - (n_tones - 1) * spacing)  # last tone stays in band
    return {"f_first": f_first, "spacing": spacing, "n_tones": n_tones, "ramp": c["ramp_s"],
            **_window_and_power(rng, cfg, "comb", duration)}


def sample_ofdm(rng, cfg, duration):
    c, f_max = cfg["classes"]["ofdm"], cfg["scene"]["f_max_hz"]
    bandwidth = draw(rng, c["bandwidth_hz"])
    freq = rng.uniform(-f_max + bandwidth / 2, f_max - bandwidth / 2)  # whole block in band
    return {"freq": freq, "bandwidth": bandwidth,
            "subcarrier_spacing": c["subcarrier_spacing_hz"], "ramp": c["ramp_s"],
            **_window_and_power(rng, cfg, "ofdm", duration)}


def sample_fsk(rng, cfg, duration):
    c, f_max = cfg["classes"]["fsk"], cfg["scene"]["f_max_hz"]
    deviation = draw(rng, c["deviation_hz"])
    freq = rng.uniform(-f_max + deviation, f_max - deviation)  # both frequencies in band
    return {"freq": freq, "deviation": deviation,
            "symbol_duration": draw(rng, c["symbol_duration_s"]), "ramp": c["ramp_s"],
            **_window_and_power(rng, cfg, "fsk", duration)}


# Anomaly class -> (function drawing its parameters, function adding it to the signal)
ANOMALY_TYPES = {
    "tone": (sample_tone, add_tone),
    "chirp": (sample_chirp, add_chirp),
    "pulse": (sample_pulse, add_pulse),
    "comb": (sample_comb, add_comb),
    "ofdm": (sample_ofdm, add_ofdm),
    "fsk": (sample_fsk, add_fsk),
}

# These classes draw random symbols themselves, so they also need the random generator
NEEDS_RNG = {"ofdm", "fsk"}

# Fixed class ids used in the masks; 0 is the background. Never renumber existing classes.
CLASS_IDS = {"tone": 1, "chirp": 2, "pulse": 3, "comb": 4, "ofdm": 5, "fsk": 6}


def anomaly_pixels(p_anomaly, fraction):
    """Mask of the strongest spectrogram pixels that together hold `fraction` of the energy."""
    values = np.sort(p_anomaly.ravel())[::-1]
    n_pixels = np.searchsorted(np.cumsum(values), fraction * values.sum()) + 1
    return p_anomaly >= values[n_pixels - 1]


def local_isr_gain(p_anomaly, p_background, isr, fraction):
    """Amplitude factor that makes the anomaly's local ISR (on the spectrogram) equal isr [dB].

    Local ISR = sum of anomaly power / sum of background power, both over the anomaly's pixels.
    Measuring there, and not over the whole recording, makes isr mean the same contrast
    whether the anomaly is narrow or wide, on top of the LTE channel or in an empty band.
    """
    mask = anomaly_pixels(p_anomaly, fraction)
    measured = np.sum(p_anomaly[mask]) / np.sum(p_background[mask])
    # Power grows with amplitude squared, so one rescaling hits the target exactly
    return np.sqrt(10 ** (isr / 10) / measured)


def add_random_anomalies(iq, fs, rng, cfg, p_background):
    """Add random anomalies to iq, as set in cfg.

    p_background: power map of the clean iq (needed for the local ISR).
    Returns (new iq, list of anomaly dicts, list of power maps of each anomaly alone).
    """
    duration = len(iq) / fs
    n_anomalies = draw(rng, cfg["scene"]["num_anomalies"])
    weights = cfg["scene"]["class_weights"]
    classes = list(weights)
    p = np.array([weights[k] for k in classes]) / sum(weights.values())

    isr_mode = cfg["power"]["isr_mode"]
    if isr_mode not in ("global", "local"):
        raise ValueError(f'isr_mode must be "global" or "local", not "{isr_mode}"')

    mixed = iq.copy()
    anomalies = []
    p_anomalies = []
    for _ in range(n_anomalies):
        kind = str(rng.choice(classes, p=p))
        sample_params, add_anomaly = ANOMALY_TYPES[kind]
        params = sample_params(rng, cfg, duration)

        extra = {"rng": rng} if kind in NEEDS_RNG else {}
        # Each anomaly is made against the clean recording, so `isr` always refers to the
        # recording itself and not to the anomalies added before it
        anomaly_alone = add_anomaly(iq, fs, **params, **extra) - iq
        p_anomaly, _ = power_map(anomaly_alone, fs)
        if isr_mode == "local":
            gain = local_isr_gain(p_anomaly, p_background, params["isr"],
                                  cfg["power"]["local_fraction"])
            anomaly_alone = anomaly_alone * gain
            p_anomaly = p_anomaly * gain ** 2
        mixed += anomaly_alone

        anomalies.append({"type": kind, "class_id": CLASS_IDS[kind], **params})
        p_anomalies.append(p_anomaly)
    return mixed, anomalies, p_anomalies


def generate_spectrogram(path, cfg, rng):
    """One spectrogram with random anomalies and its labels, from the capture at `path`.

    Returns a dict:
        image_db       float (400, 400) spectrogram in dB, rows = time, columns = frequency
        semantic_mask  uint8 (400, 400) class id per pixel, 0 = background
        anomalies      list of dicts: type, class_id, parameters, visible, mask_pixels and,
                       if visible, the box in pixels (x0, y0, x1, y1) and in Hz / s
        duration_ms    time covered by the image
    """
    fs = cfg["capture"]["sample_rate_hz"]
    threshold_db = cfg["labels"]["mask_threshold_db"]
    iq = read_capture(path, cfg["capture"]["num_samples"])
    p_background, _ = power_map(iq, fs)

    mixed, anomalies, p_anomalies = add_random_anomalies(iq, fs, rng, cfg, p_background)
    for anomaly, p_anomaly in zip(anomalies, p_anomalies):
        anomaly.update(anomaly_labels(p_anomaly, p_background, threshold_db, fs))

    class_ids = [a["class_id"] for a in anomalies]
    image_db, duration_ms = iq_to_spectrogram(mixed, fs)
    return {"image_db": image_db,
            "semantic_mask": semantic_mask(p_anomalies, class_ids, p_background, threshold_db),
            "anomalies": anomalies,
            "duration_ms": duration_ms}
