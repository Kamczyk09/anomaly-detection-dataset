"""Labels derived from the signals themselves: masks and bounding boxes.

P_k (anomaly alone, no background)
P_bg (background, no anomaly)

Both are (400, 400) linear power maps from spectrogram.power_map, so the
labels line up with the image pixel by pixel. A pixel belongs to anomaly k when P_k / P_bg is
above a threshold: the anomaly is that much stronger than the background in that pixel.

IMPORTANT:
- pixel belongs to anomaly k when P_k / P_bg is above a threshold
- labels (and above comparsions) are derived from signals not parameters --> for every anomaly we create spectrogram and compare it with background.
- when anomalies overlap, pixel is assigned anomaly class with highest ISR.
- we indicate when anomaly is not visible (visible = False) ; P_k / P_bg < threshold ==> no mask ==> no bounding box
"""
import numpy as np

from spectrogram import NFFT, HOP, FREQ_BINS_KEPT, FREQ_BLOCK, TIME_BLOCK


def instance_mask(p_anomaly, p_background, threshold_db):
    """Boolean (400, 400) mask: pixels where the anomaly is threshold_db above the background."""
    return p_anomaly > p_background * 10 ** (threshold_db / 10)


def semantic_mask(p_anomalies, class_ids, p_background, threshold_db):
    """uint8 (400, 400) mask: class id of the anomaly in each pixel, 0 = background.

    Where anomalies overlap, the pixel goes to the one with the largest P_k / P_bg.
    """
    semantic = np.zeros(p_background.shape, dtype=np.uint8)
    if len(p_anomalies) == 0:
        return semantic
    ratios = np.stack(p_anomalies) / p_background          # (n_anomalies, 400, 400)
    strongest = np.argmax(ratios, axis=0)
    above = np.max(ratios, axis=0) > 10 ** (threshold_db / 10)
    semantic[above] = np.array(class_ids)[strongest[above]]
    return semantic


def bounding_box(mask):
    """Tight box around the True pixels as pixel indices x0, y0, x1, y1 (inclusive), or None."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return {"x0": int(cols[0]), "y0": int(rows[0]), "x1": int(cols[-1]), "y1": int(rows[-1])}


def box_to_physical(box, fs):
    """Outer edges of a pixel box in frequency [Hz] (0 = centre frequency) and time [s]."""
    pixel_hz = fs / NFFT * FREQ_BLOCK              # 120 kHz at 61.44 MHz
    pixel_s = HOP * TIME_BLOCK / fs                # about 52.7 us
    f_min = -FREQ_BINS_KEPT / 2 * fs / NFFT        # -24 MHz, the left edge of column 0
    return {"f_lo_hz": f_min + box["x0"] * pixel_hz, "f_hi_hz": f_min + (box["x1"] + 1) * pixel_hz,
            "t_start_s": box["y0"] * pixel_s, "t_end_s": (box["y1"] + 1) * pixel_s}


def anomaly_labels(p_anomaly, p_background, threshold_db, fs):
    """Labels of one anomaly for its metadata: visible, mask_pixels and, if visible, its box."""
    mask = instance_mask(p_anomaly, p_background, threshold_db)
    box = bounding_box(mask)
    labels = {"visible": box is not None, "mask_pixels": int(mask.sum())}
    if box is not None:
        labels.update(box)
        labels.update(box_to_physical(box, fs))
    return labels
