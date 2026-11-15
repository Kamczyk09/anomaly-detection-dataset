# Design: synthetic RF anomaly dataset

This document specifies what the generator does and why. `CLAUDE.md` says how the code should be
written; this file says what it should compute. Statements marked **To verify** depend on the
actual `.bin` files and must be confirmed before they are relied on.

## 1. Overview

```text
.bin captures ──► catalogue ──► scenes.jsonl ──► render ──► images, masks, ratio maps, CSVs
(real IQ,         (index of     (what to insert,  (IQ mix, STFT,
 read-only)        captures)     with which        labels, shards)
                                 parameters)
```

Three stages, each a separate script, each with a file as output:

1. **Catalogue** (`scripts/build_catalogue.py`): index every `.bin` file once.
2. **Scene sampling** (`scripts/sample_scenes.py`): draw the full specification of every sample
   (which capture, which anomalies, all their parameters) and write it to `scenes.jsonl`. Nothing
   is rendered yet.
3. **Rendering** (`scripts/render.py`): turn each scene into a spectrogram and its labels.

Separating sampling from rendering has three benefits. The parameter distributions can be inspected
before anything expensive runs. Any single sample can be re-rendered from its line in
`scenes.jsonl`. And the rendering is a pure function of the scene, which makes it deterministic
and testable.

## 2. Backgrounds

Each `.bin` file is one capture of about 21 ms of raw IQ. Files are used as they are: no cutting
into segments, no train/test assignment.

**To verify:** sample format, sample rate (expected 61.44 MS/s), exact lengths, and where the band
name, centre frequency and timestamp come from. See `CLAUDE.md`, "Facts to verify".

**Fixed length.** All captures are cropped to one length `N` samples, taken from the start of the
file. `N` is chosen after the catalogue shows the length distribution. A value consistent with the
existing segmentation dataset is `N = 1,299,200` (21.15 ms at 61.44 MS/s). Files shorter than `N`
are excluded and listed in the build report.

**Catalogue** (`catalogue.csv`), one row per file: `source_capture` (file name), `path`, `band`,
`capture_time` (if parseable), `num_samples`, `included` (bool), `exclude_reason`.

**Reuse of captures.** The config parameter `variants_per_capture` (default 1) sets how many samples
are made from each capture. With 1, every background appears exactly once in the dataset. With more
than 1, the same background appears with different anomalies. This is the leakage pattern of the
old segmentation dataset, so users must then split by `source_capture` (section 9).

## 3. Spectrogram transform

One function, `stft.power_map(iq) -> (400, 400) float64`, is used for **everything**: the image, the
background power map and every anomaly power map. Labels and images can only line up if they come
from the same transform.

Proposed parameters (in the config), chosen so that both axes reduce by integer block averaging:

| step | value | result |
|---|---|---|
| window | periodic Hann, `nfft = 4096` | 15 kHz per frequency bin at 61.44 MS/s |
| frequency crop | central 3200 bins after `fftshift` | 48 MHz, guard bands removed |
| frequency averaging | blocks of 8 bins | 400 columns, 120 kHz each |
| hop | `hop = (N - nfft) // 1599`, i.e. 809 for the `N` above | at least 1600 frames; keep the first 1600 |
| time averaging | blocks of 4 frames | 400 rows, about 52.7 µs each |

Averaging is done on **linear power** (`|X|²`), never on decibels. The axis order is `(y, x)` =
`(time, frequency)`. Row 0 is the start of the capture; column 0 is the lowest frequency.

**Stored image** = `10 * log10(power_map(mixed_iq))`, saved as `float16`. The value scale is
therefore documented exactly: power in dB, relative to an arbitrary but fixed reference.

## 4. Anomaly classes

### 4.1 Contract

Each class is one module in `src/rfsynth/anomalies/` with exactly three things:

```python
@dataclass(frozen=True)
class ChirpParams:
    f_start_hz: float      # offset from the centre frequency
    f_end_hz: float
    t_start_s: float
    duration_s: float
    bandwidth_hz: float    # instantaneous bandwidth: the "thickness" of the line
    edge_ramp_s: float     # raised-cosine on/off ramps
    seed: int              # for any internal randomness, so render() stays pure

def sample(rng: np.random.Generator, cfg: dict, fs_hz: float, n: int) -> ChirpParams: ...

def render(p: ChirpParams, n: int, fs_hz: float) -> np.ndarray:
    """Return complex64 IQ of shape (n,): the anomaly alone, zero outside its active
    interval, mean power 1 over the active interval."""
```

The registry in `anomalies/__init__.py` is a hand-written dictionary from class name to module,
plus a hand-written dictionary of fixed integer class ids (0 = background). Ids never depend on
the config, so the same class has the same id in every dataset version.

### 4.2 Rules for parameters

- **Physical units only** (Hz, s). Thickness is a bandwidth or a duration, never a pixel width.
  Its appearance in the image follows from the transform.
- **Amplitude is not a class parameter.** `render` returns unit power; strength is set by the SNR in
  `mixing.py`, identically for every class.
- **Channel impairments are not class parameters.** CFO, phase noise and fading live in
  `impairments.py` and are drawn for all classes the same way. Otherwise each class would carry its
  own notion of "strength" or "noisiness", which a model could learn instead of shape.
- **Edge ramps are mandatory.** A rectangular on/off switch spreads energy over the whole band and
  makes every burst look wideband.
- **Distributions live in the config.** `sample()` reads ranges from `cfg`. Quantities spanning
  more than a decade (durations, bandwidths) are drawn log-uniformly.
- **Classes must stay distinguishable.** Neighbouring classes must not blend into each other through
  their parameter ranges. A chirp with slope near 0 is a tone, a tone with a 5 MHz bandwidth is a
  jammer, a pulse train with long pulses is an interrupted tone. The config keeps a gap between
  such ranges, and `validate.py` checks it.
- **Below one pixel, variation is invisible.** One pixel is 120 kHz × 52.7 µs. Bandwidths below
  about 120 kHz or durations below about 50 µs change brightness but not shape. Ranges should start
  around one pixel unless brightness variation is the intent.

### 4.3 Classes in version 1

| class | parameters |
|---|---|
| `tone` | `f_offset_hz`, `t_start_s`, `duration_s`, `bandwidth_hz`, `edge_ramp_s`, `seed` |
| `chirp` | `f_start_hz`, `f_end_hz`, `t_start_s`, `duration_s`, `bandwidth_hz`, `edge_ramp_s`, `seed` |
| `pulse` | `f_offset_hz`, `t_start_s`, `pulse_width_s`, `pri_s`, `n_pulses`, `pri_jitter_frac`, `bandwidth_hz`, `edge_ramp_s`, `seed` |

A non-zero `bandwidth_hz` is made by modulating the carrier with complex noise low-pass filtered to
that bandwidth, using `seed`. Initial ranges are proposals in `configs/v1.toml`, to be tuned with
the validation figures.

### 4.4 Candidate classes for later versions

Each is added on its own, with tests and a validation figure, after version 1 works end to end:
frequency hopping, partial-band noise jamming, swept (sawtooth) jammer, non-linear FM, LFM radar
pulse, foreign digitally modulated carrier, impulsive wideband noise, tone comb or harmonics, and
signal dropout (a notch filter applied to the background, i.e. not additive).

## 5. Scenes

A scene is everything needed to render one sample. It is drawn with
`rng = np.random.default_rng([global_seed, sample_id])`, so it depends only on the global seed and
the sample id, never on the order of processing.

Drawn per sample, with distributions from the config:

- the capture and the variant index;
- whether the sample is normal (probability `p_normal`);
- the number of anomalies (e.g. 1 to 3);
- per anomaly: class (by `class_weights`), class parameters, `snr_db` (continuous, e.g. uniform on
  [−10, +10] dB), and impairment parameters with their own seed.

Anomalies may overlap. Anomalies may extend past the image edge; they are then flagged `truncated`.

`scenes.jsonl` holds one JSON object per line:

```json
{"sample_id": 17, "source_capture": "IQ_2024-07-24T21-31-10_779570802.bin", "variant": 0,
 "anomalies": [{"class": "chirp", "params": {"f_start_hz": -3.1e6, "...": "..."},
                "snr_db": -4.2, "impairments": {"cfo_hz": 1200.0, "seed": 99}}]}
```

## 6. Rendering one sample

```python
bg = read_capture(path)[:N]                      # complex64, read-only source
P_bg = power_map(bg)
mixed = bg.copy()
maps = []
for a in scene.anomalies:
    s = render(a)                                # unit power, anomaly alone
    s = apply_impairments(s, a.impairments)
    P_s = power_map(s)
    gain = snr_gain(P_s, P_bg, a.snr_db)         # section 7
    mixed += np.sqrt(gain) * s
    maps.append(gain * P_s)                      # power map scales linearly with gain
image_db = 10 * np.log10(power_map(mixed))
labels = derive_labels(maps, P_bg)               # section 8
```

Receiver effects (clipping, quantisation) would be applied to `mixed` before the final transform.
They are not part of version 1.

## 7. SNR definition

The background is a real signal, not white noise, so SNR is measured **where the anomaly is**,
not over the whole band. Otherwise a narrow tone at −10 dB would be clearly visible and a
wideband anomaly at the same SNR invisible, and thickness would silently control visibility.

- **Support** `S`: the smallest set of pixels of `P_s` that holds 95% of the anomaly's energy.
- **SNR** = `10 * log10( sum(P_s[S]) / sum(P_bg[S]) )`.
- `snr_gain` returns the factor that brings this ratio to the target `snr_db`.

Because SNR is computed on the same 400 × 400 power maps as the labels, the SNR, the image and the
masks are mutually consistent by construction.

## 8. Labels

All labels come from one quantity per anomaly `k`, the **ratio map**

```text
R_k = P_k / max(P_bg, floor)     floor = 1st percentile of P_bg in this image
```

expressed in dB. It says, per pixel, how strong anomaly `k` is relative to the background there.

- **Ratio maps** are stored per anomaly as `uint8`: `code = clip(round((R_dB + 40) * 2), 0, 255)`,
  i.e. −40 to +87.5 dB in 0.5 dB steps. Users can threshold them however they like.
- **Instance mask** of anomaly `k`: `R_k_dB > threshold_db` (config, default 0 dB: the anomaly is at
  least as strong as the background).
- **Semantic mask**: per pixel, the class id of the anomaly with the largest `R_k`, if that exceeds
  the threshold; otherwise 0. This resolves overlaps, which the old masks could not.
- **Bounding box** of anomaly `k`: the tight box around its instance mask, in pixels and in physical
  units (`f_lo_hz`, `f_hi_hz`, `t_start_s`, `t_end_s`). Boxes are derived from masks, never drawn
  from parameters. In the old dataset a nominal box covered 2,915 pixels where the mask covered 199.
- **Invisible anomalies**: an anomaly whose instance mask is empty at the default threshold stays
  in the metadata with `visible = false` and an empty box. It is still part of the ground truth of
  what was inserted.

Pulse trains and hopping signals get one box per instance. Per-segment boxes can be derived from
the ratio maps by connected components; they are not stored in version 1.

## 9. Output format and splitting

```text
<out>/
├── config.toml              copy of the config used
├── build_report.json        config hash, git revision, library versions, counts, excluded files
├── catalogue.csv
├── scenes.jsonl
├── samples.csv              one row per sample
├── anomalies.csv            one row per anomaly
└── shards/shard_00000/
    ├── images.npy           float16 (n, 400, 400), power in dB
    ├── semantic.npy         uint8   (n, 400, 400), class id per pixel at the default threshold
    └── ratio.npy            uint8   (m, 400, 400), one ratio map per anomaly in this shard
```

All arrays are ordinary NumPy files written with `np.save` and read with `np.load`. No headerless
files.

`samples.csv`: `sample_id`, `shard`, `index_in_shard`, `source_capture`, `band`, `capture_time`,
`variant`, `num_anomalies`, `num_visible`, `classes` (semicolon-separated).

`anomalies.csv`: `sample_id`, `anomaly_index`, `class`, `class_id`, `snr_db`, `snr_db_measured`,
`params_json`, `impairments_json`, `shard`, `ratio_index`, `visible`, `mask_pixels`, `x0`, `y0`,
`x1`, `y1`, `f_lo_hz`, `f_hi_hz`, `t_start_s`, `t_end_s`, `truncated`.

### Splitting is the user's job

The generator assigns no splits. The README of every generated dataset states:

- split by `source_capture` whenever `variants_per_capture > 1`, so no background is on both sides;
- captures recorded close in time may have near-identical backgrounds; `capture_time` lets users
  group them;
- hold out whole `band`s to test generalisation to unseen spectrum occupancy.

## 10. Running at scale

- `scenes.jsonl` is processed in shards of `shard_size` samples (config, e.g. 1000).
- `scripts/render.py --workers K` renders shards in parallel with `multiprocessing.Pool`. One shard
  is one task, and this is the only parallelism in the code.
- **Resumable.** A finished shard is skipped. Each shard is written to a temporary directory and
  renamed when complete, so an interrupted run never leaves half a shard.
- Results do not depend on `K` or on the order in which shards finish.

## 11. Validation

`scripts/validate.py` produces, from a rendered dataset:

1. a gallery: one row per class, one column per SNR bin, showing the image and the semantic mask;
2. histograms of what the model actually sees: mask pixels per class, share of visible anomalies
   per SNR bin, positions on both axes, share of truncated anomalies;
3. measured versus target SNR (they should agree within rounding);
4. the contrast check from the old dataset: mean image value inside versus outside the masks, per
   SNR bin, which must be clearly positive and grow with SNR;
5. a check that class parameter ranges keep the gaps of section 4.2.

Diversity is judged by these plots, not by the number of parameters.

## 12. Milestones

Each milestone ends with passing tests and a figure. Do not start the next one before.

1. IQ reader and catalogue (after verifying the facts in `CLAUDE.md`); `stft.power_map`; a figure
   of one raw background.
2. `tone`, `mixing.snr_gain`, `labels.derive_labels`; a figure of one tone over a real background
   at several SNRs, with masks.
3. Scene sampling, `scenes.jsonl`, serial `render.py`, writer, `build_report.json`; one shard end
   to end.
4. `chirp` and `pulse`.
5. Parallel shards and resumability.
6. Channel impairments: CFO, phase noise, slow fading.
7. New classes from section 4.4, one at a time.
8. Optional: receiver effects (clipping, quantisation).

## 13. Open questions

- The exact `N`, after the catalogue shows the length distribution.
- `variants_per_capture`: 1, or more than 1 to multiply the dataset size.
- The default mask threshold (0 dB proposed); worth checking visually at low SNR.
- Whether to store IQ for a small subset, for later models that take IQ as input. A full IQ copy
  would be about 10 MB per sample.
