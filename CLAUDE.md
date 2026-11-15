# CLAUDE.md

Instructions for Claude Code working in this repository. Read this file and `docs/DESIGN.md`
before writing any code.

## What this project is

`rfsynth` generates a synthetic dataset for anomaly segmentation and detection in radio-frequency
spectrograms. Backgrounds are real recordings: WASD captures stored as `.bin` files with raw IQ
samples, about 21 ms each. Anomalies (tones, chirps, pulse trains, and new classes added over
time) are generated as complex IQ signals, added to the real IQ, and only then turned into
spectrograms. Every label is derived from the signals themselves: per-anomaly power maps,
segmentation masks, bounding boxes.

The output is a dataset that other people will load and split themselves. The generator does not
assign train/validation/test splits. It does record the columns needed to split correctly
(see `docs/DESIGN.md`, section "Splitting is the user's job").

All code, comments and documentation are in English.

## Code style: simple, readable, few dependencies

This is the most important section. The code will be read by people who are not signal-processing
specialists. It must stay easy to read and easy to change. When in doubt, pick the more boring
option.

### Dependencies

Allowed:

- Python 3.11 or newer and its standard library (`argparse`, `csv`, `json`, `tomllib`,
  `multiprocessing`, `pathlib`, `dataclasses`, `hashlib`)
- `numpy`
- `scipy` (only `scipy.signal`, and only where numpy alone would be clearly worse)
- `matplotlib` (only in validation and figure scripts, never in the generator itself)
- `pytest` (tests only)

Not allowed without asking first: `pandas`, `pyarrow`, `polars`, `h5py`, `zarr`, `torch`, `jax`,
`numba`, `pydantic`, `hydra`, `omegaconf`, `click`, `typer`, `rich`, `tqdm`, `attrs`, and any
package not on the allowed list. If something seems to need a new dependency, stop and explain
why, with the plain-numpy alternative next to it. Do not add it on your own.

### How the code should look

- **Plain functions and frozen dataclasses.** No class hierarchies, abstract base classes,
  metaclasses, decorator-based registries, plugin systems or dependency injection. The anomaly
  registry is a plain dictionary written out by hand.
- **Short functions with one job each.** If a function needs a comment to separate its parts,
  split it.
- **Physical units in names.** `duration_s`, `bandwidth_hz`, `snr_db`, `f_start_hz`. Parameters are
  in hertz and seconds, never in pixels. Pixels appear only in `stft.py` and `labels.py`.
- **Explicit randomness.** Every function that draws random numbers takes an
  `rng: np.random.Generator` argument. No global seeds, no `np.random.*` module-level calls, no
  hidden state.
- **Type hints on every public function**, plus a docstring that states what the function does,
  its units, and the array shapes and dtypes.
- **Comments explain why, not what.** Physics belongs in comments: why there is an edge ramp, why
  SNR is measured inside the anomaly's support.
- **Readability before speed.** Vectorise with numpy where it is natural. No micro-optimisations,
  no caching layers and no parallelism inside functions. Parallelism exists in exactly one place:
  the shard loop in `scripts/render.py`.
- **Configuration in TOML** (`configs/*.toml`), read with `tomllib`. Parameter distributions live
  in the config, not in code. No defaults hidden deep inside functions; the config is complete.
- **Command-line scripts use `argparse`** and are thin. They parse arguments and call functions
  from `src/rfsynth/`.
- **No clever tricks.** No one-liners that need a second read, no nested comprehensions, no
  `functools` acrobatics, no monkey-patching.

### Tests

- `pytest`, in `tests/`. Every anomaly class gets a test that checks its physics. For example: a
  tone at frequency f peaks in the expected column; a pulse train has the expected number of
  bursts; `render` output has unit power over its active interval.
- Determinism test: the same seed and sample id give byte-identical outputs.
- A scene without anomalies produces an image identical to the background-only image.
- Tests use short synthetic signals so the suite runs in seconds. They do not need the real
  `.bin` files.

## How to work in this repository

- **One piece at a time.** Build and test one module before starting the next. Follow the
  milestone order in `docs/DESIGN.md`. Do not scaffold empty modules for later milestones.
- **Ask, do not guess.** If the data format, a physical definition or the scope is unclear, stop
  and ask. In particular, do not guess the `.bin` sample format (see "Facts to verify" below).
- **Never modify the raw `.bin` files.** Open them read-only. All output goes to a separate output
  directory given on the command line.
- **Keep `docs/DESIGN.md` true.** If an implementation decision changes something the design
  states, update the design in the same change and say so.
- **Show results, not claims.** After implementing an anomaly class or a labelling step, produce a
  small figure (spectrogram plus mask) with `scripts/validate.py` so it can be inspected by eye.

## Facts to verify before writing the IQ reader

These are not yet confirmed. Check them against the actual files and ask if anything is ambiguous:

1. Sample format of the `.bin` files: dtype (`int16`, `float32`, ...), interleaved I/Q or
   separate blocks, byte order.
2. Sample rate. Expected 61.44 MS/s; confirm it is the same for all files.
3. Exact number of samples per file, and its distribution across files. Lengths are "about 21 ms",
   so the generator crops to one fixed length N (see `docs/DESIGN.md`).
4. Where the band name and the centre frequency come from: file name, directory, or a separate
   metadata file.
5. Whether capture timestamps can be parsed from file names (e.g.
   `IQ_2024-07-24T21-31-10_779570802.bin`).
