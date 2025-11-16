# SigFlow

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![License](https://img.shields.io/badge/license-TBD-lightgrey)

SigFlow is a hybrid C++/Python sandbox for simulating wireless channels, generating synthetic IQ datasets, engineering RF features, and training ensemble classifiers for modulation recognition. The C++20 DSP core (FFTW3, Eigen3, pybind11) synthesizes Rayleigh/Rician fading, AWGN, and Doppler impairments; Python utilities orchestrate dataset generation, feature extraction, experimentation, and visualization.

## Table of contents

1. [Quickstart](#quickstart)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Repository layout](#repository-layout)
4. [Requirements & setup](#requirements--setup)
5. [Building the C++ core](#building-the-c-core)
6. [Dataset + feature generation](#dataset--feature-generation)
7. [Training classifiers](#training-classifiers)
8. [Inference & demos](#inference--demos)
9. [CLI reference](#cli-reference)
10. [Testing & validation](#testing--validation)
11. [Documentation & resources](#documentation--resources)
12. [Troubleshooting](#troubleshooting)

## Features

---

### Signal Generation

- QPSK baseband IQ generation  
- Configurable length, randomness, and normalization  

### DSP Block

- FIR filtering  
- FFT analysis  
- Feature extraction (temporal + spectral)  

### Channel Models

- AWGN with controllable SNR  
- Rayleigh fading (multipath convolution)  

### ML Block

- Random Forest modulation/SNR prediction  
- Plug-and-play featurization  
- Can be replaced with CNN/LSTM models  

### Metrics & Evaluation

- BER calculation  
- Latency measurement  
- Accuracy vs SNR  
- Automated CSV logging  

### Visualization

- Frequency spectra  
- Channel distortion plots  
- Performance graphs (BER/SNR/Latency)  

---

## Quickstart

1. **Clone & submodules**

```bash
git clone https://github.com/Spidy104/sigflow.git
cd sigflow
```

1. **Install prerequisites**

   - CMake ≥ 3.24, Ninja (optional but faster), a C++20 compiler (clang++ 14+/g++ 11+)
   - FFTW3 (double precision) and Eigen3 development headers
   - Python 3.10 with `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `h5py`

1. **Build + validate**

```bash
cmake -S libdsp -B libdsp/build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build libdsp/build
python validate_core.py
```

## Architecture at a glance

- **DSP core (`libdsp/`)** – C++20 modules for signal synthesis, fading channels, noise models, and feature extraction exposed to Python via pybind11.
- **Python pipeline (`python_src/`)** – dataset generation, feature computation, ML training, evaluation, and helper utilities.
- **Experiments & notebooks** – quick scripts (`ber_*.py`, `run_demo.py`) plus `python_src/training.ipynb` for exploratory model development.
- **Tests** – CTest targets exercise the C++ core, while Python smoke tests ensure bindings and utilities run end-to-end.

## System explanation

This repository implements a small end-to-end signal simulation and ML pipeline. At a high level:

- Signal generation: the C++ DSP core (`libdsp`) synthesizes baseband IQ traces (modulated symbols, pulse shaping). These traces include configurable impairments: AWGN, multipath taps (Rayleigh/Rician), and Doppler.
- Feature extraction: Python wrappers call into `libdsp` (pybind11) to compute time- and frequency-domain statistics (moments, constellation metrics, PSDs, cyclic features) and write feature matrices to `results/full_dataset/`.
- Training & inference: feature matrices feed into scikit-learn pipelines (Random Forests by default). Models are saved to `results/models/` and loaded for batched inference or single-shot predictions.
- Evaluation & logging: BER sweeps and diagnostics write CSV summaries (`results/logs.csv`, `results/fading_impact.csv`) and optional plots are emitted to `results/plots/` for visual inspection.

Data flow (simple):

1. `libdsp` -> generate IQ -> save NPZ
2. `python_src/extract_features.py` -> read NPZ -> write features.npz / CSV
3. `python_src/ml_inference.py --train` -> train model -> save model
4. `python_src/ml_inference.py --predict` -> load model -> predict -> append logs/plots

The design separates heavy DSP (C++) from experiment control and ML (Python) so you can iterate models quickly without rebuilding the core.

## Repository layout

| Path | Description |
| --- | --- |
| `libdsp/CMakeLists.txt` | CMake entry point that configures the DSP static library, pybind11 module, and CTest suites. |
| `libdsp/include/channel.h` / `channel.cpp` | Channel models (Rayleigh, Rician, AWGN taps, Doppler). |
| `libdsp/include/dsp.h` / `dsp.cpp` | DSP primitives (modulation, filtering, FFT wrappers) shared by bindings/tests. |
| `libdsp/src/binding.cpp` | pybind11 bindings that expose the C++ API as the `libdsp` Python module. |
| `libdsp/tests/test_channel.cpp` | Unit tests that validate fading/channel math; invoked through `ctest`. |
| `python_src/config.py` | Central place for dataset sizes, modulation sets, SNR grids, and feature toggles. |
| `python_src/dataset.py` | Generates IQ datasets by driving the C++ core, serializes into NPZ/HDF5. |
| `python_src/extract_features.py` | Converts raw IQ into engineered RF features and writes to `results/full_dataset`. |
| `python_src/channel_estimation.py` | Helper routines for estimating channel impulse responses or equalization taps. |
| `python_src/ml_inference.py` | Loads trained models, performs batch or single-shot inference on features/IQ. |
| `python_src/plot_utils.py` | Matplotlib helpers for constellation plots, BER curves, and fading visualizations. |
| `python_src/smoke_test_generate.py` | Quick end-to-end dataset + feature generation sanity check. |
| `python_src/training.ipynb` | Notebook for experimenting with feature importance and ensemble models. |
| `python_src/utils.py` | Shared IO helpers (NPZ/CSV), logging, deterministic seeding, etc. |
| `ber_sweep.py` | Sweeps BER vs. SNR/fading profiles using the Python pipeline. |
| `ber_test_correct.py` | Regression test that compares BER curves against analytical baselines. |
| `ber_test_with_equalization.py` | BER sweep that enables equalization paths for comparison. |
| `diagnose_fading.py` | Visualizes fading statistics and Doppler spectra for different channels. |
| `fading_test.py` | Minimal reproduction of fading-only scenarios for debugging. |
| `run_demo.py` | Produces a full dataset, trains a lightweight model, and reports live predictions. |
| `test_all.py` | Convenience launcher that runs CTest and Python smoke tests in one go. |
| `test_channel_simple.py` | Lightweight Python check for channel bindings without full dataset generation. |
| `validate_core.py` | Ensures the compiled `libdsp` module can be imported and basic transforms succeed. |
| `results/` | Captured datasets, logs, and CSV summaries generated by experiments (ignored in git). |
| `theory.md` | Notes on channel theory, signal models, and derivations referenced in the code. |

Additional folder details:

- `libdsp/build/` — out-of-source CMake build artifacts and binaries (ignored from Git). Contains the compiled pybind11 module used by Python tests.
- `results/full_dataset/` — full IQ datasets and their feature matrices (NPZ/HDF5). Large files are not checked into git; use this folder for local experiments.
- `results/plots/` — generated PNG/SVG plots (BER curves, PSDs, constellation snapshots) produced by the demo and diagnostic scripts.
- `configs/` — (optional) experiment configuration YAML files (not present by default; add your own to standardize runs).

If you'd like, I can add small example artifacts to `results/plots/` to make the README show images inline.

> Need a quick reference for another file? Search for the filename above and jump directly to its section.

## Requirements & setup

1. Install FFTW3 and Eigen3 packages (e.g., `sudo apt install libfftw3-dev libeigen3-dev`).
1. Install pybind11 headers (`pip install pybind11` or distribution packages).
1. Create and activate a Python 3.10 virtual environment, then install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # coming soon; for now install packages listed above
```

1. Export `PYTHONPATH` to include `python_src` if running scripts directly outside the repo root.

## Building the C++ core

```bash
cmake -S libdsp -B libdsp/build \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYBIND11_FINDPYTHON=ON
cmake --build libdsp/build
ctest --test-dir libdsp/build --output-on-failure
```

Tips:

- Pass `-DBUILD_SHARED_LIBS=ON` if you prefer shared artifacts.
- Use `-DCMAKE_PREFIX_PATH` to point CMake to non-system FFTW3/Eigen installs.
- The generated `libdsp.cpython-*.so` lives in `libdsp/build/src/`; `validate_core.py` dynamically adds it to `sys.path`.

## Dataset + feature generation

| Command | Purpose |
| --- | --- |
| `python python_src/smoke_test_generate.py` | Generates a small dataset + feature matrix to validate the full pipeline. |
| `python ber_sweep.py --snr 0 20 2` | Sweeps BER over 0–20 dB in 2 dB steps (arguments optional). |
| `python python_src/extract_features.py --dataset results/full_dataset/dataset.npz` | Converts IQ captures into engineered features. |

Artifacts end up in `results/` (ignored by git). Update `python_src/config.py` to change SNR grids, modulation families, or dataset sizes.

### Quick usage examples

Generate a small dataset and features (fast smoke test):

```bash
python python_src/smoke_test_generate.py --samples 1024 --snr 10
python python_src/extract_features.py --dataset results/full_dataset/dataset.npz --out results/full_dataset/features.npz
```

Train a baseline model (Random Forest) on the generated features:

```bash
python python_src/ml_inference.py --train --features results/full_dataset/features.npz --out results/models/baseline.joblib
```

Run inference and produce plots/logs:

```bash
python python_src/ml_inference.py --predict --model results/models/baseline.joblib --features results/full_dataset/features.npz --out results/logs.csv --plot-dir results/plots
```

Generate BER sweep and plot results:

```bash
python ber_sweep.py --snr-start 0 --snr-stop 20 --snr-step 2 --out results/logs_ber.csv --plot-dir results/plots
```

Example plot commands (Python snippet) — plot BER vs SNR from the CSV:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/logs_ber.csv')
plt.plot(df['snr_db'], df['ber'], marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.savefig('results/plots/ber_vs_snr.png', dpi=150)
```

## Training classifiers

- Open `python_src/training.ipynb` to explore feature importance, grid-search hyperparameters, and evaluate baseline models (Random Forest, XGBoost, LightGBM).
- For scripted training, use `python python_src/ml_inference.py --train --config configs/baseline.yaml` (placeholder config – adapt to your environment).
- Save trained models in `results/models/` and point inference scripts to that directory using CLI flags or config entries.

## Inference & demos

```bash
python run_demo.py --snr 10 --modulation qpsk
python python_src/ml_inference.py --model results/models/best.joblib --features results/full_dataset/features.npz
```

The demo script generates a synthetic channel trace, feeds it through the trained model, and prints/logs predicted modulation classes. Use `--plot` to enable matplotlib visualizations when running locally.

## CLI reference

| Script | Highlights |
| --- | --- |
| `validate_core.py` | Imports `libdsp`, runs FFT/BER sanity checks, and prints summary stats. |
| `test_all.py` | Calls CTest plus Python smoke tests; convenient pre-commit gate. |
| `ber_test_correct.py` | Compares empirical BER to closed-form curves; fails fast if drift detected. |
| `ber_test_with_equalization.py` | Runs the BER suite with and without equalization for regression tracking. |
| `diagnose_fading.py` | Dumps fading taps, Doppler spectra, and coherence times to CSV/plots. |

Run any script with `-h/--help` to see available options.

## Testing & validation

1. **C++ unit tests** – `ctest --test-dir libdsp/build` after configuring the build.
2. **Python smoke** – `python python_src/smoke_test_generate.py` ensures bindings and config defaults still work.
3. **Full-stack** – `python test_all.py` to run both suites sequentially; integrate with CI before pushing.

## Documentation & resources

- `theory.md` – derivations for BER curves, fading PDFs, and assumptions used in simulations.
- `results/logs.csv` + `results/fading_impact.csv` – example outputs referenced in the paper notebook.
- External references: Proakis (Digital Communications, 5e), Rappaport (Wireless Communications, 2e).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: libdsp` | Ensure `cmake --build` succeeded and the generated `.so` is on `PYTHONPATH`. `validate_core.py` temporarily amends `sys.path`; mimic that behavior in custom scripts. |
| `Could NOT find FFTW3` | Install `libfftw3-dev` or set `FFTW3_DIR`. Use `cmake -DFFTW3_DIR=/opt/fftw`. |
| Numerical drift in BER curves | Rebuild in Release mode, verify RNG seeds in `python_src/config.py`, and rerun `ber_test_correct.py`. |
| Matplotlib backend errors on headless servers | Export `MPLBACKEND=Agg` or run scripts with `--no-plot`. |

Questions or contributions? Open an issue or discussion on [GitHub](https://github.com/Spidy104/sigflow/issues).
