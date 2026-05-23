# 🌊 SigFlow

[![C++ Toolchain](https://img.shields.shields.shields.shields.shields.shields.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/20)
[![Python Version](https://img.shields.shields.shields.shields.shields.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-green.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Compiler](https://img.shields.shields.shields.shields.shields.shields.shields.io/badge/Compiler-MSVC%20%28VS%202022%29-orange.svg?style=flat-square&logo=microsoft-visual-studio)](https://visualstudio.microsoft.com/)
[![License](https://img.shields.shields.shields.shields.shields.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

**SigFlow** is a high-performance C++20 and Python RF signal processing and channel simulation toolkit. Designed for speed, reproducibility, and visual machine learning integration, SigFlow brings together hardware-accelerated DSP engines, complex multipath fading channel models, and modern Python deep-learning ready dataset pipelines.

---

## 🚀 Key Features

*   **⚡ Native C++20 DSP Engine**: Fully vectorized signal pipelines compiled with the MSVC toolchain, incorporating dynamic multithreading via **OpenMP**.
*   **🔗 Modern Python Bindings**: Seamless integration between C++ arrays and NumPy `complex64` buffers using **nanobind** for zero-copy memory transfers.
*   **🔬 Accurate RF Channel Simulator**: Real-time simulation of:
    *   **AWGN** (Additive White Gaussian Noise) with precise SNR settings.
    *   **Rayleigh Fading** (flat and frequency-selective multi-tap fading) supporting block-fading channel coherence.
    *   **Doppler Shift** with continuous, time-variant phase rotation.
*   **📦 Clean Third-Party Integration**: Direct integration of **FFTW3** and **Google Test** using CMake's `FetchContent`. No external dependencies or package managers required.
*   **🧠 Deep Learning Ready**: Complete dataset generation, pilot-aided channel estimation, MMSE equalization, and feature extraction (moments, instant frequency, histograms, log PSD bins) ready for neural network training.

---

## 🛠️ Tech Stack & Architecture

```mermaid
graph TD
    A[Python Dataset Generation / ML] -->|NumPy ndarray complex64| B(nanobind Bridge)
    B -->|std::span| C[libdsp C++20 Engine]
    C -->|OpenMP Parallelism| D[Channel Simulation]
    C -->|FFTW3 Engine| E[FFT Processing]
    C -->|GTest Suite| F[C++ Unit Tests]
```

*   **C++ Compiler**: Microsoft Visual C++ (`cl.exe` / VS 2022) with C++20 standards enabled.
*   **Parallelization**: OpenMP multithreading (`/openmp`).
*   **Fourier Transform**: FFTW3 (automatically fetched, linked dynamically with generated standard MSVC import libraries).
*   **Python Bindings**: nanobind 2.12+.
*   **Test Runner**: Google Test (for C++ targets), Pytest (for Python targets).
*   **Environment**: Python `uv` package manager.

---

## 📦 Getting Started

### 1. Prerequisites
- **Windows OS**
- **Visual Studio 2022** (with *Desktop development with C++* workload)
- **uv** (recommended) or **Python 3.12+**

### 2. Setting Up the Virtual Environment
Using `uv` to sync dependencies and set up the workspace:
```powershell
uv sync
```

### 3. Building the C++ Engine & Library
Run the pre-configured PowerShell build script to compile the core C++ engine, generate MSVC import libraries, build the nanobind module, and run the C++ unit tests:
```powershell
.\libdsp\build.ps1 -Clean
```
This builds `python_src/libdsp.pyd` and places `libfftw3f-3.dll` into the output folder automatically.

---

## 🧪 Testing

SigFlow includes a comprehensive two-tier test suite covering the underlying C++ code and the high-level Python utilities.

### Running C++ Tests
To run only the C++ tests (testing raw DSP routines and channel models):
```powershell
.\libdsp\build.ps1 -TestOnly
```

### Running Python Tests
Our extended test suite is fully configured to run under `uv` without any configuration collision:
```powershell
uv run pytest
```
*Tests verify:*
- FFT energy conservation & conjugate symmetry.
- Rayleigh fading amplitude distributions and block fading coherence.
- Doppler phase rotation drift.
- QPSK constellations and BER (Bit Error Rate) calculations.
- Pilot-aided least-squares channel estimation and MMSE equalizer performance.
- Dataset generation shapes, formats (complex or stacked real-imag), and ML feature extraction.

---

## 📖 Usage Examples

### Python: Generating a Labeled RF Dataset
Generate a high-fidelity dataset of multiple modulations under Rayleigh fading and frequency offsets, ready for training classification networks:

```python
import python_src.dataset as dataset

# Configuration
modulations = ["BPSK", "QPSK", "8PSK", "16QAM"]
snr_levels = [5.0, 10.0, 15.0, 20.0]

# Generate synthetic dataset (stacked real/imag IQ windows)
X, y, meta = dataset.generate_dataset(
    mod_list=modulations,
    snr_db_list=snr_levels,
    examples_per_snr=100,
    samples_per_example=256,
    apply_flat_fading=True,
    fading_std=0.5,
    return_complex=False
)

# X shape: (1600, 256, 2) -> 1600 examples, 256 samples, I and Q channels
print("Dataset Shape:", X.shape)

# Extract statistics, instantaneous features, histograms, and PSD bins
features = dataset.extract_features(X)
print("Features Matrix Shape:", features.shape)  # Shape: (1600, N_features)
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
