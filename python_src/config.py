# python_src/config.py
"""System-wide configuration parameters"""

# Signal parameters
SYMBOLS_PER_BLOCK = 10000
PILOT_INTERVAL = 100
PILOT_VALUE = 1.0 + 0.0j

# Channel parameters  
DEFAULT_SNR_DB = 10.0
DEFAULT_N_TAPS = 1  # Flat Rayleigh
DEFAULT_DOPPLER_HZ = 0.0

# ML parameters
ML_FEATURES = 20
ML_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM"]

# Benchmark parameters
SNR_SWEEP = list(range(0, 21, 2))  # 0 to 20 dB