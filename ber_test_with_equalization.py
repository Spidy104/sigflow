import numpy as np
import sys
sys.path.insert(0, "python_src")
import libdsp


def generate_qpsk(n_syms):
    """Generate QPSK symbols"""
    bits_i = np.random.randint(0, 2, n_syms)
    bits_q = np.random.randint(0, 2, n_syms)
    return (2 * bits_i - 1) + 1j * (2 * bits_q - 1)

def hard_decision_qpsk(rx):
    """Hard decision QPSK demodulation"""
    real = np.where(rx.real >= 0, 1, -1)
    imag = np.where(rx.imag >= 0, 1, -1)
    return real + 1j * imag

def compute_ber(tx, rx):
    """Calculate BER"""
    tx_sym = hard_decision_qpsk(tx)
    rx_sym = hard_decision_qpsk(rx)
    return np.mean(tx_sym != rx_sym)

def estimate_channel_perfect(tx, rx):
    """Genie-aided LS channel estimate per 100-symbol coherence block."""
    coherence_symbols = 100
    n = len(tx)
    h_est = np.zeros(n, dtype=np.complex64)
    for block_start in range(0, n, coherence_symbols):
        block_end = min(block_start + coherence_symbols, n)
        h_est[block_start:block_end] = np.mean(rx[block_start:block_end] / tx[block_start:block_end])
    return h_est

def equalize_channel(rx, h_est):
    """Zero-forcing equalization."""
    return rx / h_est

# Test parameters
N = 200000
snr_db = 10.0

print(f"BER Comparison at SNR = {snr_db} dB (QPSK, {N} symbols)")
print("=" * 70)

# Test 1: AWGN only
tx = generate_qpsk(N).astype(np.complex64)
rx = libdsp.apply_channel(tx, snr_db, n_taps=0, doppler_hz=0.0)
ber_awgn = compute_ber(tx, rx)
print(f"AWGN only (n_taps=0)                        | BER = {ber_awgn:.2e}")

# Test 2: Rayleigh flat fading WITHOUT equalization
tx = generate_qpsk(N).astype(np.complex64)
rx = libdsp.apply_channel(tx, snr_db, n_taps=1, doppler_hz=0.0)
ber_rayleigh_no_eq = compute_ber(tx, rx)
print(f"Rayleigh Flat (no equalization)             | BER = {ber_rayleigh_no_eq:.2e}")

# Test 3: Rayleigh flat fading WITH perfect channel estimation
h_est = estimate_channel_perfect(tx, rx)
rx_equalized = equalize_channel(rx, h_est)
ber_rayleigh_with_eq = compute_ber(tx, rx_equalized)
print(f"Rayleigh Flat (WITH perfect equalization)   | BER = {ber_rayleigh_with_eq:.2e}")
print(f"  Degradation vs AWGN: {ber_rayleigh_with_eq / ber_awgn:.1f}x")

print("\n" + "=" * 70)
print(f"AWGN:              BER = {ber_awgn:.2e}")
print(f"Rayleigh no eq:    BER = {ber_rayleigh_no_eq:.2e}")
print(f"Rayleigh + ZF eq:  BER = {ber_rayleigh_with_eq:.2e} ({ber_rayleigh_with_eq / ber_awgn:.1f}x worse than AWGN)")

