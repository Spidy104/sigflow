#!/usr/bin/env python3
"""
BER vs SNR sweep for QPSK in multiple channel conditions.
Saves results to results/logs.csv and plots to results/ber_vs_snr.png
"""

import numpy as np
import sys
import csv
import os

# Use non-interactive backend to avoid segfaults in MSYS2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure we load the local libdsp
sys.path.insert(0, "python_src")
import libdsp


def generate_qpsk(n_syms: int, seed: int = None) -> np.ndarray:
    """Generate QPSK symbols with optional fixed seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    bits_i = np.random.randint(0, 2, n_syms)
    bits_q = np.random.randint(0, 2, n_syms)
    return (2 * bits_i - 1) + 1j * (2 * bits_q - 1)


def hard_decision_qpsk(rx: np.ndarray) -> np.ndarray:
    """Map received samples to nearest QPSK constellation point."""
    real = np.where(rx.real >= 0, 1, -1)
    imag = np.where(rx.imag >= 0, 1, -1)
    return real + 1j * imag


def compute_ber(tx: np.ndarray, rx: np.ndarray) -> float:
    """Compute Bit Error Rate for QPSK."""
    tx_sym = hard_decision_qpsk(tx)
    rx_sym = hard_decision_qpsk(rx)
    return np.mean(tx_sym != rx_sym)


def theoretical_qpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    """Theoretical BER for QPSK in AWGN: 0.5 * erfc(sqrt(SNR_linear/2))."""
    snr_lin = 10 ** (snr_db / 10)
    # Use tight approximation to avoid special functions
    return 0.5 * (1 - np.sqrt(snr_lin / (2 + snr_lin)))


def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Parameters
    n_syms = 100000  # Increase for better high-SNR accuracy
    snrs = np.arange(0, 13, 1)  # 0 to 12 dB

    # Test multiple channel conditions
    channels = {
        "AWGN (n_taps=0)": {"n_taps": 0, "doppler_hz": 0.0},
        "Rayleigh Flat (n_taps=1)": {"n_taps": 1, "doppler_hz": 0.0},
        "Rayleigh Freq-Selective (n_taps=8)": {"n_taps": 8, "doppler_hz": 0.0}
    }

    results = {}

    for name, params in channels.items():
        bers = []
        print(f"\nTesting: {name}")
        for snr in snrs:
            tx = generate_qpsk(n_syms, seed=42)  # Fixed seed for reproducibility
            rx = libdsp.apply_channel(
                tx.astype(np.complex64),
                snr,
                n_taps=params["n_taps"],
                doppler_hz=params["doppler_hz"]
            )
            ber = compute_ber(tx, rx)
            bers.append(ber)
            print(f"  SNR={snr:2d} dB | BER={ber:.1e}")

        results[name] = bers

    # Save to CSV
    with open("results/logs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db"] + list(channels.keys()))
        for i, snr in enumerate(snrs):
            row = [snr] + [results[name][i] for name in channels.keys()]
            writer.writerow(row)

    # Plot
    plt.figure(figsize=(10, 6))
    for name in channels.keys():
        plt.semilogy(snrs, results[name], 'o-', label=name, linewidth=2)

    # Theoretical AWGN curve
    theo_ber = theoretical_qpsk_ber(snrs)
    plt.semilogy(snrs, theo_ber, 'k--', label="QPSK Theory (AWGN)", linewidth=2)

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("BER", fontsize=12)
    plt.title("QPSK BER vs SNR in Different Channel Conditions", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig("results/ber_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nBER sweep complete.")
    print(f"  Results: results/logs.csv")
    print(f"  Plot:    results/ber_vs_snr.png")


if __name__ == "__main__":
    main()