#!/usr/bin/env python3
"""
Quantify BER degradation due to Rayleigh fading.
Compares AWGN vs flat vs frequency-selective fading at fixed SNR.
"""

import numpy as np
import sys
import os
import csv
# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "python_src")
import libdsp


def generate_qpsk(n_syms: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    bits_i = np.random.randint(0, 2, n_syms)
    bits_q = np.random.randint(0, 2, n_syms)
    return (2 * bits_i - 1) + 1j * (2 * bits_q - 1)


def hard_decision_qpsk(rx: np.ndarray) -> np.ndarray:
    real = np.where(rx.real >= 0, 1, -1)
    imag = np.where(rx.imag >= 0, 1, -1)
    return real + 1j * imag


def compute_ber(tx: np.ndarray, rx: np.ndarray) -> float:
    tx_sym = hard_decision_qpsk(tx)
    rx_sym = hard_decision_qpsk(rx)
    return np.mean(tx_sym != rx_sym)


def main():
    os.makedirs("results", exist_ok=True)

    # Fixed test parameters
    snr_db = 10.0
    n_syms = 200000  # High count for stable BER
    tx = generate_qpsk(n_syms, seed=42)

    # Test scenarios
    scenarios = [
        {"name": "AWGN", "n_taps": 0, "doppler_hz": 0.0},
        {"name": "Rayleigh Flat", "n_taps": 1, "doppler_hz": 0.0},
        {"name": "Rayleigh Freq-Selective", "n_taps": 8, "doppler_hz": 0.0},
        {"name": "Rayleigh + Doppler (50 Hz)", "n_taps": 1, "doppler_hz": 50.0}
    ]

    print(f"BER Comparison at SNR = {snr_db} dB (QPSK, {n_syms} symbols)")
    print("-" * 60)

    bers = []
    names = []

    for scenario in scenarios:
        rx = libdsp.apply_channel(
            tx.astype(np.complex64),
            snr_db,
            n_taps=scenario["n_taps"],
            doppler_hz=scenario["doppler_hz"]
        )
        ber = compute_ber(tx, rx)
        bers.append(ber)
        names.append(scenario["name"])
        print(f"{scenario['name']:<30} | BER = {ber:.1e}")

    # Save to CSV
    with open("results/fading_impact.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "ber"])
        for name, ber in zip(names, bers):
            writer.writerow([name, ber])

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(names))
    plt.bar(y_pos, bers, color=['steelblue', 'orange', 'green', 'red'], alpha=0.8)
    plt.xticks(y_pos, names, rotation=15, ha='right')
    plt.ylabel("BER")
    plt.title(f"BER Degradation Due to Fading (SNR = {snr_db} dB)")
    plt.yscale('log')
    plt.grid(True, axis='y', which='both')
    plt.tight_layout()
    plt.savefig("results/fading_impact.png", dpi=150)
    plt.close()

    print("\nFading impact analysis complete.")
    print(f"  Results: results/fading_impact.csv")
    print(f"  Plot:    results/fading_impact.png")


if __name__ == "__main__":
    main()