import matplotlib
matplotlib.use('Agg')  # Prevent MSYS2 segfaults
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings


def plot_ber_vs_snr(snrs: list, bers: dict, filename: str = "ber_vs_snr.png"):
    """Plot BER vs SNR curves. bers maps label -> BER values (same length as snrs)."""
    if not bers:
        raise ValueError("bers dictionary is empty")

    snrs = np.asarray(snrs)
    plt.figure(figsize=(10, 6))

    plotted = False
    for label, ber_values in bers.items():
        ber_arr = np.asarray(ber_values)
        if ber_arr.size != snrs.size:
            warnings.warn(
                f"BER array for '{label}' length {ber_arr.size} != snrs length {snrs.size}; skipping"
            )
            continue
        plt.semilogy(snrs, ber_arr, 'o-', label=label, linewidth=2)
        plotted = True

    if not plotted:
        raise ValueError("No valid BER series to plot (check lengths of inputs)")

    snr_lin = 10 ** (snrs / 10.0)
    theo_ber = 0.5 * np.erfc(np.sqrt(snr_lin / 2.0))
    plt.semilogy(snrs, theo_ber, 'k--', label="QPSK Theory (AWGN)", linewidth=2)

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR - SigFlow Validation")
    plt.legend()
    plt.grid(True, which="both")

    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", filename), dpi=150, bbox_inches="tight")
    plt.close()


def plot_latency_breakdown(metrics: dict, filename: str = "latency_breakdown.png"):
    """Plot DSP vs ML processing time. metrics keys: dsp_time_ms, ml_time_ms."""
    dsp_time = float(metrics.get('dsp_time_ms', 0.0))
    ml_time = float(metrics.get('ml_time_ms', 0.0))

    labels = ['DSP Processing', 'ML Inference']
    times = [dsp_time, ml_time]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, times, color=['steelblue', 'orange'])
    plt.ylabel("Time (ms)")
    plt.title("End-to-End Latency Breakdown")

    # Label bars with values
    for bar, t in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{t:.2f} ms",
                 ha='center', va='bottom', fontsize=10)

    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", filename), dpi=150, bbox_inches="tight")
    plt.close()