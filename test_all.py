"""SigFlow DSP Library - manual validation tests."""

import numpy as np
import sys
sys.path.insert(0, "python_src")
import libdsp

print("SIGFLOW DSP LIBRARY - TEST SUITE")
print()

print("TEST 1: FFT Energy Conservation")
n = 1024
x = np.random.randn(n).astype(np.complex64)
energy_time = np.sum(np.abs(x) ** 2)

X = libdsp.dsp_process(x, "fft")
energy_freq = np.sum(np.abs(X) ** 2)

print(f"Time-domain energy: {energy_time:.6f}")
print(f"Freq-domain energy: {energy_freq:.6f}")
print(f"Relative error:     {abs(energy_time - energy_freq) / energy_time * 100:.4f}%")

if np.isclose(energy_time, energy_freq, rtol=1e-4):
    print("PASS")
else:
    print("FAIL: FFT energy mismatch")
print()

print("TEST 2: AWGN Noise Power Validation")
x = np.ones(10000, dtype=np.complex64)  # Unit power signal
snr_db = 10.0
y = libdsp.apply_channel(x, snr_db, n_taps=0, doppler_hz=0.0)

signal_power = np.mean(np.abs(x) ** 2)
received_power = np.mean(np.abs(y) ** 2)
noise_power = received_power - signal_power
theoretical_noise = signal_power / (10 ** (snr_db / 10))

print(f"Signal power:       {signal_power:.6f}")
print(f"Measured noise:     {noise_power:.6f}")
print(f"Theoretical noise:  {theoretical_noise:.6f}")
print(f"Relative error:     {abs(noise_power - theoretical_noise) / theoretical_noise * 100:.2f}%")

if np.isclose(noise_power, theoretical_noise, rtol=0.15):
    print("PASS")
else:
    print("FAIL: AWGN noise power mismatch")
print()

print("TEST 3: Rayleigh Fading Power Conservation")
x = np.ones(10000, dtype=np.complex64)
y = libdsp.apply_channel(x, snr_db=100.0, n_taps=1, doppler_hz=0.0)

input_power = np.mean(np.abs(x) ** 2)
output_power = np.mean(np.abs(y) ** 2)
power_ratio = output_power / input_power

print(f"Input power:        {input_power:.6f}")
print(f"Output power:       {output_power:.6f}")
print(f"Power ratio:        {power_ratio:.6f}")

if 0.95 < power_ratio < 1.05:
    print("PASS")
else:
    print("FAIL: Rayleigh fading power not preserved")
print()

print("TEST 4: Rayleigh Fading Magnitude Distribution")
h_est = y / x  # Channel coefficients
magnitudes = np.abs(h_est)

measured_mean = np.mean(magnitudes)
theoretical_mean = np.sqrt(np.pi / 4)  # For σ = 1/sqrt(2)

print(f"Measured mean |h|:  {measured_mean:.4f}")
print(f"Theoretical mean:   {theoretical_mean:.4f}")
print(f"Relative error:     {abs(measured_mean - theoretical_mean) / theoretical_mean * 100:.2f}%")

if np.isclose(measured_mean, theoretical_mean, rtol=0.10):
    print("PASS")
else:
    print("FAIL: Rayleigh magnitude distribution mismatch")
print()

print("TEST 5: Block Fading Coherence (100 symbols per block)")
x = np.ones(1000, dtype=np.complex64)
y = libdsp.apply_channel(x, snr_db=100.0, n_taps=1, doppler_hz=0.0)
h_est = y / x

# Check first block (samples 0-99 should be identical)
h_block1 = h_est[0:100]
block1_variation = np.std(np.abs(h_block1))

# Check second block (samples 100-199 should be identical)
h_block2 = h_est[100:200]
block2_variation = np.std(np.abs(h_block2))

# Blocks should be different
block_difference = np.abs(h_block1[0] - h_block2[0])

print(f"Block 1 variation:  {block1_variation:.6f} (should be ~0)")
print(f"Block 2 variation:  {block2_variation:.6f} (should be ~0)")
print(f"Inter-block diff:   {block_difference:.4f} (should be >0)")

if block1_variation < 0.01 and block2_variation < 0.01 and block_difference > 0.1:
    print("PASS")
else:
    print("FAIL: Block fading coherence issue")
print()

print("TEST 6: Doppler Shift Phase Rotation")
x = np.ones(1000, dtype=np.complex64)
y_no_doppler = libdsp.apply_channel(x, snr_db=100.0, n_taps=0, doppler_hz=0.0)
y_with_doppler = libdsp.apply_channel(x, snr_db=100.0, n_taps=0, doppler_hz=100.0)

phase_no_doppler = np.abs(np.angle(y_no_doppler[-1]))
phase_with_doppler = np.abs(np.angle(y_with_doppler[-1]))

print(f"Final phase (no Doppler):   {phase_no_doppler:.4f} rad")
print(f"Final phase (100 Hz):       {phase_with_doppler:.4f} rad")

if phase_no_doppler < 0.5 and phase_with_doppler > 0.1:
    print("PASS")
else:
    print("FAIL: Doppler shift not working")
print()
print("All tests completed.")

