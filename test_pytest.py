"""SigFlow pytest test suite.

Run with:  uv run pytest test_pytest.py -v
"""
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "python_src")
import libdsp  # noqa: E402


# ---------------------------------------------------------------------------
# DSP / FFT tests
# ---------------------------------------------------------------------------

class TestFFT:
    """Tests for libdsp.dsp_process (FFT with 1/sqrt(N) normalisation)."""

    def test_empty_input(self):
        x = np.zeros(0, dtype=np.complex64)
        assert len(libdsp.dsp_process(x)) == 0

    @pytest.mark.parametrize("N", [1, 2, 3, 7, 8, 16, 31, 100, 1024, 4096])
    def test_output_length(self, N):
        x = np.ones(N, dtype=np.complex64)
        assert len(libdsp.dsp_process(x)) == N

    def test_impulse_flat_spectrum(self):
        """FFT of delta[0] should have constant magnitude 1/sqrt(N)."""
        N = 64
        x = np.zeros(N, dtype=np.complex64)
        x[0] = 1.0 + 0j
        X = libdsp.dsp_process(x)
        expected_mag = 1.0 / math.sqrt(N)
        np.testing.assert_allclose(np.abs(X), expected_mag, rtol=1e-4)

    def test_shifted_impulse_flat_spectrum(self):
        """FFT of delta[k] should also have constant magnitude (time-shift property)."""
        N = 32
        x = np.zeros(N, dtype=np.complex64)
        x[7] = 1.0 + 0j
        X = libdsp.dsp_process(x)
        np.testing.assert_allclose(np.abs(X), 1.0 / math.sqrt(N), rtol=1e-4)

    def test_energy_conservation(self):
        """Parseval: sum|x|^2 == sum|FFT(x)|^2."""
        rng = np.random.default_rng(42)
        x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(
            np.complex64
        )
        X = libdsp.dsp_process(x)
        np.testing.assert_allclose(
            np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2), rtol=1e-4
        )

    def test_linearity(self):
        """FFT(a*x + b*y) == a*FFT(x) + b*FFT(y)."""
        N = 32
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        y = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        a, b = 2.0 + 0j, 0.5 - 1j
        combo = (a * x + b * y).astype(np.complex64)
        np.testing.assert_allclose(
            libdsp.dsp_process(combo),
            a * libdsp.dsp_process(x) + b * libdsp.dsp_process(y),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_frequency_shift_property(self):
        """x[n] * e^(j2pi*k0*n/N) <-> X[(k - k0) mod N]."""
        N = 32
        k0 = 5
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        n = np.arange(N)
        x_shifted = (x * np.exp(2j * np.pi * k0 * n / N)).astype(np.complex64)

        X = libdsp.dsp_process(x)
        Y = libdsp.dsp_process(x_shifted)
        np.testing.assert_allclose(np.abs(Y), np.abs(np.roll(X, k0)), rtol=1e-3, atol=1e-5)

    def test_double_fft_time_reversal(self):
        """FFT(FFT(x))[n] == x[(N - n) % N]  (with 1/sqrt(N) normalisation)."""
        x = np.array([1 + 0.5j, 2 + 1j, 3 + 1.5j, 4 + 2j,
                      5 + 2.5j, 6 + 3j, 7 + 3.5j, 8 + 4j], dtype=np.complex64)
        N = len(x)
        XX = libdsp.dsp_process(libdsp.dsp_process(x))
        expected = np.roll(x[::-1], 1)  # x[(N - n) % N]
        np.testing.assert_allclose(XX, expected, rtol=1e-4, atol=1e-4)

    def test_real_input_conjugate_symmetry(self):
        """For real-valued input X[k] == conj(X[N-k])."""
        N = 64
        x = np.cos(2 * np.pi * np.arange(N) / 8).astype(np.complex64)
        X = libdsp.dsp_process(x)
        for k in range(1, N // 2):
            assert abs(X[k] - np.conj(X[N - k])) < 1e-4, f"Symmetry broken at bin {k}"

    def test_complex_exponential_bin_localization(self):
        """FFT of e^(j2pi*k0*n/N) concentrates all energy at bin k0."""
        N, k0 = 64, 7
        n = np.arange(N)
        x = np.exp(2j * np.pi * k0 * n / N).astype(np.complex64)
        X = libdsp.dsp_process(x)
        assert np.argmax(np.abs(X)) == k0
        assert np.abs(X[k0]) > 0.99  # magnitude should be 1/sqrt(N) * N^0.5 = 1

    def test_zero_input_zero_output(self):
        N = 64
        X = libdsp.dsp_process(np.zeros(N, dtype=np.complex64))
        assert np.max(np.abs(X)) < 1e-6

    def test_invalid_mode_raises(self):
        with pytest.raises(Exception):
            libdsp.dsp_process(np.ones(16, dtype=np.complex64), mode="invalid")

    @pytest.mark.parametrize("N", [1024, 2048, 4096, 8192])
    def test_large_fft_energy_conservation(self, N):
        rng = np.random.default_rng(N)
        x = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        X = libdsp.dsp_process(x)
        np.testing.assert_allclose(
            np.sum(np.abs(x) ** 2), np.sum(np.abs(X) ** 2), rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Channel tests
# ---------------------------------------------------------------------------

class TestChannel:
    """Tests for libdsp.apply_channel."""

    def test_empty_input(self):
        x = np.zeros(0, dtype=np.complex64)
        assert len(libdsp.apply_channel(x, 10.0)) == 0

    @pytest.mark.parametrize("N", [1, 10, 100, 1000])
    def test_output_length(self, N):
        x = np.ones(N, dtype=np.complex64)
        assert len(libdsp.apply_channel(x, 20.0, 1, 0.0)) == N

    def test_awgn_only_noise_power(self):
        """n_taps=0, large N: measured noise variance == signal_power * 10^(-SNR/10)."""
        N = 200_000
        snr_db = 10.0
        x = np.ones(N, dtype=np.complex64)
        y = libdsp.apply_channel(x, snr_db, 0, 0.0)

        noise_power = float(np.mean(np.abs(y - x) ** 2))
        signal_power = float(np.mean(np.abs(x) ** 2))
        theoretical = signal_power * 10 ** (-snr_db / 10)

        rel_error = abs(noise_power - theoretical) / theoretical
        assert rel_error < 0.03, (
            f"AWGN power mismatch: measured {noise_power:.6f}, "
            f"theoretical {theoretical:.6f} (rel error {rel_error:.3%})"
        )

    def test_very_high_snr_awgn_preserves_power(self):
        """At 60 dB AWGN-only, output power must equal input power within 1%."""
        N = 10_000
        x = np.ones(N, dtype=np.complex64)
        y = libdsp.apply_channel(x, 60.0, 0, 0.0)
        np.testing.assert_allclose(
            np.mean(np.abs(y) ** 2), np.mean(np.abs(x) ** 2), rtol=0.01
        )

    def test_flat_rayleigh_average_power(self):
        """n_taps=1, 50 000 symbols: E[|h|^2] -> 1 by LLN (unit-power Rayleigh)."""
        N = 50_000
        x = np.ones(N, dtype=np.complex64)
        y = libdsp.apply_channel(x, 100.0, 1, 0.0)  # near-zero noise
        output_power = float(np.mean(np.abs(y) ** 2))
        assert abs(output_power - 1.0) < 0.10, (
            f"Rayleigh mean power should be ~1.0, got {output_power:.4f}"
        )

    def test_higher_snr_less_awgn_noise(self):
        """Higher SNR => lower noise power in AWGN-only mode."""
        N = 5_000
        x = np.ones(N, dtype=np.complex64)
        noise_low_snr  = float(np.mean(np.abs(libdsp.apply_channel(x,  0.0, 0, 0.0) - x) ** 2))
        noise_high_snr = float(np.mean(np.abs(libdsp.apply_channel(x, 30.0, 0, 0.0) - x) ** 2))
        assert noise_high_snr < noise_low_snr

    def test_doppler_causes_phase_rotation(self):
        """Doppler phase at sample n = 2*pi*f_d*n/fs (fs = 1 MHz internal).
        At sample 999 with 100 Hz Doppler: phase ≈ 0.628 rad (well below 2*pi)."""
        N = 1_000
        x = np.ones(N, dtype=np.complex64)
        y_nodop = libdsp.apply_channel(x, 100.0, 0, 0.0)
        y_dop   = libdsp.apply_channel(x, 100.0, 0, 100.0)
        # Expected Doppler phase at sample 999: 2*pi * 100 * 999 / 1e6 ≈ 0.628 rad
        phase_diff = abs(float(np.angle(y_dop[N - 1])) - float(np.angle(y_nodop[N - 1])))
        assert phase_diff > 0.1, f"Expected phase diff > 0.1 rad, got {phase_diff:.4f}"

    def test_reproducibility(self):
        """Fixed internal seed: identical calls return identical results."""
        x = np.ones(100, dtype=np.complex64)
        np.testing.assert_array_equal(
            libdsp.apply_channel(x, 15.0, 4, 50.0),
            libdsp.apply_channel(x, 15.0, 4, 50.0),
        )

    @pytest.mark.parametrize("snr", [-20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0])
    def test_snr_range_no_crash(self, snr):
        x = np.ones(100, dtype=np.complex64)
        y = libdsp.apply_channel(x, snr, 1, 0.0)
        assert len(y) == 100
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("n_taps", [0, 1, 2, 4, 8, 16, 32])
    def test_n_taps_variants_no_crash(self, n_taps):
        x = np.ones(500, dtype=np.complex64)
        y = libdsp.apply_channel(x, 20.0, n_taps, 0.0)
        assert len(y) == 500
        assert np.all(np.isfinite(y))

    def test_negative_snr_does_not_crash(self):
        x = np.ones(100, dtype=np.complex64)
        y = libdsp.apply_channel(x, -5.0)
        assert len(y) == 100
        assert np.all(np.isfinite(y))

    def test_output_not_all_zeros(self):
        x = np.ones(100, dtype=np.complex64)
        y = libdsp.apply_channel(x, 10.0, 1, 0.0)
        assert np.any(np.abs(y) > 1e-3)

    def test_zero_input_contains_awgn(self):
        """Zero input with AWGN should produce non-zero output (pure noise)."""
        x = np.zeros(1000, dtype=np.complex64)
        y = libdsp.apply_channel(x, 0.0, 0, 0.0)
        assert float(np.mean(np.abs(y) ** 2)) > 0.0


# ---------------------------------------------------------------------------
# Python Utils / Helpers tests
# ---------------------------------------------------------------------------

class TestPythonUtils:
    """Tests for the helper functions in python_src/utils.py."""

    def test_generate_qpsk(self):
        import python_src.utils as utils
        n = 100
        syms = utils.generate_qpsk(n, seed=42)
        assert len(syms) == n
        assert syms.dtype == np.complex128 or syms.dtype == np.complex64
        # QPSK symbols are from {1+1j, 1-1j, -1+1j, -1-1j}
        for s in syms:
            assert abs(s.real) == 1.0
            assert abs(s.imag) == 1.0

        # Reproducibility
        syms2 = utils.generate_qpsk(n, seed=42)
        np.testing.assert_array_equal(syms, syms2)

    def test_hard_decision_qpsk(self):
        import python_src.utils as utils
        # Test constellation mapping
        inputs = np.array([0.5 + 0.5j, -0.1 + 2.0j, -1.5 - 0.2j, 3.0 - 0.1j])
        expected = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        np.testing.assert_array_equal(utils.hard_decision_qpsk(inputs), expected)

    def test_compute_ber(self):
        import python_src.utils as utils
        tx = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        # Identical
        assert utils.compute_ber(tx, tx) == 0.0
        # Fully inverted (all symbols different)
        rx_inverted = -tx
        assert utils.compute_ber(tx, rx_inverted) == 1.0
        # Half inverted
        rx_half = np.array([-1-1j, 1-1j, -1+1j, 1+1j])
        # Only 2 symbols differ
        assert utils.compute_ber(tx, rx_half) == 0.5

    def test_insert_pilots_and_extract(self):
        import python_src.utils as utils
        from python_src.config import PILOT_INTERVAL, PILOT_VALUE
        data = np.ones(100, dtype=np.complex64) * (2.0 + 2j)
        tx_signal, pilot_indices = utils.insert_pilots(data)
        
        # Check first pilot is at index 0
        assert pilot_indices[0] == 0
        assert tx_signal[0] == PILOT_VALUE
        
        # Check other pilot spacing
        for idx in pilot_indices:
            assert tx_signal[idx] == PILOT_VALUE
            assert idx % (PILOT_INTERVAL + 1) == 0

        # Extract symbols and verify we get original data back
        extracted, data_indices = utils.extract_data_symbols(tx_signal, pilot_indices)
        assert len(extracted) == len(data)
        np.testing.assert_array_equal(extracted, data)


# ---------------------------------------------------------------------------
# Channel Estimation & Equalization tests
# ---------------------------------------------------------------------------

class TestChannelEstimation:
    """Tests for channel estimation and equalization."""

    def test_pilot_aided_estimator(self):
        import python_src.utils as utils
        import python_src.channel_estimation as channel_estimation
        from python_src.config import PILOT_INTERVAL, PILOT_VALUE
        
        data = np.ones(300, dtype=np.complex64)
        tx_signal, pilot_indices = utils.insert_pilots(data)
        
        # Apply a simple constant channel gain of 2.0 + 1.0j and no noise
        h_true = 2.0 + 1.0j
        rx_signal = tx_signal * h_true
        
        estimator = channel_estimation.PilotAidedEstimator(pilot_interval=PILOT_INTERVAL)
        h_est = estimator.estimate(rx_signal, pilot_indices)
        
        # In noise-free case, we should estimate h exactly
        np.testing.assert_allclose(h_est, h_true, rtol=1e-5)

    def test_equalize_signal(self):
        import python_src.channel_estimation as channel_estimation
        # Simple signal
        rx = np.array([2.0 + 1.0j, -2.0 - 1.0j])
        h_est = np.array([2.0 + 1.0j, 2.0 + 1.0j])
        
        # Noise var = 0 (perfect equalization in MMSE should act like zero-forcing)
        equalized = channel_estimation.equalize_signal(rx, h_est, noise_var=0.0)
        np.testing.assert_allclose(equalized, np.array([1.0, -1.0]), rtol=1e-5)


# ---------------------------------------------------------------------------
# Dataset & Feature Extraction tests
# ---------------------------------------------------------------------------

class TestDatasetAndFeatures:
    """Tests for dataset generation and feature extraction."""

    def test_constellation_map(self):
        import python_src.dataset as dataset
        for mod in ["bpsk", "qpsk", "8psk", "16qam"]:
            pts = dataset._constellation_map(mod)
            # Verify unit energy normalization: mean(|pts|^2) == 1.0
            np.testing.assert_allclose(np.mean(np.abs(pts)**2), 1.0, rtol=1e-5)

        with pytest.raises(ValueError):
            dataset._constellation_map("invalid_mod")

    def test_generate_dataset(self):
        import python_src.dataset as dataset
        mods = ["bpsk", "qpsk"]
        snrs = [10.0, 20.0]
        examples_per_snr = 5
        samples_per_example = 64
        
        # Test complex representation
        X, y, meta = dataset.generate_dataset(
            mod_list=mods,
            snr_db_list=snrs,
            examples_per_snr=examples_per_snr,
            samples_per_example=samples_per_example,
            seed=42,
            return_complex=True
        )
        
        # 2 modulations * 2 SNRs * 5 examples = 20 total examples
        assert X.shape == (20, 64)
        assert X.dtype == np.complex64
        assert len(y) == 20
        assert meta["mod_names"] == mods
        assert len(meta["snrs"]) == 20

        # Test real-imag stacked representation
        X_stack, y_stack, meta_stack = dataset.generate_dataset(
            mod_list=mods,
            snr_db_list=snrs,
            examples_per_snr=examples_per_snr,
            samples_per_example=samples_per_example,
            seed=42,
            return_complex=False
        )
        assert X_stack.shape == (20, 64, 2)
        assert X_stack.dtype in [np.float32, np.float64]

    def test_extract_features(self):
        import python_src.dataset as dataset
        # 3 examples of length 128
        X_complex = (np.random.randn(3, 128) + 1j * np.random.randn(3, 128)).astype(np.complex64)
        feats1 = dataset.extract_features(X_complex, n_hist_bins=16, n_psd_bins=16)
        
        # Moments: 4 (real) + 4 (imag) + 4 (mag) = 12
        # Inst freq: mean + std = 2
        # Histograms: 16 (amp) + 16 (phase) = 32
        # PSD bins: 16
        # Total = 12 + 2 + 32 + 16 = 62 features per window
        assert feats1.shape == (3, 62)

        # Test with stacked representation (3, 128, 2)
        X_stack = np.stack([X_complex.real, X_complex.imag], axis=-1)
        feats2 = dataset.extract_features(X_stack, n_hist_bins=16, n_psd_bins=16)
        assert feats2.shape == (3, 62)
        np.testing.assert_allclose(feats1, feats2, rtol=1e-5, atol=1e-5)

    def test_save_load_and_generate_save(self):
        import python_src.dataset as dataset
        import tempfile
        
        mods = ["bpsk"]
        snrs = [15.0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            res = dataset.generate_and_save(
                out_dir=tmpdir,
                mod_list=mods,
                snr_db_list=snrs,
                examples_per_snr=2,
                samples_per_example=32,
                seed=42,
                return_complex=True
            )
            
            import os
            assert os.path.exists(res["dataset"])
            assert os.path.exists(res["features"])
            
            # Load and verify (using context manager to properly release the file handle under Windows)
            with np.load(res["dataset"], allow_pickle=True) as ds_data:
                assert "X" in ds_data
                assert "y" in ds_data
                assert ds_data["X"].shape == (2, 32)

