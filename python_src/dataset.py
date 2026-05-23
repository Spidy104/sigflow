"""Dataset generation and feature extraction utilities."""
from __future__ import annotations

import os
from typing import Tuple, List, Dict, Any

import numpy as np


def _constellation_map(mod: str) -> np.ndarray:
    mod = mod.lower()
    if mod == "bpsk":
        pts = np.array([1 + 0j, -1 + 0j])
    elif mod == "qpsk":
        pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    elif mod == "8psk":
        angles = 2 * np.pi * np.arange(8) / 8
        pts = np.exp(1j * angles)
    elif mod == "16qam":
        re = np.array([-3, -1, 1, 3])
        im = re.copy()
        grid = np.array([x + 1j * y for x in re for y in im])
        pts = grid / np.sqrt(np.mean(np.abs(grid) ** 2))
    else:
        raise ValueError(f"unsupported modulation: {mod}")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def generate_dataset(
    mod_list: List[str],
    snr_db_list: List[float],
    examples_per_snr: int = 100,
    samples_per_example: int = 256,
    seed: int | None = None,
    freq_offset_max: float = 0.0,
    random_phase: bool = True,
    apply_flat_fading: bool = False,
    fading_std: float = 0.0,
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a synthetic labeled dataset of modulated IQ windows.

    Returns (X, y, meta). X is either complex (N, L) if ``return_complex`` is True,
    otherwise real-imag stacked (N, L, 2).
    """
    rng = np.random.default_rng(seed)

    mod_list = list(mod_list)
    label_map = {m: i for i, m in enumerate(mod_list)}
    total_per_snr = examples_per_snr * len(mod_list)
    total_examples = total_per_snr * len(snr_db_list)

    Xc = np.empty((total_examples, samples_per_example), dtype=np.complex64)
    y = np.empty((total_examples,), dtype=np.int32)
    snrs = np.empty((total_examples,), dtype=np.float32)

    idx = 0
    for snr_db in snr_db_list:
        noise_var = 10 ** (-snr_db / 10.0)  # assume unit symbol energy
        for mod in mod_list:
            pts = _constellation_map(mod)
            for _ in range(examples_per_snr):
                sym_idx = rng.integers(0, len(pts), size=samples_per_example)
                symbols = pts[sym_idx].astype(np.complex64)

                if freq_offset_max > 0:
                    fo = rng.uniform(-freq_offset_max, freq_offset_max)
                    phase_ramp = 2j * np.pi * fo * np.arange(samples_per_example)
                    symbols = symbols * np.exp(phase_ramp)

                if random_phase:
                    phi = rng.uniform(0, 2 * np.pi)
                    symbols = symbols * np.exp(1j * phi)

                if apply_flat_fading:
                    if fading_std == 0:
                        h = 1.0 + 0j
                    else:
                        h = 1.0 + fading_std * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
                    symbols = symbols * h

                noise = np.sqrt(noise_var / 2.0) * (
                    rng.normal(size=samples_per_example) + 1j * rng.normal(size=samples_per_example)
                )
                rx = symbols + noise

                Xc[idx, :] = rx
                y[idx] = label_map[mod]
                snrs[idx] = float(snr_db)
                idx += 1

    perm = rng.permutation(total_examples)
    Xc = Xc[perm]
    y = y[perm]
    snrs = snrs[perm]

    if return_complex:
        X = Xc
    else:
        X = np.stack((Xc.real, Xc.imag), axis=-1)

    meta = {"mod_names": mod_list, "label_map": label_map, "snrs": snrs}
    return X, y, meta


def _moment_stats(x: np.ndarray) -> List[float]:
    mu = x.mean()
    x0 = x - mu
    sigma = x0.std(ddof=0)
    if sigma == 0:
        return [float(mu), 0.0, 0.0, 0.0]
    skew = (np.mean(x0 ** 3)) / (sigma ** 3)
    kurt = (np.mean(x0 ** 4)) / (sigma ** 4) - 3.0
    return [float(mu), float(sigma), float(skew), float(kurt)]


def _psd_bins(x: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Compute a simple PSD summary for a complex baseband window.

    Uses full FFT (since input is complex) then folds to one-sided magnitude
    by taking the first half of bins. Returns log10 power aggregated into
    `n_bins` buckets.
    """
    x = np.asarray(x, dtype=np.complex64)
    N = x.size
    if N == 0:
        return np.zeros(n_bins, dtype=np.float32)
    window = np.hanning(N).astype(np.float32)
    Xf = np.fft.fft(x * window)
    p = np.abs(Xf) ** 2
    # one-sided approximation (for complex baseband we could keep both; choose half for compactness)
    p = p[: N // 2]
    if p.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    # aggregate into n_bins by averaging contiguous segments
    seg_len = max(1, p.size // n_bins)
    pb = np.array([p[i : i + seg_len].mean() for i in range(0, p.size, seg_len)])
    if pb.size > n_bins:
        pb = pb[:n_bins]
    elif pb.size < n_bins:
        pb = np.pad(pb, (0, n_bins - pb.size), mode="constant", constant_values=pb.min())
    return np.log10(pb + 1e-12).astype(np.float32)


def extract_features(
    X: np.ndarray,
    n_hist_bins: int = 32,
    n_psd_bins: int = 32,
) -> np.ndarray:
    """Extract features from IQ windows.

    X may be complex (N, L) or real-imag stacked (N, L, 2). Returns feature matrix (N, F).
    """
    if X.ndim == 3 and X.shape[-1] == 2:
        # real-imag stacked
        Xc = X[..., 0] + 1j * X[..., 1]
    elif X.ndim == 2 and np.iscomplexobj(X):
        # complex matrix (N, L)
        Xc = X
    else:
        # strict: only accept (N, L, 2) or complex (N, L)
        raise ValueError(
            f"Unsupported X shape for feature extraction: shape={getattr(X, 'shape', None)}, dtype={getattr(X, 'dtype', None)}. "
            "Expected complex array (N, L) or real-imag stacked (N, L, 2)."
        )

    N, L = Xc.shape
    feats = []
    for i in range(N):
        x = Xc[i]
        re = x.real
        im = x.imag
        mag = np.abs(x)
        phase = np.angle(x)

        # moments: real, imag, mag
        stats = []
        stats += _moment_stats(re)
        stats += _moment_stats(im)
        stats += _moment_stats(mag)

        # instantaneous frequency
        up = np.unwrap(phase)
        inst_freq = np.diff(up)
        if inst_freq.size > 0:
            stats.append(float(inst_freq.mean()))
            stats.append(float(inst_freq.std()))
        else:
            stats += [0.0, 0.0]

        # histograms: amplitude and phase
        amp_hist, _ = np.histogram(mag, bins=n_hist_bins, density=True)
        phase_hist, _ = np.histogram(phase, bins=n_hist_bins, range=(-np.pi, np.pi), density=True)

        # PSD bins
        psd = _psd_bins(x, n_psd_bins)

        vec = np.concatenate([np.asarray(stats), amp_hist.astype(float), phase_hist.astype(float), psd.astype(float)])
        feats.append(vec)

    return np.vstack(feats)


def save_dataset(path: str, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, y=y, meta=meta)


def save_features(path: str, features: np.ndarray, y: np.ndarray, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, features=features, y=y, meta=meta)


def generate_and_save(
    out_dir: str,
    mod_list: List[str],
    snr_db_list: List[float],
    examples_per_snr: int = 100,
    samples_per_example: int = 256,
    **gen_kwargs,
) -> Dict[str, str]:
    """Generate dataset, extract features, and save both to `out_dir`.

    Returns dict with saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    X, y, meta = generate_dataset(mod_list, snr_db_list, examples_per_snr=examples_per_snr, samples_per_example=samples_per_example, **gen_kwargs)
    ds_path = os.path.join(out_dir, "dataset.npz")
    save_dataset(ds_path, X, y, meta)

    feats = extract_features(X)
    feat_path = os.path.join(out_dir, "features.npz")
    save_features(feat_path, feats, y, meta)

    return {"dataset": ds_path, "features": feat_path}
