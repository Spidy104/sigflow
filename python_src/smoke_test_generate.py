"""Smoke test for dataset generation and feature extraction.

Run this script from the repository root. It will create a small dataset in
results/smoke_test and verify the files are present and loadable.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import argparse

from python_src.dataset import generate_and_save


def run(full: bool = True, yes: bool = False, examples_override: int | None = None, samples_override: int | None = None, seed_override: int | None = None) -> None:
    """Run the smoke/generation test.

    If `full` is True this will generate the full dataset requested earlier
    (SNR 0..28 dB, 1000 examples per SNR per modulation). This can take a while
    and produce large files. Set `full=False` to run the small quick smoke test.
    """
    if full:
        out_dir = os.path.join("results", "full_dataset")
        mod_list = ["bpsk", "qpsk", "8psk", "16qam"]
        snr_db_list = list(range(0, 29))
        examples_per_snr = examples_override if examples_override is not None else 1000
        samples_per_example = samples_override if samples_override is not None else 256
        seed = seed_override if seed_override is not None else 123
        return_complex = False
        print("Preparing to generate FULL dataset:")
        print(f" - mods: {mod_list}")
        print(f" - SNRs: {snr_db_list[0]}..{snr_db_list[-1]} ({len(snr_db_list)} points)")
        print(f" - examples per SNR per mod: {examples_per_snr}")
        print(f" - samples per example: {samples_per_example}")
        est_examples = len(mod_list) * examples_per_snr * len(snr_db_list)
        print(f"Estimated examples: {est_examples}")
        if not yes:
            ans = input("Proceed? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted by user.")
                return
    else:
        out_dir = os.path.join("results", "smoke_test")
        mod_list = ["qpsk", "bpsk"]
        snr_db_list = [0, 10]
        examples_per_snr = examples_override if examples_override is not None else 5
        samples_per_example = samples_override if samples_override is not None else 128
        seed = seed_override if seed_override is not None else 123
        return_complex = False

    import time
    print("Generating dataset (this may take some time)...")
    t0 = time.time()
    try:
        paths = generate_and_save(
            out_dir=out_dir,
            mod_list=mod_list,
            snr_db_list=snr_db_list,
            examples_per_snr=examples_per_snr,
            samples_per_example=samples_per_example,
            seed=seed,
            return_complex=return_complex,
        )
    except KeyboardInterrupt:
        print("Generation interrupted by user.")
        return
    dt = time.time() - t0
    print(f"Generation completed in {dt:.1f} seconds.")

    print("Saved files:")
    for k, p in paths.items():
        print(f" - {k}: {p}")
        if not os.path.exists(p):
            print(f"ERROR: expected file missing: {p}")
            sys.exit(2)

    # quick load and basic checks
    ds = np.load(paths["dataset"], allow_pickle=True)
    feats = np.load(paths["features"], allow_pickle=True)

    X = ds["X"]
    y = ds["y"]
    meta = ds["meta"].item() if hasattr(ds["meta"], "item") else ds["meta"]

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")
    print(f"Features shape: {feats['features'].shape}")
    print(f"Meta keys: {list(meta.keys())}")

    # basic consistency checks
    assert X.shape[0] == y.shape[0], "number of examples mismatch"
    assert feats["features"].shape[0] == X.shape[0], "feature count mismatch"

    print("Dataset generation and validation complete.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Generate synthetic RF dataset and extract features.")
    parser.add_argument("--full", action="store_true", help="Generate the full dataset (SNR 0..28, 1000 examples per SNR per modulation)")
    parser.add_argument("--yes", action="store_true", help="Don't prompt for confirmation when generating full dataset")
    parser.add_argument("--examples-per-snr", type=int, default=None, help="Override examples per SNR per modulation")
    parser.add_argument("--samples", type=int, default=None, help="Override samples per example")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    args = parser.parse_args()
    run(full=args.full, yes=args.yes, examples_override=args.examples_per_snr, samples_override=args.samples, seed_override=args.seed)
