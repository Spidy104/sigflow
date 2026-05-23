#include "../include/channel.h"
#include <random>
#include <cmath>
#include <numbers>
#include <execution>
#include <numeric>
#include <algorithm>

namespace sigflow::channel {

template<ComplexFloat T>
[[nodiscard]] auto apply(std::span<const T> input,
                        const float snr_db,
                        int n_taps,
                        const float doppler_hz)
    -> std::vector<T>
{
    const std::size_t n = input.size();
    if (n == 0) return {};

    // Create RNG with fixed seed for reproducibility in tests
    // Note: For realistic simulations, multiple independent transmissions should
    // call this function multiple times with different input sequences
    std::mt19937_64 rng{12345};

    // Copy input to output (we'll mutate it)
    std::vector<T> out(input.begin(), input.end());

    // === 1. Multi-tap Rayleigh fading ===
    // n_taps=0: AWGN only (no fading)
    // n_taps=1: Rayleigh flat fading (single time-varying tap, no ISI)
    // n_taps>1: Rayleigh frequency-selective fading (multiple taps with ISI)

    if (n_taps == 1) {
        // Flat Rayleigh fading: block fading model with coherence
        // Multiple symbols share the same fading coefficient (coherence time)
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(2.0f)); // Unit power complex Gaussian

        constexpr size_t coherence_symbols = 100; // Number of symbols per fading block

        for (std::size_t block_start = 0; block_start < n; block_start += coherence_symbols) {
            // Generate one fading coefficient for this coherence block
            T fading_coeff(dist(rng), dist(rng));

            // Apply same fading to all symbols in this block
            const size_t block_end = std::min(block_start + coherence_symbols, n);
            for (std::size_t i = block_start; i < block_end; ++i) {
                out[i] *= fading_coeff;
            }
        }
    } else if (n_taps > 1) {
        // Frequency-selective fading with multiple taps
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(2.0f)); // Unit power complex Gaussian

        // Calculate normalization factor to preserve average power across taps
        float total_power = 0.0f;
        for (int tap = 0; tap < n_taps; ++tap) {
            const float tap_gain = (tap == 0) ? 1.0f : 0.5f / static_cast<float>(tap + 1);
            total_power += tap_gain * tap_gain;
        }
        const float norm_factor = 1.0f / std::sqrt(total_power);

        std::vector<T> faded(n, T(0.0f, 0.0f));

        constexpr size_t coherence_symbols = 100; // Block fading coherence time

        // Generate tap coefficients per coherence block
        for (size_t block_start = 0; block_start < n; block_start += coherence_symbols) {
            // Generate tap coefficients for this coherence block
            std::vector<T> tap_coeffs(n_taps);
            for (int tap = 0; tap < n_taps; ++tap) {
                const float tap_gain = (tap == 0) ? 1.0f : 0.5f / static_cast<float>(tap + 1);
                tap_coeffs[tap] = norm_factor * tap_gain * T(dist(rng), dist(rng));
            }

            // Apply same tap coefficients to all symbols in this coherence block
            const size_t block_end = std::min(block_start + coherence_symbols, n);
            for (size_t i = block_start; i < block_end; ++i) {
                // Apply multipath: sum contributions from all delayed taps
                for (int tap = 0; tap < n_taps; ++tap) {
                    if (i >= static_cast<std::size_t>(tap)) {
                        const std::size_t input_idx = i - static_cast<std::size_t>(tap);
                        faded[i] += out[input_idx] * tap_coeffs[tap];
                    }
                }
            }
        }
        out = std::move(faded);
    }

    // === 2. Calculate noise parameters based on INPUT signal power ===
    // Use original input power for SNR calculation
    double input_power = 0.0;
    for (const auto& sample : input) {
        input_power += std::norm(sample);
    }
    input_power /= static_cast<double>(n);

    // === 3. AWGN injection ===
    const double snr_linear = std::pow(10.0, static_cast<double>(snr_db) / 10.0);

    // For zero input signals, use a reference power of 1.0 for noise calculation
    // This ensures noise is always added regardless of input signal level
    const double reference_power = (input_power > 0.0) ? input_power : 1.0;
    const double noise_power = reference_power / snr_linear;
    const double noise_variance = noise_power / 2.0; // per dimension
    const float noise_std = static_cast<float>(std::sqrt(noise_variance));

    std::normal_distribution noise_dist(0.0f, noise_std);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] += T(noise_dist(rng), noise_dist(rng));
    }

    // === 4. Doppler shift (time-varying phase) ===
    if (doppler_hz > 0.0f) {
        for (std::size_t i = 0; i < n; ++i) {
            constexpr float fs = 1e6f;
            const float phase = 2.0f * std::numbers::pi_v<float>
                                * doppler_hz * static_cast<float>(i) / fs;
            out[i] *= std::exp(T(0.0f, phase));
        }
    }

    return out;
}

// Explicit instantiation
template std::vector<std::complex<float>>
apply<std::complex<float>>(std::span<const std::complex<float>>, float, int, float);

} // namespace sigflow::channel
