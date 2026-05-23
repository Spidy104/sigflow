#include "../include/dsp.h"
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cmath>
#include <numbers>

using namespace sigflow::dsp;

class DSPTest : public ::testing::Test {
protected:
    static constexpr float TOLERANCE = 1e-5f;

    // Helper to check if two complex numbers are approximately equal
    static bool approxEqual(const std::complex<float>& a, const std::complex<float>& b, float tol = TOLERANCE) {
        return std::abs(a - b) < tol;
    }
};

// Test 1: Empty input handling
TEST_F(DSPTest, EmptyInputReturnsEmpty) {
    std::vector<std::complex<float>> empty_input;
    auto result = process<std::complex<float>>(empty_input, "fft");

    EXPECT_TRUE(result.empty());
}

// Test 2: Single element FFT
TEST_F(DSPTest, SingleElementFFT) {
    std::vector<std::complex<float>> input = {{1.0f, 0.0f}};
    auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), 1);
    EXPECT_TRUE(approxEqual(result[0], {1.0f, 0.0f}));
}

// Test 3: DC signal (all ones) - FFT should have energy only at DC bin
TEST_F(DSPTest, DCSignalFFT) {
    constexpr size_t N = 8;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // DC bin should have all the energy (normalized)
    float dc_magnitude = std::abs(result[0]);
    EXPECT_GT(dc_magnitude, 0.9f * std::sqrt(static_cast<float>(N))); // Most energy at DC

    // Other bins should be near zero
    for (size_t i = 1; i < N; ++i) {
        EXPECT_LT(std::abs(result[i]), 0.1f);
    }
}

// Test 4: Pure sine wave - energy should concentrate at specific frequency
TEST_F(DSPTest, SineWaveFFT) {
    constexpr size_t N = 64;
    std::vector<std::complex<float>> input(N);

    // Generate sine wave: sin(2*pi*freq*n/N)
    for (size_t n = 0; n < N; ++n) {
        constexpr float freq = 1.0f;
        const float angle = 2.0f * std::numbers::pi_v<float> * freq * static_cast<float>(n) / static_cast<float>(N);
        input[n] = {std::sin(angle), 0.0f};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // Energy should be at bin 1 and N-1 (positive and negative frequencies)
    const float bin1_magnitude = std::abs(result[1]);
    const float bin_last_magnitude = std::abs(result[N-1]);

    EXPECT_GT(bin1_magnitude, 3.0f); // Significant energy at frequency bin
    EXPECT_GT(bin_last_magnitude, 3.0f);
}

// Test 5: Complex exponential (single frequency)
TEST_F(DSPTest, ComplexExponentialFFT) {
    constexpr size_t N = 32;
    constexpr size_t k = 4; // Frequency bin
    std::vector<std::complex<float>> input(N);

    // Generate complex exponential: exp(j*2*pi*k*n/N)
    for (size_t n = 0; n < N; ++n) {
        const float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(k * n) / static_cast<float>(N);
        input[n] = {std::cos(angle), std::sin(angle)};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // All energy should be at bin k
    float target_bin_magnitude = std::abs(result[k]);
    EXPECT_GT(target_bin_magnitude, std::sqrt(static_cast<float>(N)) * 0.9f);

    // Other bins should be near zero
    for (size_t i = 0; i < N; ++i) {
        if (i != k) {
            EXPECT_LT(std::abs(result[i]), 0.5f);
        }
    }
}

// Test 6: Parseval's theorem - energy conservation
TEST_F(DSPTest, EnergyConservation) {
    constexpr size_t N = 128;
    std::vector<std::complex<float>> input(N);

    // Generate random-like signal
    for (size_t n = 0; n < N; ++n) {
        input[n] = {std::cos(static_cast<float>(n) * 0.1f),
                    std::sin(static_cast<float>(n) * 0.2f)};
    }

    // Calculate input energy
    float input_energy = 0.0f;
    for (const auto& sample : input) {
        input_energy += std::norm(sample);
    }

    auto result = process<std::complex<float>>(input, "fft");

    // Calculate output energy (already normalized by implementation)
    float output_energy = 0.0f;
    for (const auto& sample : result) {
        output_energy += std::norm(sample);
    }

    // Energies should be equal (within tolerance)
    EXPECT_NEAR(input_energy, output_energy, input_energy * 0.01f); // 1% tolerance
}

// Test 7: Power of 2 sizes
TEST_F(DSPTest, PowerOfTwoSizes) {
    for (size_t power = 0; power <= 10; ++power) {
        size_t N = 1 << power; // 2^power
        std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

        EXPECT_NO_THROW({
            auto result = process<std::complex<float>>(input, "fft");
            EXPECT_EQ(result.size(), N);
        });
    }
}

// Test 8: Non-power of 2 sizes (FFTW handles these too)
TEST_F(DSPTest, NonPowerOfTwoSizes) {
    std::vector<size_t> sizes = {3, 5, 7, 10, 15, 30, 100};

    for (size_t N : sizes) {
        std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

        EXPECT_NO_THROW({
            auto result = process<std::complex<float>>(input, "fft");
            EXPECT_EQ(result.size(), N);
        });
    }
}

// Test 9: Invalid mode should throw
TEST_F(DSPTest, InvalidModeThrows) {
    std::vector<std::complex<float>> input = {{1.0f, 0.0f}, {2.0f, 0.0f}};

    EXPECT_THROW({
        auto result = process<std::complex<float>>(input, "invalid_mode");
    }, std::invalid_argument);
}

// Test 10: Linearity test - FFT is linear
TEST_F(DSPTest, LinearityProperty) {
    constexpr size_t N = 16;
    std::vector<std::complex<float>> signal1(N);
    std::vector<std::complex<float>> signal2(N);
    std::vector<std::complex<float>> combined(N);

    constexpr float a = 2.0f;
    constexpr float b = 3.0f;

    // Generate two different signals
    for (size_t n = 0; n < N; ++n) {
        signal1[n] = {std::cos(static_cast<float>(n) * 0.5f), 0.0f};
        signal2[n] = {std::sin(static_cast<float>(n) * 0.3f), 0.0f};
        combined[n] = a * signal1[n] + b * signal2[n];
    }

    const auto fft1 = process<std::complex<float>>(signal1, "fft");
    const auto fft2 = process<std::complex<float>>(signal2, "fft");
    const auto fft_combined = process<std::complex<float>>(combined, "fft");

    // FFT(a*x1 + b*x2) should equal a*FFT(x1) + b*FFT(x2)
    for (size_t i = 0; i < N; ++i) {
        std::complex<float> expected = a * fft1[i] + b * fft2[i];
        EXPECT_TRUE(approxEqual(fft_combined[i], expected, 0.01f));
    }
}

// Test 11: Large input handling
TEST_F(DSPTest, LargeInputHandling) {
    constexpr size_t N = 4096;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    EXPECT_NO_THROW({
        const auto result = process<std::complex<float>>(input, "fft");
        EXPECT_EQ(result.size(), N);
    });
}

// Test 12: Symmetry for real inputs
TEST_F(DSPTest, RealInputSymmetry) {
    constexpr size_t N = 16;
    std::vector<std::complex<float>> input(N);

    // Real-valued input
    for (size_t n = 0; n < N; ++n) {
        input[n] = {std::cos(static_cast<float>(n) * 0.2f), 0.0f};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    // For real input, FFT should have conjugate symmetry: X[k] = conj(X[N-k])
    for (size_t k = 1; k < N/2; ++k) {
        std::complex<float> expected_conjugate = std::conj(result[N - k]);
        EXPECT_TRUE(approxEqual(result[k], expected_conjugate, 0.01f));
    }
}

// Test 13: Zero input handling
TEST_F(DSPTest, ZeroInputFFT) {
    const size_t N = 16;
    std::vector<std::complex<float>> input(N, {0.0f, 0.0f});
    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // All output should be zero (or very close due to normalization)
    for (const auto& sample : result) {
        EXPECT_LT(std::abs(sample), 0.001f);
    }
}

// Test 14: Impulse response (delta function)
TEST_F(DSPTest, ImpulseResponseFFT) {
    constexpr size_t N = 32;
    std::vector<std::complex<float>> input(N, {0.0f, 0.0f});
    input[0] = {1.0f, 0.0f}; // Impulse at time 0

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // FFT of impulse should be flat spectrum (all bins equal)
    const float first_magnitude = std::abs(result[0]);
    EXPECT_GT(first_magnitude, 0.1f);

    for (size_t i = 1; i < N; ++i) {
        EXPECT_NEAR(std::abs(result[i]), first_magnitude, 0.1f);
    }
}

// Test 15: Shifted impulse (time delay property)
TEST_F(DSPTest, ShiftedImpulseFFT) {
    constexpr size_t N = 16;
    constexpr size_t delay = 4;

    std::vector<std::complex<float>> input(N, {0.0f, 0.0f});
    input[delay] = {1.0f, 0.0f}; // Impulse at time 'delay'

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // All magnitudes should be equal (flat spectrum)
    const float reference_magnitude = std::abs(result[0]);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(std::abs(result[i]), reference_magnitude, reference_magnitude * 0.1f);
    }
}

// Test 16: Cosine wave at Nyquist frequency
TEST_F(DSPTest, NyquistFrequencyFFT) {
    constexpr size_t N = 32;
    std::vector<std::complex<float>> input(N);

    // Generate cosine at Nyquist frequency (alternating +1, -1)
    for (size_t n = 0; n < N; ++n) {
        input[n] = {(n % 2 == 0) ? 1.0f : -1.0f, 0.0f};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // Energy should be concentrated at Nyquist bin (N/2)
    float nyquist_magnitude = std::abs(result[N/2]);
    EXPECT_GT(nyquist_magnitude, 3.0f);

    // Other bins should be much smaller
    for (size_t i = 1; i < N/2; ++i) {
        EXPECT_LT(std::abs(result[i]), 1.0f);
    }
}

// Test 17: Multiple frequency components
TEST_F(DSPTest, MultipleFrequenciesFFT) {
    constexpr size_t N = 64;
    constexpr size_t freq1 = 5;
    constexpr size_t freq2 = 10;
    std::vector<std::complex<float>> input(N);

    // Generate sum of two complex exponentials
    for (size_t n = 0; n < N; ++n) {
        const float angle1 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(freq1 * n) / static_cast<float>(N);
        const float angle2 = 2.0f * std::numbers::pi_v<float> * static_cast<float>(freq2 * n) / static_cast<float>(N);

        input[n] = {std::cos(angle1) + std::cos(angle2),
                    std::sin(angle1) + std::sin(angle2)};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // Both frequency bins should have significant energy
    EXPECT_GT(std::abs(result[freq1]), 3.0f);
    EXPECT_GT(std::abs(result[freq2]), 3.0f);

    // Other bins should be smaller
    for (size_t i = 0; i < N; ++i) {
        if (i != freq1 && i != freq2) {
            EXPECT_LT(std::abs(result[i]), 1.0f);
        }
    }
}

// Test 18: Phase accuracy
TEST_F(DSPTest, PhaseAccuracyFFT) {
    constexpr size_t N = 32;
    constexpr size_t k = 8;
    constexpr float input_phase = std::numbers::pi_v<float> / 4.0f; // 45 degrees
    std::vector<std::complex<float>> input(N);

    // Generate complex exponential with specific phase offset
    for (size_t n = 0; n < N; ++n) {
        float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(k * n) / static_cast<float>(N) + input_phase;
        input[n] = {std::cos(angle), std::sin(angle)};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // Check phase at frequency bin k
    const float output_phase = std::arg(result[k]);

    // Phase should match input phase (within tolerance)
    EXPECT_NEAR(output_phase, input_phase, 0.1f);
}

// Test 19: Prime number sizes
TEST_F(DSPTest, PrimeNumberSizes) {
    std::vector<size_t> prime_sizes = {7, 11, 13, 17, 19, 23, 29, 31};

    for (size_t N : prime_sizes) {
        std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

        EXPECT_NO_THROW({
            auto result = process<std::complex<float>>(input, "fft");
            EXPECT_EQ(result.size(), N);

            // DC component should dominate
            EXPECT_GT(std::abs(result[0]), std::sqrt(static_cast<float>(N)) * 0.8f);
        });
    }
}

// Test 20: Very large FFT sizes
TEST_F(DSPTest, LargeFFTSizes) {
    std::vector<size_t> large_sizes = {1024, 2048, 4096, 8192};

    for (size_t N : large_sizes) {
        std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

        EXPECT_NO_THROW({
            auto result = process<std::complex<float>>(input, "fft");
            EXPECT_EQ(result.size(), N);

            // Energy conservation check
            float input_energy = static_cast<float>(N); // All ones
            float output_energy = 0.0f;
            for (const auto& sample : result) {
                output_energy += std::norm(sample);
            }

            EXPECT_NEAR(output_energy, input_energy, input_energy * 0.01f);
        });
    }
}

// Test 21: Floating point precision edge cases
TEST_F(DSPTest, FloatingPointPrecision) {
    constexpr size_t N = 16;

    // Very small values
    std::vector<std::complex<float>> small_input(N, {1e-10f, 1e-10f});
    EXPECT_NO_THROW({
        const auto result = process<std::complex<float>>(small_input, "fft");
        EXPECT_EQ(result.size(), N);
    });

    // Very large values
    std::vector<std::complex<float>> large_input(N, {1e6f, 1e6f});
    EXPECT_NO_THROW({
        auto result = process<std::complex<float>>(large_input, "fft");
        EXPECT_EQ(result.size(), N);

        // Check for NaN or infinity
        for (const auto& sample : result) {
            EXPECT_FALSE(std::isnan(sample.real()));
            EXPECT_FALSE(std::isnan(sample.imag()));
            EXPECT_FALSE(std::isinf(sample.real()));
            EXPECT_FALSE(std::isinf(sample.imag()));
        }
    });
}

// Test 22: Window function compatibility
TEST_F(DSPTest, WindowedSignalFFT) {
    constexpr size_t N = 64;
    std::vector<std::complex<float>> input(N);

    // Generate windowed sine wave (Hann window)
    constexpr float freq = 8.0f;
    for (size_t n = 0; n < N; ++n) {
        const float window = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> * static_cast<float>(n) / static_cast<float>(N-1)));
        const float signal = std::sin(2.0f * std::numbers::pi_v<float> * freq * static_cast<float>(n) / static_cast<float>(N));
        input[n] = {window * signal, 0.0f};
    }

    const auto result = process<std::complex<float>>(input, "fft");

    ASSERT_EQ(result.size(), N);

    // Main lobe should be around frequency bin
    const size_t main_bin = static_cast<size_t>(freq);
    float peak_magnitude = 0.0f;
    size_t peak_bin = 0;

    constexpr size_t start_bin = main_bin - 2;  // freq=8, so main_bin=8, always >= 2
    const size_t end_bin = std::min(main_bin + 2, N - 1);

    for (size_t i = start_bin; i <= end_bin; ++i) {
        if (std::abs(result[i]) > peak_magnitude) {
            peak_magnitude = std::abs(result[i]);
            peak_bin = i;
        }
    }

    EXPECT_GE(peak_bin, main_bin - 1);
    EXPECT_LE(peak_bin, main_bin + 1);
    EXPECT_GT(peak_magnitude, 1.5f); // Hann window reduces amplitude significantly
}

// Test 23: Frequency resolution
TEST_F(DSPTest, FrequencyResolution) {
    constexpr size_t N = 128;
    constexpr float freq1 = 10.0f;
    constexpr float freq2 = 11.0f; // Adjacent bins

    std::vector<std::complex<float>> input1(N), input2(N);

    // Generate two signals with close frequencies
    for (size_t n = 0; n < N; ++n) {
        const float angle1 = 2.0f * std::numbers::pi_v<float> * freq1 * static_cast<float>(n) / static_cast<float>(N);
        const float angle2 = 2.0f * std::numbers::pi_v<float> * freq2 * static_cast<float>(n) / static_cast<float>(N);

        input1[n] = {std::cos(angle1), std::sin(angle1)};
        input2[n] = {std::cos(angle2), std::sin(angle2)};
    }

    const auto result1 = process<std::complex<float>>(input1, "fft");
    const auto result2 = process<std::complex<float>>(input2, "fft");

    // Should be able to distinguish between adjacent bins
    EXPECT_GT(std::abs(result1[static_cast<size_t>(freq1)]), std::abs(result1[static_cast<size_t>(freq2)]));
    EXPECT_GT(std::abs(result2[static_cast<size_t>(freq2)]), std::abs(result2[static_cast<size_t>(freq1)]));
}

// Test 24: Frequency shift property - multiplying by e^(j2pi*k0*n/N) shifts FFT by k0 bins
TEST_F(DSPTest, FrequencyShiftProperty) {
    constexpr size_t N = 32;
    constexpr size_t k0 = 5;

    std::vector<std::complex<float>> x(N), x_modulated(N);
    for (size_t n = 0; n < N; ++n) {
        x[n] = {std::cos(static_cast<float>(n) * 0.4f), std::sin(static_cast<float>(n) * 0.3f)};
        const float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(k0 * n) / static_cast<float>(N);
        x_modulated[n] = x[n] * std::complex<float>{std::cos(angle), std::sin(angle)};
    }

    const auto X = process<std::complex<float>>(x, "fft");
    const auto Y = process<std::complex<float>>(x_modulated, "fft");

    // Y[k] should equal X[(k - k0 + N) % N] (circular frequency shift)
    for (size_t k = 0; k < N; ++k) {
        const size_t shifted_k = (k + N - k0) % N;
        EXPECT_NEAR(Y[k].real(), X[shifted_k].real(), 0.01f)
            << "Frequency shift real part failed at bin " << k;
        EXPECT_NEAR(Y[k].imag(), X[shifted_k].imag(), 0.01f)
            << "Frequency shift imag part failed at bin " << k;
    }
}

// Test 25: Double FFT equals time reversal - FFT(FFT(x))[n] == x[(N-n) % N]
// With 1/sqrt(N) normalisation: applying FFT twice produces a time-reversed copy of x.
TEST_F(DSPTest, DoubleFFTIsTimeReversal) {
    constexpr size_t N = 8;
    std::vector<std::complex<float>> x(N);
    for (size_t n = 0; n < N; ++n) {
        x[n] = {static_cast<float>(n + 1), static_cast<float>(N - n) * 0.5f};
    }

    const auto X  = process<std::complex<float>>(x, "fft");
    const auto XX = process<std::complex<float>>(X, "fft");

    // XX[0] == x[0]; XX[n] == x[N-n] for n >= 1
    EXPECT_NEAR(XX[0].real(), x[0].real(), 0.01f);
    EXPECT_NEAR(XX[0].imag(), x[0].imag(), 0.01f);
    for (size_t n = 1; n < N; ++n) {
        EXPECT_NEAR(XX[n].real(), x[N - n].real(), 0.01f)
            << "Double-FFT time reversal failed at n=" << n;
        EXPECT_NEAR(XX[n].imag(), x[N - n].imag(), 0.01f)
            << "Double-FFT time reversal failed at n=" << n;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
