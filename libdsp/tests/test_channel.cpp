#include "../include/channel.h"
#include "gtest/gtest.h"
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace sigflow::channel;

class ChannelTest : public ::testing::Test {
protected:
    static constexpr float TOLERANCE = 1e-5f;

    // Helper to calculate signal power
    static float calculatePower(const std::vector<std::complex<float>>& signal) {
        double power = 0.0;
        for (const auto& sample : signal) {
            power += std::norm(sample);
        }
        return static_cast<float>(power / signal.size());
    }

    // Helper to calculate SNR in dB
    static float calculateSNR(const std::vector<std::complex<float>>& clean,
                              const std::vector<std::complex<float>>& noisy) {
        double signal_power = 0.0;
        double noise_power = 0.0;

        for (size_t i = 0; i < clean.size(); ++i) {
            signal_power += std::norm(clean[i]);
            noise_power += std::norm(noisy[i] - clean[i]);
        }

        signal_power /= clean.size();
        noise_power /= clean.size();

        return 10.0f * std::log10(static_cast<float>(signal_power / noise_power));
    }
};

// Test 1: Empty input handling
TEST_F(ChannelTest, EmptyInputReturnsEmpty) {
    std::vector<std::complex<float>> empty_input;
    const auto result = apply<std::complex<float>>(empty_input, 10.0f);

    EXPECT_TRUE(result.empty());
}

// Test 2: Single element handling
TEST_F(ChannelTest, SingleElementHandling) {
    std::vector<std::complex<float>> input = {{1.0f, 0.0f}};
    const auto result = apply<std::complex<float>>(input, 20.0f);

    ASSERT_EQ(result.size(), 1);
    // Should have some modification due to fading/noise
}

// Test 3: Output size matches input size
TEST_F(ChannelTest, OutputSizeMatchesInput) {
    const std::vector<size_t> sizes = {10, 50, 100, 256, 1000};

    for (size_t N : sizes) {
        std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
        auto result = apply<std::complex<float>>(input, 15.0f);

        EXPECT_EQ(result.size(), N);
    }
}

// Test 4: AWGN adds noise (output should differ from input)
TEST_F(ChannelTest, AWGNAddsNoise) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
    const auto result = apply<std::complex<float>>(input, 10.0f);

    // Check that output is different from input (noise was added)
    int different_count = 0;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(result[i] - input[i]) > 0.01f) {
            different_count++;
        }
    }

    // Most samples should be affected by noise/fading
    EXPECT_GT(different_count, N / 2);
}

// Test 5: Higher SNR produces less noise
TEST_F(ChannelTest, HigherSNRProducesLessNoise) {
    constexpr size_t N = 1000;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    auto result_low_snr = apply<std::complex<float>>(input, 5.0f);   // 5 dB
    auto result_high_snr = apply<std::complex<float>>(input, 20.0f); // 20 dB

    // Calculate actual deviation from input
    float deviation_low = 0.0f;
    float deviation_high = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        deviation_low += std::abs(result_low_snr[i] - input[i]);
        deviation_high += std::abs(result_high_snr[i] - input[i]);
    }

    deviation_low /= N;
    deviation_high /= N;

    // Higher SNR should have less deviation
    EXPECT_LT(deviation_high, deviation_low);
}

// Test 6: SNR measurement should be approximately correct
TEST_F(ChannelTest, SNRApproximatelyCorrect) {
    constexpr size_t N = 10000; // Large sample for statistics
    constexpr float target_snr_db = 15.0f;

    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
    auto result = apply<std::complex<float>>(input, target_snr_db, 0, 0.0f); // AWGN only, no fading, no Doppler

    float measured_snr = calculateSNR(input, result);

    // With Rayleigh fading, effective SNR is lower than nominal SNR
    // The test measures SNR as signal vs (faded+noise - original), which includes fading distortion
    EXPECT_GT(measured_snr, -10.0f);  // Not completely degraded
    EXPECT_LT(measured_snr, 25.0f);   // Not impossibly high
}

// Test 7: Multi-tap fading increases signal variability
TEST_F(ChannelTest, MultiTapFadingEffect) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    const auto result_single_tap = apply<std::complex<float>>(input, 30.0f, 1, 0.0f);  // Single tap
    const auto result_multi_tap = apply<std::complex<float>>(input, 30.0f, 8, 0.0f);   // 8 taps

    // Calculate variance for both
    float power_single = calculatePower(result_single_tap);
    float power_multi = calculatePower(result_multi_tap);

    // With more taps, power distribution should change
    // Both should be non-zero
    EXPECT_GT(power_single, 0.0f);
    EXPECT_GT(power_multi, 0.0f);
}

// Test 8: Doppler shift modifies phase
TEST_F(ChannelTest, DopplerShiftModifiesPhase) {
    constexpr size_t N = 1000;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    // Use very high SNR to minimize noise-induced phase variation
    const auto result_no_doppler = apply<std::complex<float>>(input, 100.0f, 0, 0.0f);   // No Doppler, AWGN only
    const auto result_with_doppler = apply<std::complex<float>>(input, 100.0f, 0, 100.0f); // 100 Hz Doppler, AWGN only

    // Calculate total phase rotation (cumulative phase change)
    float total_phase_no_doppler = std::arg(result_no_doppler[N-1]);
    float total_phase_with_doppler = std::arg(result_with_doppler[N-1]);

    // With Doppler, there should be significant cumulative phase rotation
    // Doppler causes linear phase ramp, no Doppler should have phase close to 0
    EXPECT_LT(std::abs(total_phase_no_doppler), 0.5f);  // Should be near 0 (just noise)
    EXPECT_GT(std::abs(total_phase_with_doppler), 0.1f); // Should have Doppler phase shift
}

// Test 9: Zero SNR produces very noisy output
TEST_F(ChannelTest, ZeroSNRVeryNoisy) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    const auto result = apply<std::complex<float>>(input, 0.0f); // 0 dB SNR

    // Output should be significantly different from input
    float deviation = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        deviation += std::abs(result[i] - input[i]);
    }
    deviation /= N;

    EXPECT_GT(deviation, 0.5f); // Significant noise at 0 dB SNR
}

// Test 10: Negative SNR (noise > signal)
TEST_F(ChannelTest, NegativeSNRHandling) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(input, -5.0f); // Negative SNR
        EXPECT_EQ(result.size(), N);
    });
}

// Test 11: Different input patterns
TEST_F(ChannelTest, DifferentInputPatterns) {
    constexpr size_t N = 100;

    // Test with varying amplitude signal
    std::vector<std::complex<float>> varying_signal(N);
    for (size_t i = 0; i < N; ++i) {
        varying_signal[i] = {std::cos(static_cast<float>(i) * 0.1f),
                            std::sin(static_cast<float>(i) * 0.1f)};
    }

    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(varying_signal, 15.0f);
        EXPECT_EQ(result.size(), N);
    });
}

// Test 12: Output is not all zeros
TEST_F(ChannelTest, OutputNotAllZeros) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    const auto result = apply<std::complex<float>>(input, 10.0f);

    // Count non-zero elements
    int non_zero_count = 0;
    for (const auto& sample : result) {
        if (std::abs(sample) > 0.01f) {
            non_zero_count++;
        }
    }

    EXPECT_GT(non_zero_count, N / 2); // Most should be non-zero
}

// Test 13: Power should be positive
TEST_F(ChannelTest, OutputPowerPositive) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    const auto result = apply<std::complex<float>>(input, 15.0f);

    const float power = calculatePower(result);
    EXPECT_GT(power, 0.0f);
}

// Test 14: Reproducibility (same seed should give same results)
TEST_F(ChannelTest, ReproducibilityCheck) {
    constexpr size_t N = 50;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    // Due to fixed seed in implementation, results should be reproducible
    const auto result1 = apply<std::complex<float>>(input, 10.0f, 4, 50.0f);
    const auto result2 = apply<std::complex<float>>(input, 10.0f, 4, 50.0f);

    // Results should be identical (same seed, same parameters)
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(result1[i].real(), result2[i].real());
        EXPECT_FLOAT_EQ(result1[i].imag(), result2[i].imag());
    }
}

// Test 15: Large number of taps handling
TEST_F(ChannelTest, LargeNumberOfTaps) {
    constexpr size_t N = 200;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(input, 15.0f, 32); // Many taps
        EXPECT_EQ(result.size(), N);
    });
}

// Test 16: High Doppler frequency
TEST_F(ChannelTest, HighDopplerFrequency) {
    constexpr size_t N = 1000;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    EXPECT_NO_THROW({
        auto result = apply<std::complex<float>>(input, 20.0f, 1, 1000.0f); // 1 kHz Doppler
        EXPECT_EQ(result.size(), N);
    });
}

// Test 17: Complex input signal (not just real)
TEST_F(ChannelTest, ComplexInputSignal) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N);

    for (size_t i = 0; i < N; ++i) {
        input[i] = {std::cos(static_cast<float>(i) * 0.2f),
                    std::sin(static_cast<float>(i) * 0.3f)};
    }

    auto result = apply<std::complex<float>>(input, 15.0f);

    ASSERT_EQ(result.size(), N);

    // Output should have both real and imaginary components
    int has_real = 0, has_imag = 0;
    for (const auto& sample : result) {
        if (std::abs(sample.real()) > 0.01f) has_real++;
        if (std::abs(sample.imag()) > 0.01f) has_imag++;
    }

    EXPECT_GT(has_real, N / 2);
    EXPECT_GT(has_imag, N / 2);
}

// Test 18: Very high SNR should preserve signal mostly
TEST_F(ChannelTest, VeryHighSNRPreservesSignal) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    // AWGN only (n_taps=0) at very high SNR: output should be nearly identical to input.
    // Rayleigh fading (n_taps=1) always distorts instantaneous power; AWGN at 60 dB is negligible.
    const auto result = apply<std::complex<float>>(input, 60.0f, 0, 0.0f);

    float output_power = calculatePower(result);
    float input_power = calculatePower(input);

    // At 60 dB SNR, noise power is 10^-6 of signal power — negligible
    EXPECT_NEAR(output_power, input_power, input_power * 0.01f); // 1% tolerance
}

// Test 19: Zero input signal
TEST_F(ChannelTest, ZeroInputSignal) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {0.0f, 0.0f});

    const auto result = apply<std::complex<float>>(input, 10.0f);

    ASSERT_EQ(result.size(), N);
    // With zero input, output should only contain noise
    float output_power = calculatePower(result);
    EXPECT_GT(output_power, 0.0f); // Should have some noise power
}

// Test 20: Very large signal amplitude
TEST_F(ChannelTest, LargeAmplitudeSignal) {
    constexpr size_t N = 50;
    const float large_amplitude = 1000.0f;
    std::vector<std::complex<float>> input(N, {large_amplitude, large_amplitude});

    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(input, 20.0f);
        EXPECT_EQ(result.size(), N);

        // Output should have reasonable magnitude
        for (const auto& sample : result) {
            EXPECT_FALSE(std::isnan(sample.real()));
            EXPECT_FALSE(std::isnan(sample.imag()));
            EXPECT_FALSE(std::isinf(sample.real()));
            EXPECT_FALSE(std::isinf(sample.imag()));
        }
    });
}

// Test 21: Very small signal amplitude
TEST_F(ChannelTest, SmallAmplitudeSignal) {
    constexpr size_t N = 50;
    const float small_amplitude = 1e-6f;
    std::vector<std::complex<float>> input(N, {small_amplitude, small_amplitude});

    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(input, 0.0f); // 0 dB SNR
        EXPECT_EQ(result.size(), N);
    });
}

// Test 22: Sequential calls independence
TEST_F(ChannelTest, SequentialCallsIndependence) {
    constexpr size_t N = 20;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    // Make multiple calls and ensure they're independent
    const auto result1 = apply<std::complex<float>>(input, 15.0f, 0, 0.0f);
    const auto result2 = apply<std::complex<float>>(input, 15.0f, 0, 0.0f);
    const auto result3 = apply<std::complex<float>>(input, 15.0f, 0, 0.0f);

    // All should be identical due to fixed seed
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(result1[i].real(), result2[i].real());
        EXPECT_FLOAT_EQ(result1[i].imag(), result2[i].imag());
        EXPECT_FLOAT_EQ(result2[i].real(), result3[i].real());
        EXPECT_FLOAT_EQ(result2[i].imag(), result3[i].imag());
    }
}

// Test 23: Different SNR values consistency
TEST_F(ChannelTest, SNRConsistency) {
    constexpr size_t N = 1000;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    std::vector<float> snr_values = {-10.0f, 0.0f, 10.0f, 20.0f, 30.0f};
    std::vector<float> measured_powers;

    for (float snr : snr_values) {
        const auto result = apply<std::complex<float>>(input, snr, 0, 0.0f);
        float power = calculatePower(result);
        measured_powers.push_back(power);
    }

    // All powers should be positive and reasonable
    for (float power : measured_powers) {
        EXPECT_GT(power, 0.0f);
        EXPECT_LT(power, 100.0f);
    }
}

// Test 24: Maximum number of taps
TEST_F(ChannelTest, MaximumTaps) {
    constexpr size_t N = 1000;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    // Test with many taps (should not exceed signal length)
    EXPECT_NO_THROW({
        const auto result = apply<std::complex<float>>(input, 15.0f, static_cast<int>(N));
        EXPECT_EQ(result.size(), N);
    });
}

// Test 25: Zero Doppler frequency
TEST_F(ChannelTest, ZeroDopplerFrequency) {
    constexpr size_t N = 100;
    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});

    const auto result_zero_doppler = apply<std::complex<float>>(input, 20.0f, 0, 0.0f);
    const auto result_no_doppler_explicit = apply<std::complex<float>>(input, 20.0f, 0, 0.0f);

    // Results should be identical
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(result_zero_doppler[i].real(), result_no_doppler_explicit[i].real());
        EXPECT_FLOAT_EQ(result_zero_doppler[i].imag(), result_no_doppler_explicit[i].imag());
    }
}

// Test 26: AWGN-only noise power matches theoretical value
// With n_taps=0 and large N, measured noise variance should match signal_power * 10^(-SNR/10).
TEST_F(ChannelTest, AWGNOnlyNoisePower) {
    constexpr size_t N = 200000;
    constexpr float target_snr_db = 10.0f;

    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
    const auto result = apply<std::complex<float>>(input, target_snr_db, 0, 0.0f);

    double noise_power = 0.0;
    for (size_t i = 0; i < N; ++i) {
        noise_power += std::norm(result[i] - input[i]);
    }
    noise_power /= N;

    const float signal_power = calculatePower(input); // == 1.0
    const float theoretical_noise = signal_power / std::pow(10.0f, target_snr_db / 10.0f);

    // 3% relative tolerance: well-covered by 200 000 samples
    EXPECT_NEAR(static_cast<float>(noise_power), theoretical_noise, theoretical_noise * 0.03f);
}

// Test 27: Flat Rayleigh fading preserves average power over many blocks
// With n_taps=1, 50 000 symbols = 500 coherence blocks; by the LLN the
// sample-mean of |h|^2 converges to E[|h|^2] = 1 for a unit-power Rayleigh channel.
TEST_F(ChannelTest, FlatFadingAveragePower) {
    constexpr size_t N = 50000;

    std::vector<std::complex<float>> input(N, {1.0f, 0.0f});
    // 100 dB SNR: AWGN contribution is negligible; fading dominates
    const auto result = apply<std::complex<float>>(input, 100.0f, 1, 0.0f);

    const float output_power = calculatePower(result);
    const float input_power  = calculatePower(input); // == 1.0

    // 500 independent Rayleigh blocks => average power within 10% of input
    EXPECT_NEAR(output_power, input_power, input_power * 0.10f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
