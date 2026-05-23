#include "../include/dsp.h"
#include "fftw3.h"
#include <cmath>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <algorithm>

// FFTW requires aligned memory; use fftwf_malloc/free
struct fftw_deleter {
    void operator()(void* p) const { fftwf_free(p); }
};

using fftw_buffer = std::unique_ptr<fftwf_complex[], fftw_deleter>;

namespace sigflow::dsp {

template<ComplexFloat T>
[[nodiscard]] auto process(std::span<const T> input,
                          std::string_view mode)
    -> std::vector<T>
{
    if (mode != "fft") [[unlikely]] {
        throw std::invalid_argument("Only mode=\"fft\" supported");
    }

    const std::size_t n = input.size();
    if (n == 0) [[unlikely]] {
        return {};
    }

    // Allocate aligned buffer for FFTW
    fftw_buffer in_buffer{
        static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * n))
    };
    fftw_buffer out_buffer{
        static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * n))
    };

    // Copy input (complex<float> is layout-compatible with fftwf_complex)
    std::memcpy(
        in_buffer.get(),
        input.data(),
        n * sizeof(fftwf_complex)
    );

    // Create plan (ESTIMATE for validation; MEASURE in prod)
    auto plan = std::unique_ptr<fftwf_plan_s, decltype(&fftwf_destroy_plan)>{
        fftwf_plan_dft_1d(
            static_cast<int>(n),
            in_buffer.get(),
            out_buffer.get(),
            FFTW_FORWARD,
            FFTW_ESTIMATE
        ),
        &fftwf_destroy_plan
    };

    if (!plan) [[unlikely]] {
        throw std::runtime_error("FFTW plan creation failed");
    }

    // Execute FFT
    fftwf_execute_dft(plan.get(), in_buffer.get(), out_buffer.get());

    // Convert back to std::complex<float> vector
    std::vector<T> result(n);
    std::memcpy(
        result.data(),
        out_buffer.get(),
        n * sizeof(T)
    );

    // Normalize — parallelised with OpenMP when available, sequential otherwise
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    const auto  len   = static_cast<std::ptrdiff_t>(n);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < len; ++i) {
        result[static_cast<std::size_t>(i)] *= scale;
    }
#else
    for (auto& c : result) c *= scale;
#endif

    return result;
}

// Explicit instantiation
template std::vector<std::complex<float>>
process<std::complex<float>>(std::span<const std::complex<float>>, std::string_view);

} // namespace sigflow::dsp
