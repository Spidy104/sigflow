//
// Created by scien on 08-11-2025.
//

#ifndef LIBDSP_CHANNEL_H
#define LIBDSP_CHANNEL_H

#include <complex>
#include <span>
#include <vector>
#include <concepts>

namespace sigflow::channel {

    // Constraint: only float complex supported
    template<typename T>
    concept ComplexFloat = std::same_as<T, std::complex<float>>;

    template<ComplexFloat T>
    [[nodiscard]] auto apply(std::span<const T> input,
                            float snr_db,
                            int n_taps = 8,
                            float doppler_hz = 10.0f)
        -> std::vector<T>;

    extern template std::vector<std::complex<float>>
    apply<std::complex<float>>(std::span<const std::complex<float>>, float, int, float);

} // namespace sigflow::channel

#endif //LIBDSP_CHANNEL_H
