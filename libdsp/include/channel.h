#ifndef LIBDSP_CHANNEL_H
#define LIBDSP_CHANNEL_H

#include <complex>
#include <span>
#include <vector>
#include <concepts>

namespace sigflow::channel {

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

} 

#endif //LIBDSP_CHANNEL_H
