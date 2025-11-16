#ifndef LIBDSP_DSP_H
#define LIBDSP_DSP_H

#include <complex>
#include <span>
#include <string_view>
#include <concepts>
#include <vector>

namespace sigflow::dsp {
    template<typename T>
    concept ComplexFloat = std::same_as<T, std::complex<float>>;

    template<ComplexFloat T>
    [[nodiscard]] auto process(std::span<const T> input,
                              std::string_view mode = "fft")
        -> std::vector<T>;

    extern template std::vector<std::complex<float>>
    process<std::complex<float>>(std::span<const std::complex<float>>, std::string_view);

} 

#endif //LIBDSP_DSP_H
