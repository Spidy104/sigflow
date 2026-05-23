#ifndef LIBDSP_DSP_H
#define LIBDSP_DSP_H


#include <complex>
#include <span>
#include <string_view>
#include <concepts>
#include <vector>

namespace sigflow::dsp {
    // Constraint: only float complex supported
    template<typename T>
    concept ComplexFloat = std::same_as<T, std::complex<float>>;

    // Main interface: takes a view, returns owning array (to be wrapped by nanobind)
    template<ComplexFloat T>
    [[nodiscard]] auto process(std::span<const T> input,
                              std::string_view mode = "fft")
        -> std::vector<T>;

    // Explicit instantiation declaration (for binding.cpp)
    extern template std::vector<std::complex<float>>
    process<std::complex<float>>(std::span<const std::complex<float>>, std::string_view);

} // namespace sigflow::dsp



#endif //LIBDSP_DSP_H