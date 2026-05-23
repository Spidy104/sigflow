#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "../include/dsp.h"
#include "../include/channel.h"

#include <complex>
#include <cstring>
#include <span>
#include <vector>

namespace nb = nanobind;
using nb::literals::operator""_a;

using cf32 = std::complex<float>;

// 1-D contiguous read-only complex64 input (CPU memory, any numpy-compatible source)
using InArray  = nb::ndarray<const cf32, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
// 1-D owned numpy complex64 output
using OutArray = nb::ndarray<nb::numpy, cf32, nb::ndim<1>>;

/// Transfer a std::vector into a new heap-allocated numpy array (single memcpy, zero extra copies).
[[nodiscard]] static OutArray vec_to_numpy(std::vector<cf32> v) {
    const std::size_t n = v.size();
    auto* buf = new cf32[n];
    std::memcpy(buf, v.data(), n * sizeof(cf32));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<cf32*>(p); });
    const std::size_t shape[1] = {n};
    return OutArray(buf, 1, shape, owner);
}

NB_MODULE(libdsp, m) {
    m.doc() = "SigFlow DSP Library \u2014 C++20 signal processing and channel simulation";

    m.def("dsp_process",
        [](InArray signal, const std::string& mode) -> OutArray {
            std::span<const cf32> sp{signal.data(), signal.size()};
            return vec_to_numpy(sigflow::dsp::process<cf32>(sp, mode));
        },
        "signal"_a, "mode"_a = "fft",
        "Run FFT on a 1-D complex64 NumPy array. Only ``mode='fft'`` is supported.");

    m.def("apply_channel",
        [](InArray signal, float snr_db, int n_taps, float doppler_hz) -> OutArray {
            std::span<const cf32> sp{signal.data(), signal.size()};
            return vec_to_numpy(sigflow::channel::apply<cf32>(sp, snr_db, n_taps, doppler_hz));
        },
        "signal"_a, "snr_db"_a,
        "n_taps"_a = 8, "doppler_hz"_a = 10.0f,
        "Apply channel impairments (AWGN, Rayleigh fading, Doppler shift) to a complex64 signal.");
}