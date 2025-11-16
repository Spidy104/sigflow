#include "../include/dsp.h"
#include "../include/channel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;

template<typename T>
auto to_span(const py::array_t<T, py::array::c_style>& arr) -> std::span<const T> {
    py::buffer_info info = arr.request();
    return {static_cast<const T*>(info.ptr), static_cast<std::size_t>(info.size)};
}

... (trimmed) ...
