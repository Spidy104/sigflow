# SigFlow — C++20 DSP Core
# Toolchain: MSVC (cl.exe) from Visual Studio 2022
# Build backend: scikit-build-core (pip install -e .) or standalone via build.ps1
cmake_minimum_required(VERSION 3.20)
project(libdsp LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# CMP0091: enable cmake_msvc_runtime_library variable
cmake_policy(SET CMP0091 NEW)

# Position-independent code (required for shared libs / .pyd)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for IDE tooling (clangd / VS Code)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# MSVC runtime -- must match Python's CRT (always MultiThreadedDLL = /MD)
# ---------------------------------------------------------------------------
if(MSVC)
    # /MD in Release and Debug -- must match the Python interpreter's CRT
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    # UTF-8 source & execution character sets
    add_compile_options(/utf-8)
endif()

# ---------------------------------------------------------------------------
# Python & nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python: ${Python_EXECUTABLE} (${Python_VERSION})")

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _nb_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nb_result
)
if(NOT _nb_result EQUAL 0)
    message(FATAL_ERROR
        "nanobind not found for Python '${Python_EXECUTABLE}'.\n"
        "Run:  pip install nanobind   (or: uv sync)")
endif()
find_package(nanobind CONFIG REQUIRED HINTS "${_nb_cmake_dir}")
message(STATUS "Found nanobind cmake dir: ${_nb_cmake_dir}")

# ---------------------------------------------------------------------------
# FFTW3 — official Windows prebuilt DLL bundle (MSVC-compatible)
# ---------------------------------------------------------------------------
# Download the official fftw.org Windows 64-bit DLL package, then generate
# an MSVC import library (.lib) from the bundled .def file using lib.exe.
# This is the recommended approach for MSVC; no MinGW/MSYS2 needed.
# ---------------------------------------------------------------------------
include(FetchContent)

# Suppress deprecated FetchContent_Populate warning (CMP0169) — still the
# correct approach for non-CMake content (prebuilt DLL bundles).
cmake_policy(SET CMP0169 OLD)

FetchContent_Declare(
    fftw3_win64
    # Latest official Windows 64-bit prebuilt DLL package from fftw.org
    URL      https://fftw.org/pub/fftw/fftw-3.3.5-dll64.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Populate without adding as a CMake subdirectory (it's not a CMake project)
FetchContent_GetProperties(fftw3_win64)
if(NOT fftw3_win64_POPULATED)
    FetchContent_Populate(fftw3_win64)
endif()

set(FFTW3_DIR "${fftw3_win64_SOURCE_DIR}")
message(STATUS "FFTW3 prebuilt dir: ${FFTW3_DIR}")

set(FFTW3F_DLL "${FFTW3_DIR}/libfftw3f-3.dll")
set(FFTW3F_DEF "${FFTW3_DIR}/libfftw3f-3.def")
set(FFTW3F_LIB "${FFTW3_DIR}/libfftw3f-3.lib")

# Generate import library with lib.exe (once; cached on disk)
if(MSVC AND NOT EXISTS "${FFTW3F_LIB}")
    # lib.exe lives next to cl.exe
    get_filename_component(_msvc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_lib_exe "${_msvc_bin}/lib.exe")
    if(NOT EXISTS "${_lib_exe}")
        message(FATAL_ERROR "lib.exe not found at ${_lib_exe}")
    endif()
    message(STATUS "Generating MSVC import library for FFTW3f...")
    execute_process(
        COMMAND "${_lib_exe}"
                "/def:${FFTW3F_DEF}"
                "/out:${FFTW3F_LIB}"
                "/machine:x64"
        RESULT_VARIABLE _lib_result
        OUTPUT_VARIABLE _lib_out
        ERROR_VARIABLE  _lib_err
    )
    if(NOT _lib_result EQUAL 0)
        message(FATAL_ERROR "lib.exe failed:\n${_lib_out}\n${_lib_err}")
    endif()
    message(STATUS "Created ${FFTW3F_LIB}")
endif()

# Imported target so CMake tracks the dep properly
add_library(fftw3f SHARED IMPORTED GLOBAL)
set_target_properties(fftw3f PROPERTIES
    IMPORTED_IMPLIB             "${FFTW3F_LIB}"
    IMPORTED_LOCATION           "${FFTW3F_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_DIR}"
)

# ---------------------------------------------------------------------------
# OpenMP  (MSVC: /openmp  |  GCC/Clang: -fopenmp)
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found — parallel loops will run sequentially")
endif()

# ---------------------------------------------------------------------------
# 1. STATIC LIBRARY — pure C++20 DSP core (no Python deps)
# ---------------------------------------------------------------------------
add_library(dsp_core STATIC
    src/channel.cpp
    include/dsp.h
    src/dsp.cpp
    include/channel.h
)

target_include_directories(dsp_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(dsp_core PRIVATE fftw3f)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dsp_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler-specific optimisation flags
if(MSVC)
    target_compile_options(dsp_core PRIVATE
        /O2           # Maximise speed
        /fp:fast      # Fast floating-point (equiv. to -ffinite-math-only -fno-math-errno)
        /W3           # Warning level 3
        /WX-          # Warnings not errors (nanobind internals can warn)
    )
    # AVX2 on x86-64 (matches -march=native for modern machines)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        target_compile_options(dsp_core PRIVATE /arch:AVX2)
    endif()
else()
    # GCC / Clang fallback (kept for cross-platform builds)
    target_compile_options(dsp_core PRIVATE
        -march=native -mtune=native -O3 -ffinite-math-only -fno-math-errno
    )
endif()

# ---------------------------------------------------------------------------
# 2. SHARED LIBRARY — nanobind Python extension (.pyd on Windows)
# ---------------------------------------------------------------------------
nanobind_add_module(libdsp NB_STATIC
    src/binding.cpp
)

target_link_libraries(libdsp PRIVATE dsp_core)

# ── Output for standalone build (build.ps1) ─────────────────────────────────
# When built directly (not via scikit-build-core), drop the .pyd + DLL into
# python_src/ so the repo is immediately usable without installing.
if(NOT SKBUILD)
    set_target_properties(libdsp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python_src"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python_src"
        PREFIX  ""
        SUFFIX  ".pyd"
    )
    # Copy FFTW3 DLL next to libdsp.pyd
    add_custom_command(TARGET libdsp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW3F_DLL}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../python_src"
        COMMENT "Bundling libfftw3f-3.dll → python_src/"
        VERBATIM
    )
endif()

# ── Install rules for scikit-build-core wheel ────────────────────────────────
# scikit-build-core sets SKBUILD=1 and routes install() into the wheel.
# cmake.install-dir in pyproject.toml controls the destination prefix.
install(TARGETS libdsp
    LIBRARY DESTINATION .   # → python_src/ inside the wheel (see pyproject.toml)
    RUNTIME DESTINATION .
)
install(FILES "${FFTW3F_DLL}" DESTINATION .)

# ---------------------------------------------------------------------------
# 3. TESTS — Google Test
# ---------------------------------------------------------------------------
enable_testing()

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(
        googletest
        # tar.gz avoids CMake zip-extraction timestamp issues on FAT/exFAT drives
        URL      https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    # MSVC: link against DLL CRT (/MD) to match Python
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_dsp     tests/test_dsp.cpp)
add_executable(test_channel tests/test_channel.cpp)

target_link_libraries(test_dsp     PRIVATE dsp_core GTest::gtest GTest::gtest_main)
target_link_libraries(test_channel PRIVATE dsp_core GTest::gtest GTest::gtest_main)

# Copy FFTW3 DLL next to test executables so they find it at runtime
# (test_dsp uses FFTW directly; without this it fails with 0xc0000135)
foreach(_test_target test_dsp test_channel)
    add_custom_command(TARGET ${_test_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW3F_DLL}"
                "$<TARGET_FILE_DIR:${_test_target}>"
        COMMENT "Copying libfftw3f-3.dll -> ${_test_target} dir"
        VERBATIM
    )
endforeach()

add_test(NAME DSPTests     COMMAND test_dsp)
add_test(NAME ChannelTests COMMAND test_channel)

add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_dsp test_channel
    COMMENT "Running all unit tests"
    VERBATIM
)
