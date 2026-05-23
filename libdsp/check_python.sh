#!/bin/bash
# Python Detection Diagnostic Script
# Run this to see which Python versions are available

echo "=== Python Detection Diagnostic ==="
echo ""

# Check MSYSTEM
echo "1. MSYSTEM Environment:"
if [[ -v MSYSTEM ]]; then
    echo "   MSYSTEM = $MSYSTEM"
else
    echo "   ❌ MSYSTEM is not set (not in MSYS2 shell?)"
fi
echo ""

# Check 'python' in PATH
echo "2. Python in PATH:"
if command -v python &>/dev/null; then
    PYTHON_PATH="$(which python)"
    PYTHON_VER="$(python --version 2>&1)"
    echo "   Location: $PYTHON_PATH"
    echo "   Version:  $PYTHON_VER"

    if [[ "$PYTHON_PATH" == /ucrt64/bin/python* ]]; then
        echo "   ✓ This is UCRT64 Python"
    else
        echo "   ⚠️  This is NOT UCRT64 Python!"
    fi
else
    echo "   ❌ 'python' not found in PATH"
fi
echo ""

# Check for UCRT64 Python versions
echo "3. Available UCRT64 Python versions:"
for py in python3.12 python3.13 python3.14 python3 python; do
    if [[ -x "/ucrt64/bin/$py" ]]; then
        VER="$(/ucrt64/bin/$py --version 2>&1)"
        echo "   ✓ /ucrt64/bin/$py -> $VER"
    fi
done
echo ""

# Check what CMake would find
echo "4. What CMake will find:"
UCRT64_PYTHON=""
for py in python3.12 python3.13 python3.14 python3 python; do
    if [[ -x "/ucrt64/bin/$py" ]]; then
        UCRT64_PYTHON="/ucrt64/bin/$py"
        break
    fi
done

if [[ -n "$UCRT64_PYTHON" ]]; then
    PYTHON_VER="$($UCRT64_PYTHON --version 2>&1)"
    echo "   Will use: $UCRT64_PYTHON"
    echo "   Version:  $PYTHON_VER"
else
    echo "   ❌ No UCRT64 Python found!"
    echo "   Install with: pacman -S mingw-w64-ucrt-x86_64-python"
fi
echo ""

# Check Python packages
if [[ -n "$UCRT64_PYTHON" ]]; then
    echo "5. Checking installed packages:"

    # Check if numpy is installed
    if $UCRT64_PYTHON -c "import numpy" &>/dev/null; then
        NUMPY_VER="$($UCRT64_PYTHON -c "import numpy; print(numpy.__version__)")"
        echo "   ✓ NumPy $NUMPY_VER"
    else
        echo "   ❌ NumPy not found"
        echo "      Install: pacman -S mingw-w64-ucrt-x86_64-python-numpy"
    fi

    # Check nanobind (installed via uv, not pacman)
    if $UCRT64_PYTHON -c "import nanobind" &>/dev/null; then
        NB_VER="$($UCRT64_PYTHON -c "import nanobind; print(nanobind.__version__)")" 
        echo "   ✓ nanobind $NB_VER"
    else
        echo "   ❌ nanobind not found"
        echo "      Install via uv: cd .. && uv sync"
        echo "      Or manually:    $UCRT64_PYTHON -m pip install nanobind"
    fi

    # Check FFTW3
    if pacman -Q mingw-w64-ucrt-x86_64-fftw &>/dev/null; then
        FFTW_VER="$(pacman -Q mingw-w64-ucrt-x86_64-fftw | awk '{print $2}')"
        echo "   ✓ FFTW3 $FFTW_VER"

        # Verify fftw3.h exists
        if [[ -f "$MINGW_PREFIX/include/fftw3.h" ]]; then
            echo "      Header: $MINGW_PREFIX/include/fftw3.h ✓"
        else
            echo "      ⚠️  Header not found at $MINGW_PREFIX/include/fftw3.h"
        fi
    else
        echo "   ❌ FFTW3 not found"
        echo "      Install: pacman -S mingw-w64-ucrt-x86_64-fftw"
    fi

    # Check Eigen3
    if pacman -Q mingw-w64-ucrt-x86_64-eigen3 &>/dev/null; then
        EIGEN_VER="$(pacman -Q mingw-w64-ucrt-x86_64-eigen3 | awk '{print $2}')"
        echo "   ✓ Eigen3 $EIGEN_VER"

        # Verify Eigen/Core exists
        if [[ -f "$MINGW_PREFIX/include/eigen3/Eigen/Core" ]]; then
            echo "      Header: $MINGW_PREFIX/include/eigen3/Eigen/Core ✓"
        else
            echo "      ⚠️  Header not found at $MINGW_PREFIX/include/eigen3/Eigen/Core"
        fi
    else
        echo "   ❌ Eigen3 not found"
        echo "      Install: pacman -S mingw-w64-ucrt-x86_64-eigen3"
    fi
fi
echo ""

# Check MINGW_PREFIX
echo "6. MSYS2 Environment:"
if [[ -n "$MINGW_PREFIX" ]]; then
    echo "   MINGW_PREFIX = $MINGW_PREFIX"
else
    echo "   ⚠️  MINGW_PREFIX not set"
    echo "      This should be set automatically in UCRT64 shell"
fi
echo ""

echo "=== Summary ==="
if [[ "$MSYSTEM" == "UCRT64" ]] && [[ -n "$UCRT64_PYTHON" ]]; then
    echo "✓ Ready to build!"
    echo "  Run: ./build.sh"
else
    echo "❌ Not ready to build. Fix the issues above."
fi

