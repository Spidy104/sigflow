#!/bin/bash
# Quick FFTW3 installation check and fix

echo "=== FFTW3 Installation Verification ==="
echo ""

# Check if package is installed
if pacman -Q mingw-w64-ucrt-x86_64-fftw &>/dev/null; then
    VER="$(pacman -Q mingw-w64-ucrt-x86_64-fftw)"
    echo "✓ Package installed: $VER"
else
    echo "❌ Package NOT installed"
    echo ""
    echo "Installing now..."
    pacman -S --noconfirm mingw-w64-ucrt-x86_64-fftw
    exit 0
fi

echo ""
echo "Checking files:"

# Check header
if [[ -f "$MINGW_PREFIX/include/fftw3.h" ]]; then
    echo "✓ Header: $MINGW_PREFIX/include/fftw3.h"
else
    echo "❌ Header missing: $MINGW_PREFIX/include/fftw3.h"
fi

# Check library
if [[ -f "$MINGW_PREFIX/lib/libfftw3f.a" ]]; then
    echo "✓ Library (static): $MINGW_PREFIX/lib/libfftw3f.a"
else
    echo "⚠️  Static library missing: $MINGW_PREFIX/lib/libfftw3f.a"
fi

if [[ -f "$MINGW_PREFIX/bin/libfftw3f-3.dll" ]]; then
    echo "✓ Library (shared): $MINGW_PREFIX/bin/libfftw3f-3.dll"
else
    echo "⚠️  Shared library missing: $MINGW_PREFIX/bin/libfftw3f-3.dll"
fi

echo ""
echo "MINGW_PREFIX = $MINGW_PREFIX"
echo ""

# List all fftw files
echo "All FFTW3 files installed:"
pacman -Ql mingw-w64-ucrt-x86_64-fftw | grep -E '\.(h|a|dll)$' | head -20

