#!/bin/bash
# Build script for Usadel C++ solver on macOS

set -e

echo "Building Usadel C++ solver..."
echo "=============================="

# Clear any GCC environment variables that might interfere
unset CPLUS_INCLUDE_PATH
unset C_INCLUDE_PATH
unset CPATH
unset LIBRARY_PATH

# Navigate to cpp_solver directory
cd "$(dirname "$0")/cpp_solver"

# Remove old build directory completely
rm -rf build
mkdir -p build
cd build

# Configure with CMake using Apple Clang explicitly
echo "Configuring with CMake..."
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$(which python3)

# Build
echo "Building..."
make -j$(sysctl -n hw.ncpu)

# Copy the module to parent directory
echo "Installing module..."
cp usadel_cpp*.so ../../

echo ""
echo "Build complete!"
echo "Module installed to: $(dirname $0)"
echo ""
echo "Test with: python -c 'import usadel_cpp; print(usadel_cpp.__doc__)'"
