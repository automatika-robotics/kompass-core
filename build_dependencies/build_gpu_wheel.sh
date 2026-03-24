#!/bin/bash
set -xe

# =============================================================================
# build_gpu_wheel.sh
# Builds a GPU-enabled kompass-core wheel on Ubuntu.
# - LLVM/Clang + AdaptiveCpp: installed via apt/llvm.sh (fast)
# - OMPL/FCL/Boost: built from source via Conan (static, portable)
# - acpp runtime: bundled into wheel via repair script
#
# Usage: build_gpu_wheel.sh [python_executable]
#   python_executable: path to python (default: python3)
# =============================================================================

ADAPTIVE_CPP_URL="https://github.com/AdaptiveCpp/AdaptiveCpp"
ADAPTIVE_CPP_VERSION="v25.10.0"
LLVM_VERSION=17
ACPP_DEPLOY_DIR="/tmp/acpp_deploy"
CONAN_BUILD_DIR="/tmp/conan_build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
PYTHON="${1:-python3}"

export PIP_BREAK_SYSTEM_PACKAGES=1

# Helper: run conan via python module so it works under sudo
CONAN="$PYTHON -m conans.cli.cli"

# -----------------------------------------------------------------------------
# 1. Install LLVM/Clang (skipped if already present)
# -----------------------------------------------------------------------------
if ! command -v clang++-${LLVM_VERSION} &>/dev/null; then
    wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- ${LLVM_VERSION}
fi

apt-get install -y \
    "libclang-${LLVM_VERSION}-dev" "clang-tools-${LLVM_VERSION}" \
    "libomp-${LLVM_VERSION}-dev" "llvm-${LLVM_VERSION}-dev" "lld-${LLVM_VERSION}"

LLVM_DIR=$(llvm-config-${LLVM_VERSION} --cmakedir)
CLANG_EXE=$(which clang++-${LLVM_VERSION})

# -----------------------------------------------------------------------------
# 2. Build & Install AdaptiveCpp (skipped if already present)
# -----------------------------------------------------------------------------
if ! command -v acpp &>/dev/null; then
    cd /tmp
    git clone --depth 1 --branch "$ADAPTIVE_CPP_VERSION" "$ADAPTIVE_CPP_URL"
    cd AdaptiveCpp && mkdir -p build && cd build

    # Install minimal CUDA stubs to satisfy CMake's find_package(CUDA)
    # Only headers/stubs needed — no driver, no runtime, no GPU required
    if ! command -v nvcc &>/dev/null; then
        UBUNTU_VER=$(lsb_release -rs | tr -d '.')
        ARCH=$(dpkg --print-architecture)
        case "$ARCH" in
            amd64) NVIDIA_ARCH="x86_64" ;;
            arm64) NVIDIA_ARCH="sbsa" ;;
            *) NVIDIA_ARCH="$ARCH" ;;
        esac
        wget -qO /tmp/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VER}/${NVIDIA_ARCH}/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb
        apt-get update -qq
        # Find the latest available CUDA 12.x packages and install
        CUDA_VER=$(apt-cache search '^cuda-nvcc-12-' | sort -V | tail -1 | awk '{print $1}' | sed 's/cuda-nvcc//')
        echo "Installing CUDA stubs: cuda-nvcc${CUDA_VER}, cuda-cudart-dev${CUDA_VER}"
        apt-get install -y -qq "cuda-nvcc${CUDA_VER}" "cuda-cudart-dev${CUDA_VER}"
    fi

    CXX="$CLANG_EXE" cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DLLVM_DIR="$LLVM_DIR" \
        -DCLANG_EXECUTABLE_PATH="$CLANG_EXE" \
        -DWITH_CUDA_BACKEND=ON \
        ..
    make install -j$(nproc)
    cd /tmp && rm -rf AdaptiveCpp
fi

acpp --acpp-version

# -----------------------------------------------------------------------------
# 3. Deploy acpp runtime (skipped if already deployed)
# -----------------------------------------------------------------------------
if [ ! -d "$ACPP_DEPLOY_DIR" ] || [ -z "$(ls -A "$ACPP_DEPLOY_DIR" 2>/dev/null)" ]; then
    mkdir -p "$ACPP_DEPLOY_DIR"
    # Deploy core component (OMP backend, host JIT, LLVM)
    acpp --acpp-deploy=core:"$ACPP_DEPLOY_DIR"
    # Manually add CUDA components (not in core deploy component)
    # The cuda deploy component also wants libdevice.10.bc which the user
    # provides via their CUDA installation at runtime
    ACPP_LIB=/usr/local/lib
    # CUDA runtime backend (submits work to NVIDIA GPUs)
    if [ -f "$ACPP_LIB/hipSYCL/librt-backend-cuda.so" ]; then
        cp "$ACPP_LIB/hipSYCL/librt-backend-cuda.so" \
           "$ACPP_DEPLOY_DIR/hipSYCL/"
    fi
    # PTX JIT translator (compiles LLVM IR → PTX)
    if [ -f "$ACPP_LIB/hipSYCL/llvm-to-backend/libllvm-to-ptx.so" ]; then
        cp "$ACPP_LIB/hipSYCL/llvm-to-backend/libllvm-to-ptx.so" \
           "$ACPP_DEPLOY_DIR/hipSYCL/llvm-to-backend/"
    fi
    # PTX bitcode library
    if [ -f "$ACPP_LIB/hipSYCL/bitcode/libkernel-sscp-ptx-full.bc" ]; then
        cp "$ACPP_LIB/hipSYCL/bitcode/libkernel-sscp-ptx-full.bc" \
           "$ACPP_DEPLOY_DIR/hipSYCL/bitcode/"
    fi

    # Bundle indirect transitive dependencies that acpp may have missed
    # (skip libcuda, libdrm, libc, linux-vdso — these come from the driver/OS)
    echo "Checking for missing transitive dependencies..."
    LD_LIBRARY_PATH="$ACPP_DEPLOY_DIR:$LD_LIBRARY_PATH" \
        ldd $(find "$ACPP_DEPLOY_DIR" -type f -name "*.so*" | grep -v '.bc') 2>/dev/null \
        | grep -v "$ACPP_DEPLOY_DIR" \
        | grep "=> /" \
        | awk '{print $3}' \
        | sort -u \
        | grep -v -E "libc\.|libm\.|libdl\.|libpthread\.|librt\.|linux-vdso|libcuda|libdrm|ld-linux" \
        | while read dep; do
            if [ -f "$dep" ]; then
                echo "  Bundling transitive dep: $dep"
                cp -n "$dep" "$ACPP_DEPLOY_DIR/" 2>/dev/null || true
            fi
        done
fi

# -----------------------------------------------------------------------------
# 4. Build OMPL/FCL via Conan (static linking for portability)
# -----------------------------------------------------------------------------
if [ ! -d "$CONAN_BUILD_DIR" ]; then
    $PYTHON -m pip install conan "cmake<3.30"

    # Add Python bin dir to PATH so conan command is found
    PYTHON_BIN_DIR=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('scripts'))")
    export PATH="$PYTHON_BIN_DIR:$PATH"

    # Configure Conan profile with system compiler (not acpp)
    CXX=g++ conan profile detect --force
    sed -i 's/compiler.cppstd=.*/compiler.cppstd=gnu17/g' ~/.conan2/profiles/default

    # Static libs for everything; OMPL 1.7.0 requires shared on non-MSVC
    cat >> ~/.conan2/profiles/default <<EOF
[options]
*:shared=False
*:fPIC=True
ompl/*:shared=True
EOF

    # Clone OMPL recipe fork
    # TODO: Remove fork when recipe merged upstream
    mkdir -p /tmp/conan_recipes
    git clone --depth 1 https://github.com/aleph-ra/conan-center-index.git /tmp/conan_recipes

    # Build FCL and OMPL
    conan remove "ompl/*" -c || true
    conan remove "fcl/*" -c || true

    conan create /tmp/conan_recipes/recipes/fcl/all --version=0.7.0 \
        --build=missing

    conan create /tmp/conan_recipes/recipes/ompl/all --version=1.7.0 \
        --build=missing --build="b2/*" \
        -c tools.build:jobs=$(nproc)

    # Generate CMake toolchain
    mkdir -p "$CONAN_BUILD_DIR"
    conan install \
        --requires="ompl/1.7.0" \
        --requires="fcl/0.7.0" \
        -g CMakeDeps \
        -g CMakeToolchain \
        --output-folder="$CONAN_BUILD_DIR" \
        --build=missing \
        --build="b2/*" \
        -c "tools.cmake.cmaketoolchain:extra_variables={'CMAKE_POLICY_VERSION_MINIMUM':'3.5'}"

    rm -rf /tmp/conan_recipes
fi

# Install OMPL shared lib to system path (needed at build time)
OMPL_LIB_DIR="$(find ~/.conan2 -name 'libompl.so' -printf '%h\n' -quit)"
if [ -n "$OMPL_LIB_DIR" ]; then
    cp "$OMPL_LIB_DIR"/libompl.so* /usr/local/lib/
    ldconfig
fi

# -----------------------------------------------------------------------------
# 5. Build wheel
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"
rm -rf dist/

# Version pins must match pyproject.toml [build-system] requires
$PYTHON -m pip install patchelf "scikit-build-core>=0.8" "nanobind>=1.8,<2.9.2" "packaging>=22.0"

# Use acpp as compiler; point CMake at Conan-generated find modules for OMPL/FCL
# Use CMAKE_PREFIX_PATH instead of CMAKE_TOOLCHAIN_FILE to avoid Conan
# overriding the build generator (Ninja vs Make conflict with scikit-build)
CXX=acpp SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONAN_BUILD_DIR" \
    $PYTHON -m pip wheel --no-build-isolation -w dist/ .

# -----------------------------------------------------------------------------
# 6. Repair wheel (inject acpp libs + fix RPATHs)
# Also bundle OMPL shared lib since it was built via Conan
# -----------------------------------------------------------------------------
if [ -n "$OMPL_LIB_DIR" ]; then
    cp -L "$OMPL_LIB_DIR"/libompl.so* "$ACPP_DEPLOY_DIR/"
fi

WHEEL=$(ls dist/kompass_core-*.whl | head -1)
bash build_dependencies/repair_gpu_wheel.sh "$WHEEL" dist/repaired/
echo "=== GPU wheel built: ==="
ls -lh dist/repaired/
