#!/bin/bash
set -xe

# =============================================================================
# build_gpu_wheel.sh
# Builds a GPU-enabled kompass-core wheel on Ubuntu.
# Installs LLVM/Clang, builds AdaptiveCpp, then builds the wheel with acpp.
# Assumes OMPL/FCL/Eigen are already installed via apt.
#
# Usage: build_gpu_wheel.sh [python_executable]
#   python_executable: path to python (default: python3)
# =============================================================================

ADAPTIVE_CPP_URL="https://github.com/AdaptiveCpp/AdaptiveCpp"
ADAPTIVE_CPP_VERSION="v25.10.0"
LLVM_VERSION=17
ACPP_DEPLOY_DIR="/tmp/acpp_deploy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
PYTHON="${1:-python3}"

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

    CXX="$CLANG_EXE" cmake \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DLLVM_DIR="$LLVM_DIR" \
        -DCLANG_EXECUTABLE_PATH="$CLANG_EXE" \
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
    acpp --acpp-deploy=core:"$ACPP_DEPLOY_DIR"
fi

# -----------------------------------------------------------------------------
# 4. Build wheel
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"
rm -rf dist/
export PIP_BREAK_SYSTEM_PACKAGES=1
# Version pins must match pyproject.toml [build-system] requires
$PYTHON -m pip install patchelf "scikit-build-core>=0.8" "nanobind>=1.8,<2.9.2" "packaging>=22.0"
CXX=acpp $PYTHON -m pip wheel --no-build-isolation -w dist/ .

# -----------------------------------------------------------------------------
# 5. Repair wheel (inject acpp libs + fix RPATHs)
# -----------------------------------------------------------------------------
WHEEL=$(ls dist/kompass_core-*.whl | head -1)
bash build_dependencies/repair_gpu_wheel.sh "$WHEEL" dist/repaired/
echo "=== GPU wheel built: ==="
ls -lh dist/repaired/
