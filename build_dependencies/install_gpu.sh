#!/bin/bash

# Stop command echoing
set +x

# Propogate errors and fail on error
set -eo pipefail

# Default log level: INFO
QUIET_MODE=false
# Debug mode off
DEBUG_MODE=false

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    case "$level" in
        INFO)
            [[ $QUIET_MODE == false ]] && echo -e "\033[1;34m[$timestamp] [INFO]\033[0m $message"
            ;;
        WARN)
            echo -e "\033[1;33m[$timestamp] [WARN]\033[0m $message" >&2
            ;;
        ERROR)
            echo -e "\033[1;31m[$timestamp] [ERROR]\033[0m $message" >&2
            ;;
        *)
            echo -e "\033[1;36m[$timestamp] [UNKNOWN]\033[0m $message" >&2
            ;;
    esac
}

# Function to check if running inside Docker or Podman or other containerized envs
is_in_container() {
    [[ -f /.dockerenv ]] && return 0
    grep -qE '(docker|podman)' /proc/self/cgroup 2>/dev/null && return 0
    grep -q 'containerd' /proc/self/cgroup 2>/dev/null && return 0
    [[ -f /run/.containerenv ]] && return 0
    return 1
}

# Function to check for sudo privileges
check_sudo() {
    if ! is_in_container && ! sudo -n true 2>/dev/null; then
        log WARN "This script requires sudo privileges. Please enter your password."
        sudo -v || { log ERROR "Failed to acquire sudo privileges. Exiting."; exit 1; }
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &>/dev/null
}

# Detect OS and Version
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID=$ID
        OS_VERSION_ID=$VERSION_ID
    else
        log ERROR "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi
}

# Function to determine if we need legacy support (Conan)
# Returns 0 (true) if legacy (old Ubuntu), 1 (false) if modern (Debian 11+, RPi OS, Ubuntu > 20)
is_legacy_system() {
    # If Debian or Raspbian, assume modern (Debian 10/Buster is old, but we assume 11+ for this context)
    if [[ "$OS_ID" == "debian" || "$OS_ID" == "raspbian" ]]; then
        return 1
    elif [[ "$OS_ID" == "ubuntu" ]]; then
        if [[ $(echo "$OS_VERSION_ID <= 20.04" | bc -l) == 1 ]]; then
            return 0
        fi
    fi
    return 1
}

# Function to install missing dependencies
install_dependencies() {
    log INFO "Updating package lists..."
    $SUDO apt update -y

    $SUDO apt install -y lsb-release bc wget gnupg

    detect_os
    log INFO "Detected OS: $OS_ID, Version: $OS_VERSION_ID"
    export DEBIAN_FRONTEND=noninteractive

    # Install specific CMake for Legacy Ubuntu
    if is_legacy_system; then
        log INFO "Legacy system detected. Installing latest cmake from Kitware..."
        $SUDO apt install -y software-properties-common
        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | $SUDO tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
        $SUDO apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
        $SUDO apt update -y && apt install -y kitware-archive-keyring
        $SUDO rm /etc/apt/trusted.gpg.d/kitware.gpg
        $SUDO apt update -y && apt install -y cmake
        $SUDO apt install -y --only-upgrade cmake
        local dependencies=("git" "make" "python3-pip" "zip" "unzip" "ninja-build" "curl" "tar" "pkg-config" "jq" "python3-minimal" "python3-apt" "python3-dev" "patch")
    else
        # Modern Debian/Raspbian/Ubuntu
        local dependencies=("git" "make" "cmake" "wget" "python3-pip" "zip" "unzip" "ninja-build" "curl" "tar" "pkg-config" "jq" "python3-minimal" "python3-apt" "python3-dev" "software-properties-common")
    fi

    # Install other tool dependencies
    local missing=()
    for dep in "${dependencies[@]}"; do
        if ! command_exists "$dep"; then
            missing+=("$dep")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        log INFO "Installing missing dependencies: ${missing[*]}"
        export DEBIAN_FRONTEND=noninteractive
        $SUDO apt install -y "${missing[@]}"
    else
        log INFO "All dependencies are already installed."
    fi

    # Ensure /usr/bin/python3 exists
    if [ ! -x /usr/bin/python3 ]; then
        log INFO "Fixing missing /usr/bin/python3 symlink..."
        $SUDO ln -sf $(command -v python3) /usr/bin/python3
    fi
}

# Function to check LLVM/Clang version
check_llvm_clang_version() {
    local llvm_version=$(llvm-config-$1 --version 2>/dev/null | cut -d. -f1 || echo "0")
    local clang_version=$(clang++-$1 --version 2>/dev/null | awk '/version/ {print $4}' | cut -d. -f1 || echo "0")
    [[ "$llvm_version" -ne "0" && "$clang_version" -ne "0" ]]
}

# Function to return LLVM/Clang version found in a specified range
check_llvm_clang_versions_in_range() {
    local min_version=$1
    local max_version=$2
    local found_version=0

    for (( version=min_version; version<=max_version; version++ )); do
        if check_llvm_clang_version $version; then
            found_version=$version
        fi
    done
    echo $found_version
}

#### kompass-core Install Script ####

KOMPASS_CORE_REPO="automatika-robotics/kompass-core"
KOMPASS_CORE_URL="https://github.com/$KOMPASS_CORE_REPO"
ADAPTIVE_CPP_URL="https://github.com/AdaptiveCpp/AdaptiveCpp"
ADAPTIVE_CPP_SOURCE_VERSION="v25.10.0"
DEFAULT_INSTALL_PREFIX="/usr/local"
DEFAULT_KEEP_SOURCE_FILES=false
MINIMUM_LLVM_VERSION=14
DEFAULT_NIGHTLY=false

INSTALL_PREFIX="$DEFAULT_INSTALL_PREFIX"
LLVM_VERSION=""
KEEP_SOURCE_FILES="$DEFAULT_KEEP_SOURCE_FILES"
NIGHTLY="$DEFAULT_NIGHTLY"
CUDA_ROOT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --acpp-install-prefix) INSTALL_PREFIX="$2"; shift ;;
        --llvm-version) LLVM_VERSION="$2"; shift ;;
        --cuda-root) CUDA_ROOT="$2"; shift ;;
        --keep-source-files) KEEP_SOURCE_FILES=true ;;
        --nightly) NIGHTLY=true ;;
        --quiet) QUIET_MODE=true ;;
        --debug) DEBUG_MODE=true ;;
        *) log ERROR "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Enable debugging if requested
if [[ $DEBUG_MODE == true ]]; then
    log INFO "Debug mode enabled."
    set -x
fi

# Check if sudo is required if not in a container
SUDO=$(is_in_container && echo "" || echo "sudo")

# Ensure sudo privileges if not in Docker
check_sudo

# Install tool dependencies
install_dependencies

# Check for LLVM/Clang versions
if [[ $LLVM_VERSION ]]; then
    # Check if given version is within range
    if (( LLVM_VERSION < 15 || LLVM_VERSION > 18 )); then
        log ERROR "LLVM Versions higher than 18 and lower than 15 are not compatible with kompass-core."
        exit 1
    fi
    if check_llvm_clang_version $LLVM_VERSION; then
        FOUND_LLVM_VERSION=$LLVM_VERSION
    else
        FOUND_LLVM_VERSION=0
    fi
else
    # Clang version 18 gives an error when compiling with boost version < 1.75
    # For Debian 13/Raspberry Pi OS, check up to 17 first
    LLVM_VERSION=17
    FOUND_LLVM_VERSION=$(check_llvm_clang_versions_in_range $MINIMUM_LLVM_VERSION $LLVM_VERSION)
fi

# Install LLVM/Clang if needed
if [ $FOUND_LLVM_VERSION -eq 0 ]; then
    log INFO "Attempting to install LLVM/Clang $LLVM_VERSION from system repositories..."

    # Try system install first (Debian 13 / Ubuntu 24 often have these)
    if $SUDO apt install -y "llvm-$LLVM_VERSION" "clang-$LLVM_VERSION" "libclang-$LLVM_VERSION-dev" "libomp-$LLVM_VERSION-dev" "lld-$LLVM_VERSION" 2>/dev/null; then
        log INFO "Successfully installed LLVM/Clang from system repositories."
    else
        log INFO "System repository install failed or incomplete. Falling back to LLVM script."
        log INFO "Downloading and installing LLVM/Clang via llvm.sh..."

        wget -q https://apt.llvm.org/llvm.sh || { log ERROR "Failed to download LLVM setup script."; exit 1; }
        chmod +x llvm.sh

        log INFO "Installing LLVM/Clang version $LLVM_VERSION..."

        # Note: On Debian Testing/Unstable, llvm.sh might warn about codenames.
        $SUDO ./llvm.sh "$LLVM_VERSION"

        # Cleanup
        rm -f llvm.sh
    fi
else
    LLVM_VERSION=$FOUND_LLVM_VERSION
    log INFO "Found LLVM/Clang version $FOUND_LLVM_VERSION. Skipping LLVM/Clang installation."
fi

# Install required packages for acpp (ensure dev headers are present)
$SUDO apt install -y \
    "libclang-${LLVM_VERSION}-dev" "clang-tools-${LLVM_VERSION}" \
    "libomp-${LLVM_VERSION}-dev" "llvm-${LLVM_VERSION}-dev" "lld-${LLVM_VERSION}"

# Get LLVM/Clang paths
LLVM_DIR=$(llvm-config-${LLVM_VERSION} --cmakedir)
log INFO "LLVM cmake path is $LLVM_DIR"
CLANG_EXECUTABLE_PATH=$(which clang++-${LLVM_VERSION})
log INFO "Clang++ executable path is $CLANG_EXECUTABLE_PATH"

# Install libstdc++ for Clang
# On some Debian systems, the gcc version parsing might be tricky, fail gracefully
GCC_VERSION=$(clang++-${LLVM_VERSION} -v 2>&1 | awk -F/ '/Selected GCC/ {print $NF}' || echo "")
if [[ -n "$GCC_VERSION" ]]; then
    log INFO "Installing libstdc++ version $GCC_VERSION..."
    $SUDO apt install -y "libstdc++-${GCC_VERSION}-dev" || log WARN "Could not explicitly install libstdc++-${GCC_VERSION}-dev, assuming standard libstdc++-dev is sufficient."
else
    $SUDO apt install -y libstdc++-dev
fi

# Clone and build AdaptiveCpp
if [[ ! -d "AdaptiveCpp" ]]; then
    log INFO "Cloning AdaptiveCpp repository..."
    git clone --depth 1 --branch $ADAPTIVE_CPP_SOURCE_VERSION "$ADAPTIVE_CPP_URL"
else
    log WARN "AdaptiveCpp directory already exists. Skipping download."
fi


cd AdaptiveCpp
mkdir -p build && cd build
log INFO "Configuring build with CMake..."
CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DLLVM_DIR=$LLVM_DIR -DCLANG_EXECUTABLE_PATH=$CLANG_EXECUTABLE_PATH"

if [[ -n "$CUDA_ROOT" ]]; then
    log INFO "CUDA root specified: $CUDA_ROOT. Enabling CUDA backend..."
    CMAKE_FLAGS="$CMAKE_FLAGS -DWITH_CUDA_BACKEND=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT"
else
    log INFO "Building with defaults."
fi
CXX=$CLANG_EXECUTABLE_PATH cmake $CMAKE_FLAGS ..
log INFO "Building and installing AdaptiveCpp to $INSTALL_PREFIX..."
$SUDO make install -j$(nproc)

# Back to main pwd
cd ../..

# Verify AdaptiveCpp installation
if ACPP_VERSION=$(acpp --acpp-version | grep 'AdaptiveCpp version:' | awk '{print $3}'); then
    log INFO "\033[1;32mAdaptiveCpp version $ACPP_VERSION has been installed to $INSTALL_PREFIX.\033[0m"
else
    log ERROR "Failed to verify AdaptiveCpp installation."
fi

# Install kompass core dependencies
log INFO "Installing kompass-core dependencies..."

# Determine source directory
KOMPASS_SRC_DIR=""

# Check if we are already inside the repo
if [[ -f "pyproject.toml" ]] && grep -q "kompass_core" "pyproject.toml"; then
    log INFO "Found pyproject.toml in current directory. Using current directory as source."
    KOMPASS_SRC_DIR="."

# Check if folder exist explicitly
elif [[ -d "kompass-core" ]]; then
    log INFO "kompass-core directory found. Using existing directory."
    KOMPASS_SRC_DIR="kompass-core"
else
    # Clone if not found
    if [ "${NIGHTLY}" = "true" ]; then
        log INFO "NIGHTLY build enabled, cloning kompass-core from main..."
        git clone --depth 1 --branch main "$KOMPASS_CORE_URL"
    else
        LATEST_TAG=$(curl -s "https://api.github.com/repos/$KOMPASS_CORE_REPO/tags" | jq -r '.[0].name')
        log INFO "Cloning kompass-core repository at tag $LATEST_TAG..."
        git clone --depth 1 --branch "$LATEST_TAG" "$KOMPASS_CORE_URL"
    fi
    KOMPASS_SRC_DIR="kompass-core"
fi

cd "$KOMPASS_SRC_DIR"

# Check if legacy system (Ubuntu <= 20.04) for CONAN usage
if is_legacy_system; then
    log WARN "Legacy system detected. Installing dependencies via Conan..."

    # Install python build tools
    log INFO "Installing Conan and Ninja..."
    export PIP_BREAK_SYSTEM_PACKAGES=1
    python3 -m pip install conan ninja "cmake<3.30"

    # Configure Conan Profile
    log INFO "Configuring Conan profile..."
    conan profile detect --force
    # Force C++17 (Required for OMPL)
    sed -i 's/compiler.cppstd=.*/compiler.cppstd=gnu17/g' ~/.conan2/profiles/default

    # Ensure the whole graph is built as static libs
    log INFO "Forcing static libs in Conan profile..."
    cat >> ~/.conan2/profiles/default <<EOF
[options]
*:shared=False
*:fPIC=True
EOF

    # TODO: This fork should be removed when recipe merged upstream
    # Clone Fork (to get the ompl recipe)
    log INFO "Cloning Conan recipes fork..."
    CONAN_RECIPES_DIR="/tmp/conan_recipes"
    rm -rf "$CONAN_RECIPES_DIR"
    git clone --depth 1 https://github.com/aleph-ra/conan-center-index.git "$CONAN_RECIPES_DIR"

    # Clean previous artifacts to ensure a fresh build
    log INFO "Cleaning previous Conan artifacts..."
    conan remove "ompl/*" -c || true
    conan remove "fcl/*" -c || true
    rm -rf build
    rm -rf conan_build

    # Build recipes
    log INFO "Building FCL and OMPL packages from source..."

    # Build FCL 0.7.0
    conan create "$CONAN_RECIPES_DIR/recipes/fcl/all" --version=0.7.0 \
        --build=missing

    # Build OMPL 1.7.0 (With optimization flags)
    conan create "$CONAN_RECIPES_DIR/recipes/ompl/all" --version=1.7.0 \
        --build=missing --build="b2/*" \
        -c tools.build:jobs=$(nproc)

    # Install Dependencies into local project folder
    log INFO "Generating Conan toolchains..."
    CONAN_BUILD_DIR="$PWD/conan_build"
    mkdir -p "$CONAN_BUILD_DIR"

    conan install \
        --requires="ompl/1.7.0" \
        --requires="fcl/0.7.0" \
        -g CMakeDeps \
        -g CMakeToolchain \
        -g VirtualRunEnv \
        --output-folder="$CONAN_BUILD_DIR" \
        --build=missing \
        --build="b2/*" \
        -c "tools.cmake.cmaketoolchain:extra_variables={'CMAKE_POLICY_VERSION_MINIMUM':'3.5'}"

    # Verification of static libs
    log INFO "Verifying OMPL Build Artifacts..."
    if [ -z "$(find ~/.conan2 -name 'libompl.a')" ]; then
        log ERROR "libompl.a NOT found. Build may have produced shared libraries."
        find ~/.conan2 -name "libompl.so*"
        exit 1
    else
        log INFO "SUCCESS: libompl.a found."
    fi

    # Set SKBUILD_CMAKE_ARGS env var which scikit-build-core picks up
    export SKBUILD_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$CONAN_BUILD_DIR/conan_toolchain.cmake"

else
    log INFO "Modern system (Debian 11+/Ubuntu 21+/Raspbian) detected."
    log INFO "Installing ompl and fcl from system apt repositories..."
    $SUDO apt install -y libompl-dev libfcl-dev libode-dev \
                libboost-serialization-dev \
                libboost-filesystem-dev \
                libboost-system-dev
fi

log INFO "Installing kompass-core with pip"
export PIP_BREAK_SYSTEM_PACKAGES=1
python3 -m pip install pip-tools
python3 -m pip uninstall -y kompass-core  # uninstall any previous versions

# Build and Install
# SKBUILD_CMAKE_ARGS is already set if we are in legacy mode
CXX=$CLANG_EXECUTABLE_PATH python3 -m pip install .

# Clean up source files if not required
if [[ $KEEP_SOURCE_FILES == false ]]; then
    log INFO "Removing source files (AdaptiveCpp, kompass-core) as --keep-source-files is not set."
    # We are currently inside kompass-core, so move up one level to delete it
    cd ..
    rm -rf AdaptiveCpp
    rm -rf kompass-core
fi

# Clean up temporary Conan recipes
if [[ -d "/tmp/conan_recipes" ]]; then
    log INFO "Removing temporary Conan recipes..."
    rm -rf /tmp/conan_recipes
fi

# Verify installation
if python3 -c "import kompass_cpp" 2>/dev/null; then
    log INFO "\033[1;32mkompass-core was installed successfully.\033[0m"
else
    log ERROR "There was an error while installing kompass-core. Please refer to the git repo and open an issue."
fi
