#!/bin/bash
set -xe

# TODO: This fork should be removed when recipe merged upstream
# Clone Fork (to get the ompl recipe)
mkdir -p /tmp/conan_recipes
git clone --depth 1 https://github.com/aleph-ra/conan-center-index.git /tmp/conan_recipes

# Install System Dependencies
yum install -y git cmake gcc-c++ patch

# Install Python Tools
# Using specific python path to ensure pip exists on manylinux
export PATH=/opt/python/cp311-cp311/bin:$PATH
# pin cmake to <3.30 because libccd build fails otherwise
pip install conan ninja "cmake<3.30"

# Configure Conan Profile
conan profile detect --force
# Force C++17 (Required for OMPL)
sed -i 's/compiler.cppstd=.*/compiler.cppstd=gnu17/g' ~/.conan2/profiles/default

# Ensure the whole graph is build as static libs
cat >> ~/.conan2/profiles/default <<EOF
[options]
*:shared=False
*:fPIC=True
EOF

# 4. Clean previous artifacts to ensure a fresh build
conan remove "ompl/*" -c || true
conan remove "fcl/*" -c || true
rm -rf /project/build
rm -rf /project/conan_build

# Build FCL 0.7.0
conan create /tmp/conan_recipes/recipes/fcl/all --version=0.7.0 \
    --build=missing \

# Build OMPL 1.7.0 (With optimization flags)
conan create /tmp/conan_recipes/recipes/ompl/all --version=1.7.0 \
    --build=missing --build="b2/*" \
    -c tools.build:jobs=1 \

# Install Dependencies
mkdir -p /project/conan_build
cd /project
conan install \
    --requires="ompl/1.7.0" \
    --requires="fcl/0.7.0" \
    -g CMakeDeps \
    -g CMakeToolchain \
    --output-folder=/project/conan_build \
    --build=missing \
    --build="b2/*" \
    -c "tools.cmake.cmaketoolchain:extra_variables={'CMAKE_POLICY_VERSION_MINIMUM':'3.5'}"

# Verify
echo "🔍 Verifying OMPL Build Artifacts..."
if [ -z "$(find /root/.conan2 -name 'libompl.a')" ]; then
    echo "❌ ERROR: libompl.a NOT found."
    find /root/.conan2 -name "libompl.so*"
    exit 1
else
    echo "✅ SUCCESS: libompl.a found."
fi
