#!/bin/bash
set -xe

# =============================================================================
# repair_gpu_wheel.sh
# Runs as cibuildwheel 'repair-wheel-command'.
# Injects the deployed AdaptiveCpp runtime libs into the wheel and fixes
# RPATHs so the bundled shared libraries are found at import time.
#
# Usage: repair_gpu_wheel.sh <wheel_path> <dest_dir>
# =============================================================================

WHEEL="$(readlink -f "$1")"
DEST_DIR="$(mkdir -p "$2" && readlink -f "$2")"
ACPP_DEPLOY_DIR="/tmp/acpp_deploy"

if [ ! -f "$WHEEL" ]; then
    echo "ERROR: Wheel not found: $WHEEL"
    exit 1
fi

if [ ! -d "$ACPP_DEPLOY_DIR" ]; then
    echo "ERROR: acpp deploy directory not found: $ACPP_DEPLOY_DIR"
    exit 1
fi

# Create temp directory for unpacking
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Unpacking wheel: $WHEEL"
unzip -q "$WHEEL" -d "$TMPDIR"

# -----------------------------------------------------------------------------
# Inject acpp deployed libs
# The deploy directory contains: libacpp-rt.so, libacpp-common.so,
# backend plugins (hipSYCL/), LLVM-to-backend translators, bitcode libs, etc.
# -----------------------------------------------------------------------------
LIBS_DIR="$TMPDIR/kompass_core.libs"
mkdir -p "$LIBS_DIR"

echo "Copying acpp runtime from $ACPP_DEPLOY_DIR..."
cp -rL "$ACPP_DEPLOY_DIR"/* "$LIBS_DIR/"

# -----------------------------------------------------------------------------
# Fix RPATHs on extension modules
# The nanobind .so modules need to find acpp libs in the adjacent libs dir
# -----------------------------------------------------------------------------
echo "Fixing RPATHs on extension modules..."
for ext in "$TMPDIR"/*.so; do
    [ -f "$ext" ] || continue
    echo "  Patching: $(basename "$ext")"
    patchelf --set-rpath '$ORIGIN/kompass_core.libs' "$ext"
done

# -----------------------------------------------------------------------------
# Fix RPATHs on acpp libs
# The acpp shared libs need to find each other and their subdirectories
# (hipSYCL/ for backend plugins, llvm-to-backend/ for JIT translators)
# -----------------------------------------------------------------------------
echo "Fixing RPATHs on acpp libraries..."
find "$LIBS_DIR" -name '*.so*' -type f | while read lib; do
    echo "  Patching: $(basename "$lib")"
    patchelf --set-rpath '$ORIGIN:$ORIGIN/hipSYCL:$ORIGIN/llvm-to-backend' "$lib" 2>/dev/null || true
done

# Fix RPATHs in hipSYCL/ (backend plugins need to find libs in parent and llvm-to-backend/)
if [ -d "$LIBS_DIR/hipSYCL" ]; then
    find "$LIBS_DIR/hipSYCL" -maxdepth 1 -name '*.so*' -type f | while read lib; do
        echo "  Patching (hipSYCL): $(basename "$lib")"
        patchelf --set-rpath '$ORIGIN:$ORIGIN/..:$ORIGIN/llvm-to-backend' "$lib" 2>/dev/null || true
    done
fi

# Fix RPATHs in hipSYCL/llvm-to-backend/ (need to find libs in parent and grandparent)
if [ -d "$LIBS_DIR/hipSYCL/llvm-to-backend" ]; then
    find "$LIBS_DIR/hipSYCL/llvm-to-backend" -name '*.so*' -type f | while read lib; do
        echo "  Patching (llvm-to-backend): $(basename "$lib")"
        patchelf --set-rpath '$ORIGIN:$ORIGIN/..:$ORIGIN/../..' "$lib" 2>/dev/null || true
    done
fi

# -----------------------------------------------------------------------------
# Repack the wheel
# -----------------------------------------------------------------------------
WHEEL_NAME=$(basename "$WHEEL")
mkdir -p "$DEST_DIR"

echo "Repacking wheel: $WHEEL_NAME"
cd "$TMPDIR"
zip -q -r "$DEST_DIR/$WHEEL_NAME" .

echo "Repaired wheel: $DEST_DIR/$WHEEL_NAME"
echo "Bundled libs size: $(du -sh "$LIBS_DIR" | cut -f1)"
