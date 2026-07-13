#!/bin/bash
set -euo pipefail

###############################################################################
# PATHS AND SETTINGS
###############################################################################

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$WORKSPACE_DIR/src"
BUILD_DIR="$WORKSPACE_DIR/build"
INSTALL_DIR="$WORKSPACE_DIR/install"

ROCM_SYSTEMS_DIR="$SRC_DIR/rocm-systems"
ROCSHMEM_SOURCE_DIR="$ROCM_SYSTEMS_DIR/projects/rocshmem"

ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"

###############################################################################
# FRONTIER MODULE ENVIRONMENT
###############################################################################

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    else
        echo "ERROR: Frontier's module command is unavailable."
        exit 1
    fi
fi

module reset
module load cpe/26.03
module swap PrgEnv-cray PrgEnv-gnu
module load gcc-native/14.2
module load rocm/7.2.0
module load craype-accel-amd-gfx90a
module load cray-mpich/9.1.0
module load cray-python/3.12.12
module load cmake

echo "Loaded modules:"
module -t list

###############################################################################
# VERIFY REQUIRED TOOLS
###############################################################################

for required_command in git cmake hipcc hipconfig CC; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: $required_command was not found after loading modules."
        exit 1
    fi
done

ROCM_PATH="${ROCM_PATH:-$(hipconfig --path)}"
HIPCC="$(command -v hipcc)"
MPICXX="$(command -v CC)"

export MPICH_GPU_SUPPORT_ENABLED=1

echo
echo "ROCm path:         $ROCM_PATH"
echo "ROCm version:      $ROCM_VERSION"
echo "HIP compiler:      $HIPCC"
echo "MPI C++ wrapper:   $MPICXX"
echo "rocSHMEM source:   $ROCSHMEM_SOURCE_DIR"
echo "Build directory:   $BUILD_DIR"
echo "Installation:      $INSTALL_DIR"
echo

###############################################################################
# CLONE ROCSHMEM SOURCE
###############################################################################

mkdir -p "$SRC_DIR"

if [[ ! -d "$ROCM_SYSTEMS_DIR/.git" ]]; then
    if [[ -e "$ROCM_SYSTEMS_DIR" ]] &&
       [[ -n "$(ls -A "$ROCM_SYSTEMS_DIR" 2>/dev/null)" ]]; then
        echo "ERROR: $ROCM_SYSTEMS_DIR exists but is not a Git repository."
        exit 1
    fi

    git clone \
        --no-checkout \
        --filter=blob:none \
        https://github.com/ROCm/rocm-systems \
        "$ROCM_SYSTEMS_DIR"

    git -C "$ROCM_SYSTEMS_DIR" sparse-checkout set \
        --cone projects/rocshmem

    git -C "$ROCM_SYSTEMS_DIR" checkout develop
else
    echo "Using existing rocm-systems checkout."

    git -C "$ROCM_SYSTEMS_DIR" sparse-checkout set \
        --cone projects/rocshmem

    git -C "$ROCM_SYSTEMS_DIR" checkout develop
    git -C "$ROCM_SYSTEMS_DIR" pull --ff-only origin develop
fi

if [[ ! -f "$ROCSHMEM_SOURCE_DIR/CMakeLists.txt" ]]; then
    echo "ERROR: rocSHMEM source checkout is incomplete."
    exit 1
fi

###############################################################################
# PREPARE BUILD AND INSTALL DIRECTORIES
###############################################################################

if [[ -d "$BUILD_DIR" ]]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

###############################################################################
# CONFIGURE ROCSHMEM
#
# RO  = cross-node communication through MPI
# IPC = direct same-node GPU communication
# GDA = disabled because Frontier uses Slingshot/CXI
###############################################################################

cmake \
    -S "$ROCSHMEM_SOURCE_DIR" \
    -B "$BUILD_DIR" \
    -DROCM_PATH="$ROCM_PATH" \
    -DEXPLICIT_ROCM_VERSION="$ROCM_VERSION" \
    -DCMAKE_CXX_COMPILER="$HIPCC" \
    -DMPI_CXX_COMPILER="$MPICXX" \
    -DMPIEXEC_EXECUTABLE="$(command -v srun)" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DGPU_TARGETS=gfx90a \
    -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF \
    -DBUILD_CODE_COVERAGE=OFF \
    -DBUILD_FUNCTIONAL_TESTS=ON \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_PYTHON_TESTS=ON \
    -DBUILD_CTESTS=ON \
    -DUSE_EXTERNAL_MPI=ON \
    -DPROFILE=OFF \
    -DUSE_GDA=OFF \
    -DGDA_MLX5=OFF \
    -DGDA_BNXT=OFF \
    -DGDA_IONIC=OFF \
    -DUSE_RO=ON \
    -DUSE_IPC=ON \
    -DUSE_SINGLE_NODE=OFF \
    -DUSE_THREADS=OFF \
    -DUSE_WF_COAL=OFF \
    -DUSE_HDP_FLUSH=OFF \
    -DUSE_HDP_FLUSH_HOST_SIDE=OFF \
    -DASAN="${ASAN:-OFF}" \
    "$@"

###############################################################################
# BUILD AND INSTALL ROCSHMEM
###############################################################################

cmake --build "$BUILD_DIR" --parallel "$PARALLEL_JOBS"
cmake --install "$BUILD_DIR"

###############################################################################
# RESULTS
###############################################################################

echo
echo "rocSHMEM build completed."
echo
echo "Source:"
echo "  $ROCSHMEM_SOURCE_DIR"
echo
echo "Build:"
echo "  $BUILD_DIR"
echo
echo "Installation:"
echo "  $INSTALL_DIR"
echo
echo "Backend configuration:"
echo "  RO:  enabled for cross-node communication"
echo "  IPC: enabled for same-node communication"
echo "  GDA: disabled"