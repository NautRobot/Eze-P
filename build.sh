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

# CPE 26.03's Cray MPICH runtime requires the ROCm 7 HIP ABI.
ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
BUILD_EXAMPLES="${BUILD_EXAMPLES:-ON}"

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
module load "rocm/$ROCM_VERSION"
module load craype-accel-amd-gfx90a
module load cray-mpich/9.1.0
module load cray-python/3.12.12
module load cmake

# The default Darshan interposer is detected as MPI_CXX by CMake on this CPE
# and can introduce a different HIP ABI. rocSHMEM must link directly to MPICH.
module unload darshan-runtime >/dev/null 2>&1 || true

if module -t list 2>&1 | grep -q '^darshan-runtime/'; then
    echo "ERROR: darshan-runtime is still loaded and would contaminate MPI detection."
    exit 1
fi

echo "Loaded modules:"
module -t list

###############################################################################
# VERIFY REQUIRED TOOLS AND ROCM VERSION
###############################################################################

for required_command in cmake hipcc hipconfig CC ldd; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: $required_command was not found after loading modules."
        exit 1
    fi
done

ROCM_PATH="${ROCM_PATH:-$(hipconfig --path)}"
ROCM_PATH="$(readlink -f "$ROCM_PATH")"
EXPECTED_ROCM_PATH="$(readlink -f "/opt/rocm-$ROCM_VERSION")"

if [[ "$ROCM_PATH" != "$EXPECTED_ROCM_PATH" ]]; then
    echo "ERROR: the loaded ROCm path does not match the requested version."
    echo "  Requested: $ROCM_VERSION"
    echo "  Expected:  $EXPECTED_ROCM_PATH"
    echo "  Loaded:    $ROCM_PATH"
    exit 1
fi

HIPCC="$(command -v hipcc)"
MPICXX="$(command -v CC)"

export ROCM_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

echo
echo "ROCm path:         $ROCM_PATH"
echo "ROCm version:      $ROCM_VERSION"
echo "HIP compiler:      $HIPCC"
echo "MPI C++ wrapper:   $MPICXX"
echo "rocSHMEM source:   $ROCSHMEM_SOURCE_DIR"
echo "Build directory:   $BUILD_DIR"
echo "Installation:      $INSTALL_DIR"
echo "Build examples:    $BUILD_EXAMPLES"
echo

###############################################################################
# VERIFY THE VENDORED ROCSHMEM SOURCE
###############################################################################

if [[ ! -f "$ROCSHMEM_SOURCE_DIR/CMakeLists.txt" ]]; then
    echo "ERROR: the vendored rocSHMEM source is missing or incomplete:"
    echo "  $ROCSHMEM_SOURCE_DIR"
    echo "This build script never clones, pulls, checks out, stages, or commits Git content."
    exit 1
fi

if [[ -e "$ROCM_SYSTEMS_DIR/.git" ]]; then
    echo "ERROR: $ROCM_SYSTEMS_DIR is still a nested Git repository."
    echo "Remove its nested .git metadata before committing it in the outer repository."
    exit 1
fi

echo "Using vendored rocSHMEM source; no Git operations will be performed."

###############################################################################
# PREPARE COMPLETELY CLEAN BUILD AND INSTALL DIRECTORIES
###############################################################################

echo "Removing old build and installation artifacts."
rm -rf "$BUILD_DIR" "$INSTALL_DIR"
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
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
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
# VERIFY THE INSTALLED RUNTIME ABI
###############################################################################

UNIT_TEST_EXE="$INSTALL_DIR/share/rocshmem/rocshmem_unit_tests"

if [[ ! -x "$UNIT_TEST_EXE" ]]; then
    echo "ERROR: the installed unit-test executable was not found:"
    echo "  $UNIT_TEST_EXE"
    exit 1
fi

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

LDD_OUTPUT="$(ldd "$UNIT_TEST_EXE")"
echo
echo "Installed HIP runtime dependency:"
printf '%s\n' "$LDD_OUTPUT" | grep 'libamdhip64' || true

if printf '%s\n' "$LDD_OUTPUT" | grep -q 'not found'; then
    echo "ERROR: the installed rocSHMEM executable has unresolved libraries:"
    printf '%s\n' "$LDD_OUTPUT" | grep 'not found'
    exit 1
fi

ROCM_MAJOR="${ROCM_VERSION%%.*}"
EXPECTED_HIP_SONAME="libamdhip64.so.$ROCM_MAJOR"
HIP_SONAMES="$(printf '%s\n' "$LDD_OUTPUT" | grep -oE 'libamdhip64\.so\.[0-9]+' | sort -u)"

if [[ "$HIP_SONAMES" != "$EXPECTED_HIP_SONAME" ]]; then
    echo "ERROR: inconsistent HIP runtime ABIs were linked."
    echo "  Expected only: $EXPECTED_HIP_SONAME"
    echo "  Found:"
    printf '    %s\n' $HIP_SONAMES
    echo "Use a ROCm version matching the Cray MPICH GTL, or use an MPI stack"
    echo "that was built against ROCm $ROCM_MAJOR."
    exit 1
fi

###############################################################################
# RESULTS
###############################################################################

echo
echo "rocSHMEM build and ABI verification completed."
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
