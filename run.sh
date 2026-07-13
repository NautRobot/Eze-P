#!/bin/bash
set -euo pipefail

###############################################################################
# USAGE
###############################################################################

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 program.cpp [program arguments...]"
    exit 1
fi

SOURCE_FILE="$(realpath "$1")"
shift

if [[ ! -f "$SOURCE_FILE" ]]; then
    echo "ERROR: Source file not found:"
    echo "  $SOURCE_FILE"
    exit 1
fi

if [[ "$SOURCE_FILE" != *.cpp ]]; then
    echo "ERROR: Input must be a .cpp file."
    exit 1
fi

###############################################################################
# PATHS AND SETTINGS
###############################################################################

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$WORKSPACE_DIR/install"

APP_NAME="$(basename "${SOURCE_FILE%.cpp}")"
RUN_BUILD_DIR="$WORKSPACE_DIR/run-build/$APP_NAME"
GENERATED_SOURCE_DIR="$RUN_BUILD_DIR/cmake-source"
NUMA_CONFIG_DIR="$RUN_BUILD_DIR/numa-package"

NODES="${NODES:-2}"
TASKS_PER_NODE="${TASKS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-7}"
TOTAL_TASKS=$((NODES * TASKS_PER_NODE))
WALLTIME="${WALLTIME:-00:15:00}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"

###############################################################################
# OBTAIN A SLURM ALLOCATION IF NECESSARY
#
# Inside an existing allocation:
#   ./run.sh program.cpp
#
# Outside an allocation:
#   ACCOUNT=csc607 ./run.sh program.cpp
###############################################################################

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${ACCOUNT:-}" ]]; then
        echo "ERROR: No active Slurm allocation."
        echo
        echo "Request an allocation first:"
        echo "  salloc -A <project> -N$NODES -t $WALLTIME"
        echo "  ./run.sh $SOURCE_FILE"
        echo
        echo "Or allow this script to request it:"
        echo "  ACCOUNT=<project> ./run.sh $SOURCE_FILE"
        exit 1
    fi

    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

    exec salloc \
        -A "$ACCOUNT" \
        -N "$NODES" \
        -t "$WALLTIME" \
        "$SCRIPT_PATH" "$SOURCE_FILE" "$@"
fi

###############################################################################
# FRONTIER MODULE ENVIRONMENT
###############################################################################

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    else
        echo "ERROR: Frontier's module environment is unavailable."
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
module load cmake

###############################################################################
# VERIFY REQUIRED TOOLS AND ROCSHMEM
###############################################################################

for required_command in cmake hipcc hipconfig CC srun; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $required_command"
        exit 1
    fi
done

if [[ ! -d "$INSTALL_DIR/include" ]]; then
    echo "ERROR: rocSHMEM installation was not found:"
    echo "  $INSTALL_DIR"
    echo
    echo "Run ./build.sh first."
    exit 1
fi

if [[ ! -d "$INSTALL_DIR/lib/cmake/rocshmem" ]] &&
   [[ ! -d "$INSTALL_DIR/lib64/cmake/rocshmem" ]]; then
    echo "ERROR: The rocSHMEM CMake package was not found under:"
    echo "  $INSTALL_DIR"
    echo
    echo "Run ./build.sh first."
    exit 1
fi

ROCM_PATH="${ROCM_PATH:-$(hipconfig --path)}"
HIPCC="$(command -v hipcc)"
MPICXX="$(command -v CC)"

###############################################################################
# ROCSHMEM RUNTIME ENVIRONMENT
###############################################################################

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:${LD_LIBRARY_PATH:-}"

# Enable GPU support in Cray MPICH.
export MPICH_GPU_SUPPORT_ENABLED=1

# RO supplies cross-node communication.
export ROCSHMEM_BACKEND=ro

# Permit IPC for communication between local GPUs.
export ROCSHMEM_DISABLE_MIXED_IPC=0

###############################################################################
# PREPARE APPLICATION BUILD DIRECTORY
###############################################################################

rm -rf "$RUN_BUILD_DIR"

mkdir -p \
    "$GENERATED_SOURCE_DIR" \
    "$NUMA_CONFIG_DIR"

###############################################################################
# ROCSHMEM NUMA PACKAGE WORKAROUND
#
# The installed rocSHMEM package declares NUMA as a CMake dependency even when
# rocSHMEM was built with GDA disabled. Frontier supplies libnuma but does not
# provide NUMAConfig.cmake, so provide the missing CMake package information.
###############################################################################

cat > "$NUMA_CONFIG_DIR/NUMAConfig.cmake" <<'NUMA_EOF'
include_guard(GLOBAL)

find_path(
    NUMA_INCLUDE_DIR
    NAMES numa.h
    PATHS
        /usr/include
        /usr/local/include
)

find_library(
    NUMA_LIBRARY
    NAMES numa
    PATHS
        /usr/lib64
        /usr/lib
        /usr/local/lib64
        /usr/local/lib
)

if(NOT TARGET numa::numa)
    if(NUMA_LIBRARY)
        add_library(numa::numa UNKNOWN IMPORTED)

        set_target_properties(
            numa::numa
            PROPERTIES
                IMPORTED_LOCATION "${NUMA_LIBRARY}"
        )

        if(NUMA_INCLUDE_DIR)
            set_target_properties(
                numa::numa
                PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
            )
        endif()
    else()
        # GDA is disabled, so this target is not used by the application.
        add_library(numa::numa INTERFACE IMPORTED)
    endif()
endif()

if(NOT TARGET NUMA::NUMA)
    add_library(NUMA::NUMA INTERFACE IMPORTED)

    set_target_properties(
        NUMA::NUMA
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "numa::numa"
    )
endif()

set(NUMA_FOUND TRUE)
NUMA_EOF

###############################################################################
# GENERATE APPLICATION CMAKE PROJECT
###############################################################################

cat > "$GENERATED_SOURCE_DIR/CMakeLists.txt" <<'CMAKE_EOF'
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(rocshmem_single_program LANGUAGES CXX)

if(NOT DEFINED APP_SOURCE)
    message(FATAL_ERROR "APP_SOURCE was not provided")
endif()

if(NOT EXISTS "${APP_SOURCE}")
    message(FATAL_ERROR "Source file does not exist: ${APP_SOURCE}")
endif()

if(NOT DEFINED APP_OUTPUT_NAME)
    set(APP_OUTPUT_NAME rocshmem_app)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(hip REQUIRED)
find_package(rocshmem REQUIRED CONFIG)

add_executable(rocshmem_app "${APP_SOURCE}")

set_target_properties(
    rocshmem_app
    PROPERTIES
        OUTPUT_NAME "${APP_OUTPUT_NAME}"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

target_compile_options(
    rocshmem_app
    PRIVATE
        --offload-arch=gfx90a
        -fgpu-rdc
)

target_link_options(
    rocshmem_app
    PRIVATE
        --offload-arch=gfx90a
        -fgpu-rdc
)

target_link_libraries(
    rocshmem_app
    PRIVATE
        MPI::MPI_CXX
        roc::rocshmem
)
CMAKE_EOF

###############################################################################
# CONFIGURE APPLICATION
###############################################################################

echo
echo "Source:          $SOURCE_FILE"
echo "Application:     $APP_NAME"
echo "ROCm:            $ROCM_PATH"
echo "rocSHMEM:        $INSTALL_DIR"
echo "Nodes:           $NODES"
echo "Tasks per node:  $TASKS_PER_NODE"
echo "Total tasks:     $TOTAL_TASKS"
echo

cmake \
    -S "$GENERATED_SOURCE_DIR" \
    -B "$RUN_BUILD_DIR/build" \
    -DAPP_SOURCE="$SOURCE_FILE" \
    -DAPP_OUTPUT_NAME="$APP_NAME" \
    -DNUMA_DIR="$NUMA_CONFIG_DIR" \
    -DCMAKE_CXX_COMPILER="$HIPCC" \
    -DMPI_CXX_COMPILER="$MPICXX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR;$ROCM_PATH" \
    -DCMAKE_BUILD_RPATH="$INSTALL_DIR/lib;$INSTALL_DIR/lib64"

###############################################################################
# BUILD APPLICATION
###############################################################################

cmake \
    --build "$RUN_BUILD_DIR/build" \
    --parallel "$PARALLEL_JOBS"

EXECUTABLE="$RUN_BUILD_DIR/build/$APP_NAME"

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: Application executable was not created:"
    echo "  $EXECUTABLE"
    exit 1
fi

###############################################################################
# RUN APPLICATION
###############################################################################

echo
echo "Launching $EXECUTABLE"
echo "Backend: RO with local IPC enabled"
echo

srun \
    -N "$NODES" \
    -n "$TOTAL_TASKS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    -c "$CPUS_PER_TASK" \
    --gpus-per-task="$GPUS_PER_TASK" \
    --gpu-bind=closest \
    "$EXECUTABLE" "$@"