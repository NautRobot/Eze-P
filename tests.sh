#!/bin/bash
set -euo pipefail

###############################################################################
# PATHS AND SETTINGS
###############################################################################

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$WORKSPACE_DIR/install"

UNIT_TEST_DIR="$INSTALL_DIR/bin/rocshmem/tests/unit"
FUNCTIONAL_TEST_DIR="$INSTALL_DIR/bin/rocshmem/tests/functional"

UNIT_TEST_BINARY="$INSTALL_DIR/share/rocshmem/rocshmem_unit_tests"
FUNCTIONAL_TEST_BINARY="$INSTALL_DIR/share/rocshmem/rocshmem_functional_tests"

NODES="${NODES:-2}"
WALLTIME="${WALLTIME:-04:00:00}"
TEST_TIMEOUT="${TEST_TIMEOUT:-600}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$WORKSPACE_DIR/test-logs/$TIMESTAMP"
HELPER_DIR="$WORKSPACE_DIR/test-helpers"
MPI_LAUNCHER="$HELPER_DIR/frontier-mpiexec"

###############################################################################
# OBTAIN A SLURM ALLOCATION
###############################################################################

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${ACCOUNT:-}" ]]; then
        echo "ERROR: No active Slurm allocation."
        echo
        echo "Run:"
        echo "  ACCOUNT=<project> ./tests.sh"
        exit 1
    fi

    SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

    exec salloc \
        -A "$ACCOUNT" \
        -N "$NODES" \
        -t "$WALLTIME" \
        "$SCRIPT_PATH"
fi

###############################################################################
# FRONTIER MODULE ENVIRONMENT
###############################################################################

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    else
        echo "ERROR: Frontier module environment is unavailable."
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

###############################################################################
# VERIFY REQUIRED COMMANDS
###############################################################################

for required_command in \
    ctest \
    hipconfig \
    ldd \
    srun; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "ERROR: Required command was not found: $required_command"
        exit 1
    fi
done

ROCM_PATH="${ROCM_PATH:-$(hipconfig --path)}"

###############################################################################
# VERIFY ROCSHMEM INSTALLATION
###############################################################################

if [[ ! -f "$UNIT_TEST_DIR/CTestTestfile.cmake" ]]; then
    echo "ERROR: Installed unit tests were not found:"
    echo "  $UNIT_TEST_DIR"
    echo
    echo "Run ./build.sh first."
    exit 1
fi

if [[ ! -f "$FUNCTIONAL_TEST_DIR/CTestTestfile.cmake" ]]; then
    echo "ERROR: Installed functional tests were not found:"
    echo "  $FUNCTIONAL_TEST_DIR"
    echo
    echo "Run ./build.sh first."
    exit 1
fi

if [[ ! -x "$UNIT_TEST_BINARY" ]]; then
    echo "ERROR: Unit-test executable was not found:"
    echo "  $UNIT_TEST_BINARY"
    exit 1
fi

if [[ ! -x "$FUNCTIONAL_TEST_BINARY" ]]; then
    echo "ERROR: Functional-test executable was not found:"
    echo "  $FUNCTIONAL_TEST_BINARY"
    exit 1
fi

###############################################################################
# RUNTIME ENVIRONMENT
###############################################################################

export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_PATH/lib:$ROCM_PATH/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

export MPICH_GPU_SUPPORT_ENABLED=1

export ROCSHMEM_BACKEND=ro
export ROCSHMEM_BACKEND_TYPE=ro
export ROCSHMEM_DISABLE_MIXED_IPC=0

# Each test creates its own srun step.
export CTEST_PARALLEL_LEVEL=1

mkdir -p "$LOG_DIR" "$HELPER_DIR"

###############################################################################
# CHECK DYNAMIC LIBRARIES BEFORE LAUNCHING 584 TESTS
###############################################################################

check_libraries() {
    local executable="$1"
    local output

    output="$(ldd "$executable" 2>&1)"

    if grep -q "not found" <<< "$output"; then
        echo "ERROR: Missing runtime libraries for:"
        echo "  $executable"
        echo
        echo "$output" | grep -E "not found|amdhip|hsa|mpi"
        echo
        echo "ROCm path:"
        echo "  $ROCM_PATH"
        echo
        echo "LD_LIBRARY_PATH:"
        echo "  $LD_LIBRARY_PATH"
        exit 1
    fi
}

check_libraries "$UNIT_TEST_BINARY"
check_libraries "$FUNCTIONAL_TEST_BINARY"

###############################################################################
# FRONTIER LAUNCH WRAPPER
###############################################################################

cat > "$MPI_LAUNCHER" <<'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

TASKS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|-np|--np)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a process count."
                exit 2
            fi

            TASKS="$2"
            shift 2
            ;;

        -mca|--mca)
            if [[ $# -lt 3 ]]; then
                echo "ERROR: $1 requires a key and value."
                exit 2
            fi

            # Drop Open MPI MCA key and value.
            shift 3
            ;;

        -x)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: -x requires an environment specification."
                exit 2
            fi

            EXPORT_SPEC="$2"

            if [[ "$EXPORT_SPEC" == *=* ]]; then
                export "$EXPORT_SPEC"
            elif [[ -n "${!EXPORT_SPEC:-}" ]]; then
                export "$EXPORT_SPEC=${!EXPORT_SPEC}"
            fi

            shift 2
            ;;

        --timeout|--map-by|--bind-to)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a value."
                exit 2
            fi

            # CTest handles timeout. Slurm handles placement and binding.
            shift 2
            ;;

        --report-bindings|--oversubscribe|--allow-run-as-root)
            shift
            ;;

        --)
            shift
            break
            ;;

        -*)
            echo "ERROR: Unsupported MPI launcher option: $1"
            exit 2
            ;;

        *)
            # First non-launcher argument is the executable.
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "ERROR: No executable was supplied to the Frontier launcher."
    exit 2
fi

EXECUTABLE="$1"
ALLOCATED_NODES="${SLURM_JOB_NUM_NODES:-1}"

if [[ "$EXECUTABLE" == *unit* ]]; then
    # Unit tests are intended to use GPUs within one node.
    LAUNCH_NODES=1
else
    # Spread functional tests across the allocation to exercise RO.
    LAUNCH_NODES="$ALLOCATED_NODES"

    if (( TASKS < LAUNCH_NODES )); then
        LAUNCH_NODES="$TASKS"
    fi
fi

if (( LAUNCH_NODES < 1 )); then
    LAUNCH_NODES=1
fi

TASKS_PER_NODE=$(((TASKS + LAUNCH_NODES - 1) / LAUNCH_NODES))

if (( TASKS_PER_NODE > 8 )); then
    echo "ERROR: Requested $TASKS_PER_NODE tasks per node."
    echo "Frontier provides 8 GPU dies per node."
    exit 2
fi

echo "Frontier launch: tasks=$TASKS nodes=$LAUNCH_NODES executable=$EXECUTABLE"

exec srun \
    --export=ALL \
    -N "$LAUNCH_NODES" \
    -n "$TASKS" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    -c7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --distribution=cyclic \
    bash -c '
        export OMPI_COMM_WORLD_LOCAL_RANK="$SLURM_LOCALID"
        export OMPI_COMM_WORLD_RANK="$SLURM_PROCID"
        export OMPI_COMM_WORLD_SIZE="$SLURM_NTASKS"

        exec "$@"
    ' bash "$@"
LAUNCHER_EOF

chmod +x "$MPI_LAUNCHER"

###############################################################################
# TEMPORARILY PATCH INSTALLED CTEST FILES
#
# Replace /usr/bin/srun with the compatibility launcher. Files are restored
# when this script exits or is interrupted.
###############################################################################

PATCH_SUFFIX=".tests-sh-backup.$$"
PATCHED_FILES=()

restore_ctest_files() {
    local file

    for file in "${PATCHED_FILES[@]}"; do
        if [[ -f "$file$PATCH_SUFFIX" ]]; then
            mv "$file$PATCH_SUFFIX" "$file"
        fi
    done
}

trap restore_ctest_files EXIT INT TERM

while IFS= read -r file; do
    cp -p "$file" "$file$PATCH_SUFFIX"
    PATCHED_FILES+=("$file")

    sed -i \
        "s|/usr/bin/srun|$MPI_LAUNCHER|g" \
        "$file"
done < <(
    grep -rlF \
        "/usr/bin/srun" \
        "$INSTALL_DIR/bin/rocshmem" \
        --include="*.cmake" \
        --include="CTestTestfile.cmake" ||
        true
)

# It is also valid if a previously generated file already references this
# launcher path.
if [[ "${#PATCHED_FILES[@]}" -eq 0 ]]; then
    if ! grep -rlF \
        "$MPI_LAUNCHER" \
        "$INSTALL_DIR/bin/rocshmem" \
        --include="*.cmake" \
        --include="CTestTestfile.cmake" \
        >/dev/null 2>&1; then
        echo "ERROR: Could not locate the test launcher in CTest metadata."
        exit 1
    fi
fi

###############################################################################
# DISPLAY CONFIGURATION
###############################################################################

echo
echo "rocSHMEM test configuration"
echo "  Allocation:   $SLURM_JOB_ID"
echo "  Nodes:        ${SLURM_JOB_NUM_NODES:-unknown}"
echo "  ROCm:         $ROCM_PATH"
echo "  Backend:      RO with local IPC"
echo "  Installation: $INSTALL_DIR"
echo "  Launcher:     $MPI_LAUNCHER"
echo "  Logs:         $LOG_DIR"
echo

###############################################################################
# RUN ALL REGISTERED TESTS
###############################################################################

TEST_STATUS=0

echo "============================================================"
echo "Running unit tests"
echo "============================================================"

if ! ctest \
    --test-dir "$UNIT_TEST_DIR" \
    --output-on-failure \
    --timeout "$TEST_TIMEOUT" \
    -j1 2>&1 | tee "$LOG_DIR/unit-tests.log"; then
    TEST_STATUS=1
fi

echo
echo "============================================================"
echo "Running all functional tests"
echo "============================================================"

if ! ctest \
    --test-dir "$FUNCTIONAL_TEST_DIR" \
    --output-on-failure \
    --timeout "$TEST_TIMEOUT" \
    -j1 2>&1 | tee "$LOG_DIR/functional-tests.log"; then
    TEST_STATUS=1
fi

###############################################################################
# SUMMARY
###############################################################################

echo
echo "============================================================"

if [[ "$TEST_STATUS" -eq 0 ]]; then
    echo "All registered rocSHMEM tests passed."
else
    echo "One or more rocSHMEM tests failed."
fi

echo
echo "Logs:"
echo "  $LOG_DIR/unit-tests.log"
echo "  $LOG_DIR/functional-tests.log"
echo "============================================================"

exit "$TEST_STATUS"