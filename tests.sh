#!/bin/bash
set -uo pipefail

# Usage:
#   UNIT_ONLY=1 ./test.sh   # unit suite only
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_REGEX='^put_n2_w1_z1024_512B(_uuid)?$' \
#     FUNCTIONAL_REPEAT=20 ./test.sh
#
# Incremental Testing Usage:
#   START_INDEX=1 BATCH_SIZE=15 ./test.sh
#   START_INDEX=16 BATCH_SIZE=15 APPEND_LOG=1 ./test.sh

ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
UNIT_ONLY="${UNIT_ONLY:-0}"
FUNCTIONAL_ONLY="${FUNCTIONAL_ONLY:-0}"
FUNCTIONAL_REGEX="${FUNCTIONAL_REGEX:-}"
FUNCTIONAL_REPEAT="${FUNCTIONAL_REPEAT:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

START_INDEX="${START_INDEX:-1}"
BATCH_SIZE="${BATCH_SIZE:-15}"
APPEND_LOG="${APPEND_LOG:-0}"

if [[ "$UNIT_ONLY" == 1 && "$FUNCTIONAL_ONLY" == 1 ]]; then
    echo "Error: UNIT_ONLY=1 and FUNCTIONAL_ONLY=1 cannot be used together."
    exit 2
fi

if [[ ! "$FUNCTIONAL_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: FUNCTIONAL_REPEAT must be a positive integer."
    exit 2
fi

module reset
module load cpe/26.03
module swap PrgEnv-cray PrgEnv-gnu
module load gcc-native/14.2
module load "rocm/$ROCM_VERSION"
module load craype-accel-amd-gfx90a
module load cray-mpich/9.1.0
module load cmake

module unload darshan-runtime >/dev/null 2>&1 || true

if module -t list 2>&1 | grep -q '^darshan-runtime/'; then
    echo "Error: darshan-runtime is still loaded."
    exit 1
fi

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$WORKSPACE_DIR/install}"
UNIT_DIR="$INSTALL_DIR/bin/rocshmem/tests/unit"
FUNCTIONAL_DIR="$INSTALL_DIR/bin/rocshmem/tests/functional"
HELPER_DIR="$WORKSPACE_DIR/test-helpers"
LOG_ROOT="$WORKSPACE_DIR/test-logs"

if [[ "$APPEND_LOG" == 1 && -d "$LOG_ROOT" ]]; then
    LAST_RUN_ID=$(ls -1t "$LOG_ROOT" 2>/dev/null | head -n 1)
    if [[ -n "$LAST_RUN_ID" ]]; then
        RUN_ID="$LAST_RUN_ID"
    else
        RUN_ID="$(date +%Y%m%d-%H%M%S)"
    fi
else
    RUN_ID="$(date +%Y%m%d-%H%M%S)"
fi

LOG_DIR="$LOG_ROOT/$RUN_ID"
LAUNCHER="$HELPER_DIR/frontier-mpiexec"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: run this script inside a Frontier compute-node allocation."
    echo "Example: salloc -A <project> -N2 -t 01:00:00"
    exit 1
fi

if [[ ! -d "$UNIT_DIR" || ! -d "$FUNCTIONAL_DIR" ]]; then
    echo "Error: installed rocSHMEM tests were not found under:"
    echo "  $INSTALL_DIR"
    echo "Run ./build.sh first."
    exit 1
fi

ALLOCATED_NODES="${SLURM_NNODES:-1}"
if [[ "$UNIT_ONLY" != 1 ]] && ((ALLOCATED_NODES < 2)); then
    echo "Error: the cross-node RO tests require at least two allocated nodes."
    echo "Example: salloc -A <project> -N2 -t 01:00:00"
    exit 1
fi

ROCM_ROOT="${ROCM_PATH:-/opt/rocm-$ROCM_VERSION}"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_ROOT/lib:$ROCM_ROOT/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

mkdir -p "$HELPER_DIR" "$LOG_DIR"

cat > "$LAUNCHER" <<'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

tasks=1

while (($#)); do
    case "$1" in
        -n|-np|--np)
            tasks="$2"
            shift 2
            ;;
        -mca|--mca)
            shift 3
            ;;
        -x)
            spec="$2"
            if [[ "$spec" == *=* ]]; then
                export "$spec"
            else
                export "$spec=${!spec-}"
            fi
            shift 2
            ;;
        --timeout|--map-by|--bind-to)
            shift 2
            ;;
        --oversubscribe)
            shift
            ;;
        --)
            shift
            break
            ;;
        -* )
            echo "frontier-mpiexec: unsupported launcher option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if (($# == 0)); then
    echo "frontier-mpiexec: no executable was supplied" >&2
    exit 2
fi

requested_nodes="${ROCSHMEM_TEST_NODES:-2}"
allocated_nodes="${SLURM_NNODES:-1}"
nodes="$requested_nodes"

((nodes > allocated_nodes)) && nodes="$allocated_nodes"
((nodes > tasks)) && nodes="$tasks"
((nodes < 1)) && nodes=1

tasks_per_node=$(((tasks + nodes - 1) / nodes))
if ((tasks_per_node > 8)); then
    echo "frontier-mpiexec: $tasks tasks on $nodes nodes requires more than 8 GPUs per node" >&2
    exit 2
fi

echo "Frontier launch: tasks=$tasks nodes=$nodes executable=$1"

exec srun \
    --export=ALL \
    --nodes="$nodes" \
    --ntasks="$tasks" \
    --ntasks-per-node="$tasks_per_node" \
    --cpus-per-task="${ROCSHMEM_CPUS_PER_TASK:-7}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    bash -c '
        export OMPI_COMM_WORLD_RANK="${SLURM_PROCID:?}"
        export OMPI_COMM_WORLD_SIZE="${SLURM_NTASKS:?}"
        export OMPI_COMM_WORLD_LOCAL_RANK="${SLURM_LOCALID:?}"
        export OMPI_COMM_WORLD_LOCAL_SIZE="${SLURM_NTASKS_PER_NODE:-1}"
        exec "$@"
    ' bash "$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

declare -a PATCHED_FILES=()
restore_ctest_files() {
    local file
    for file in "${PATCHED_FILES[@]}"; do
        if [[ -f "$file.frontier-backup" ]]; then
            mv -f "$file.frontier-backup" "$file"
        fi
    done
}
trap restore_ctest_files EXIT INT TERM

while IFS= read -r file; do
    cp -p "$file" "$file.frontier-backup"
    PATCHED_FILES+=("$file")
    sed -i "s|/usr/bin/srun|$LAUNCHER|g" "$file"
done < <(grep -RIl --include='CTestTestfile.cmake' '/usr/bin/srun' "$UNIT_DIR" "$FUNCTIONAL_DIR")

echo
echo "rocSHMEM test configuration"
echo "  Allocation:  $SLURM_JOB_ID"
echo "  Nodes:       $ALLOCATED_NODES"
echo "  Unit tests:  IPC on one node"
echo "  Functional:  RO across two nodes, mixed IPC disabled"
echo "  Launcher:    $LAUNCHER"
echo "  Logs:        $LOG_DIR"

PREFLIGHT_EXE="$INSTALL_DIR/share/rocshmem/rocshmem_unit_tests"
echo
echo "Checking runtime libraries on a compute node"
export ROCSHMEM_TEST_NODES=1
if ! "$LAUNCHER" -n 1 bash -c '
    missing="$(ldd "$1" | sed -n "/not found/p")"
    if [[ -n "$missing" ]]; then
        echo "Missing runtime libraries:" >&2
        echo "$missing" >&2
        exit 1
    fi
    if ! python3 -c '\''import ctypes; ctypes.CDLL("libmpi.so")'\''; then
        echo "Unable to load libmpi.so through LD_LIBRARY_PATH." >&2
        exit 1
    fi
    echo "MPI runtime-library check passed."
    echo "Runtime-library check passed."
' bash "$PREFLIGHT_EXE"; then
    echo
    echo "Error: required libraries are not visible inside the Slurm task."
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi

overall_rc=0
unit_rc=0
functional_rc=0

declare -a CTEST_COMMON_ARGS=(--output-on-failure -j1)
if [[ "$STOP_ON_FAILURE" == 1 ]]; then
    CTEST_COMMON_ARGS+=(--stop-on-failure)
fi

REDIRECT=">"
if [[ "$APPEND_LOG" == 1 ]]; then
    REDIRECT=">>"
fi

if [[ "$FUNCTIONAL_ONLY" != 1 ]]; then
    echo
    echo "============================================================"
    echo "Running unit tests with the IPC backend"
    echo "============================================================"
    export ROCSHMEM_BACKEND=ipc
    export ROCSHMEM_BACKEND_TYPE=ipc
    export ROCSHMEM_TEST_NODES=1
    export ROCSHMEM_DISABLE_MIXED_IPC=0

    if [[ "$APPEND_LOG" == 1 ]]; then
        ctest --test-dir "$UNIT_DIR" "${CTEST_COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/unit-tests.log"
    else
        ctest --test-dir "$UNIT_DIR" "${CTEST_COMMON_ARGS[@]}" 2>&1 | tee "$LOG_DIR/unit-tests.log"
    fi
    unit_rc=${PIPESTATUS[0]}
    if ((unit_rc != 0)); then
        overall_rc=1
    fi
fi

if [[ "$UNIT_ONLY" == 1 ]]; then
    echo
    echo "============================================================"
    if ((unit_rc == 0)); then
        echo "The rocSHMEM unit suite passed."
    else
        echo "The rocSHMEM unit suite failed."
    fi
    echo "Log: $LOG_DIR/unit-tests.log"
    echo "============================================================"
    exit "$unit_rc"
fi

if ((unit_rc != 0)) && [[ "$STOP_ON_FAILURE" == 1 ]]; then
    echo
    echo "Stopping before the functional suite because the unit suite failed."
    echo "Log: $LOG_DIR/unit-tests.log"
    exit "$unit_rc"
fi

echo
echo "============================================================"
echo "Preparing Functional Suite Batch Execution"
echo "============================================================"
export ROCSHMEM_BACKEND=ro
export ROCSHMEM_BACKEND_TYPE=ro
export ROCSHMEM_TEST_NODES=2
export ROCSHMEM_DISABLE_MIXED_IPC=1

export FI_CXI_RX_MATCH_MODE=hybrid
export MPICH_NEMESIS_ASYNC_PROGRESS=1
export FI_CXI_DEFAULT_TX_SIZE=524288

declare -a FUNCTIONAL_CTEST_ARGS=("${CTEST_COMMON_ARGS[@]}")

echo "Gathering list of functional tests..."
ALL_FUNCTIONAL_TESTS=$(ctest --test-dir "$FUNCTIONAL_DIR" -N | awk -F': ' '/Test *#/ {print $2}' | tr -d '[:blank:]')
TOTAL_TESTS=$(echo "$ALL_FUNCTIONAL_TESTS" | wc -l)

END_INDEX=$((START_INDEX + BATCH_SIZE - 1))
BATCH_TESTS=$(echo "$ALL_FUNCTIONAL_TESTS" | tail -n +"$START_INDEX" | head -n "$BATCH_SIZE")
BATCH_COUNT=$(echo "$BATCH_TESTS" | grep -v '^$' | wc -l || echo 0)

if (( BATCH_COUNT == 0 )); then
    echo "Error: No tests found in range ${START_INDEX} to ${END_INDEX}. Total functional tests: ${TOTAL_TESTS}."
    exit 2
fi

INCREMENT_REGEX="^($(echo "$BATCH_TESTS" | paste -sd '|' -))$"
FUNCTIONAL_CTEST_ARGS+=(--tests-regex "$INCREMENT_REGEX")

echo "Running functional tests index ${START_INDEX} to $((START_INDEX + BATCH_COUNT - 1)) (Batch size: ${BATCH_COUNT} of ${TOTAL_TESTS})"

if [[ -n "$FUNCTIONAL_REGEX" ]]; then
    echo "Warning: Overriding explicit FUNCTIONAL_REGEX because index-based batching is active."
fi

if ((FUNCTIONAL_REPEAT > 1)); then
    FUNCTIONAL_CTEST_ARGS+=(--repeat "until-fail:$FUNCTIONAL_REPEAT")
    echo "Each selected functional test must pass $FUNCTIONAL_REPEAT consecutive runs."
fi

if [[ "$APPEND_LOG" == 1 ]]; then
    echo -e "\n--- BATCH START: INDEX ${START_INDEX} TO ${END_INDEX} ---" >> "$LOG_DIR/functional-tests.log"
    ctest --test-dir "$FUNCTIONAL_DIR" "${FUNCTIONAL_CTEST_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/functional-tests.log"
else
    ctest --test-dir "$FUNCTIONAL_DIR" "${FUNCTIONAL_CTEST_ARGS[@]}" 2>&1 | tee "$LOG_DIR/functional-tests.log"
fi
functional_rc=${PIPESTATUS[0]}
if ((functional_rc != 0)); then
    overall_rc=1
fi

echo
echo "============================================================"
if ((overall_rc == 0)); then
    echo "All selected rocSHMEM tests passed."
else
    echo "One or more rocSHMEM tests failed."
fi
echo
echo "Logs:"
if [[ "$FUNCTIONAL_ONLY" != 1 ]]; then
    echo "  $LOG_DIR/unit-tests.log"
fi
echo "  $LOG_DIR/functional-tests.log"
echo "============================================================"

exit "$overall_rc"