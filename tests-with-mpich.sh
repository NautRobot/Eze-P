#!/bin/bash
set -uo pipefail

# Examples:
#   UNIT_ONLY=1 ./test.sh
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_BACKEND=ro \
#     FUNCTIONAL_LAYOUT=2x1 BATCH_SIZE=10000 ./test.sh
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_BACKEND=ro \
#     FUNCTIONAL_LAYOUT=2x2 BATCH_SIZE=10000 ./test.sh
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_BACKEND=ipc \
#     FUNCTIONAL_LAYOUT=1x2 BATCH_SIZE=10000 ./test.sh
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_BACKEND=ipc \
#     FUNCTIONAL_REGEX='^tile_put_.*' ./test.sh
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_BACKEND=ro \
#     FUNCTIONAL_REGEX='^(put|waveput|wgput)_n2_.*_v1073741824B(_uuid)?$' ./test.sh

ROCM_VERSION="${ROCM_VERSION:-7.2.0}"
UNIT_ONLY="${UNIT_ONLY:-0}"
FUNCTIONAL_ONLY="${FUNCTIONAL_ONLY:-0}"
FUNCTIONAL_BACKEND="${FUNCTIONAL_BACKEND:-ro}"
FUNCTIONAL_REGEX="${FUNCTIONAL_REGEX:-}"
FUNCTIONAL_REPEAT="${FUNCTIONAL_REPEAT:-1}"
ROCSHMEM_RO_TEST_NODES="${ROCSHMEM_RO_TEST_NODES:-2}"
FUNCTIONAL_LAYOUT="${FUNCTIONAL_LAYOUT:-auto}"
FUNCTIONAL_HEAP_SIZE="${FUNCTIONAL_HEAP_SIZE:-51539607552}"
RO_DISABLE_MIXED_IPC="${RO_DISABLE_MIXED_IPC:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
LARGE_VOLUME_BATCH_SIZE="${LARGE_VOLUME_BATCH_SIZE:-1}"

START_INDEX="${START_INDEX:-1}"
BATCH_SIZE="${BATCH_SIZE:-15}"
APPEND_LOG="${APPEND_LOG:-0}"

if [[ "$UNIT_ONLY" == 1 && "$FUNCTIONAL_ONLY" == 1 ]]; then
    echo "Error: UNIT_ONLY=1 and FUNCTIONAL_ONLY=1 cannot be combined."
    exit 2
fi

if [[ "$FUNCTIONAL_BACKEND" != ro && "$FUNCTIONAL_BACKEND" != ipc ]]; then
    echo "Error: FUNCTIONAL_BACKEND must be ro or ipc."
    exit 2
fi

if [[ "$FUNCTIONAL_LAYOUT" != auto && "$FUNCTIONAL_LAYOUT" != 1x2 && \
      "$FUNCTIONAL_LAYOUT" != 2x1 && "$FUNCTIONAL_LAYOUT" != 2x2 ]]; then
    echo "Error: FUNCTIONAL_LAYOUT must be auto, 1x2, 2x1, or 2x2."
    exit 2
fi

if [[ "$RO_DISABLE_MIXED_IPC" != 0 && "$RO_DISABLE_MIXED_IPC" != 1 ]]; then
    echo "Error: RO_DISABLE_MIXED_IPC must be 0 or 1."
    exit 2
fi

for value_name in FUNCTIONAL_REPEAT START_INDEX BATCH_SIZE ROCSHMEM_RO_TEST_NODES LARGE_VOLUME_BATCH_SIZE FUNCTIONAL_HEAP_SIZE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $value_name must be a positive integer."
        exit 2
    fi
done

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
LAUNCHER="$HELPER_DIR/frontier-mpiexec"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: run this script inside a Frontier compute-node allocation."
    echo "Example: salloc -A <project> -N8 -t 04:00:00"
    exit 1
fi

if [[ ! -d "$UNIT_DIR" || ! -d "$FUNCTIONAL_DIR" ]]; then
    echo "Error: installed rocSHMEM tests were not found under $INSTALL_DIR."
    echo "Run ./build.sh first."
    exit 1
fi

ALLOCATED_NODES="${SLURM_NNODES:-1}"
LAYOUT_NODES=0
LAYOUT_GPUS_PER_NODE=0
LAYOUT_RANKS=0
case "$FUNCTIONAL_LAYOUT" in
    1x2) LAYOUT_NODES=1; LAYOUT_GPUS_PER_NODE=2; LAYOUT_RANKS=2 ;;
    2x1) LAYOUT_NODES=2; LAYOUT_GPUS_PER_NODE=1; LAYOUT_RANKS=2 ;;
    2x2) LAYOUT_NODES=2; LAYOUT_GPUS_PER_NODE=2; LAYOUT_RANKS=4 ;;
esac

if [[ "$UNIT_ONLY" != 1 && "$FUNCTIONAL_LAYOUT" != auto ]]; then
    if ((LAYOUT_NODES > ALLOCATED_NODES)); then
        echo "Error: FUNCTIONAL_LAYOUT=$FUNCTIONAL_LAYOUT needs $LAYOUT_NODES nodes, but only $ALLOCATED_NODES are allocated."
        exit 1
    fi
    if [[ "$FUNCTIONAL_BACKEND" == ipc && "$LAYOUT_NODES" != 1 ]]; then
        echo "Error: the IPC backend is node-local; use FUNCTIONAL_LAYOUT=1x2."
        exit 2
    fi
elif [[ "$UNIT_ONLY" != 1 && "$FUNCTIONAL_BACKEND" == ro ]]; then
    if ((ROCSHMEM_RO_TEST_NODES > ALLOCATED_NODES)); then
        echo "Error: ROCSHMEM_RO_TEST_NODES=$ROCSHMEM_RO_TEST_NODES exceeds the $ALLOCATED_NODES allocated nodes."
        exit 1
    fi
fi

if [[ "$APPEND_LOG" == 1 && -d "$LOG_ROOT" ]]; then
    RUN_ID="$(ls -1t "$LOG_ROOT" 2>/dev/null | head -n 1)"
    [[ -n "$RUN_ID" ]] || RUN_ID="$(date +%Y%m%d-%H%M%S)"
else
    RUN_ID="$(date +%Y%m%d-%H%M%S)"
fi
LOG_DIR="$LOG_ROOT/$RUN_ID"

ROCM_ROOT="${ROCM_PATH:-/opt/rocm-$ROCM_VERSION}"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_ROOT/lib:$ROCM_ROOT/lib64:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export ROCSHMEM_LARGE_VOLUME_BATCH_SIZE="$LARGE_VOLUME_BATCH_SIZE"
export ROCSHMEM_FUNCTIONAL_HEAP_SIZE="$FUNCTIONAL_HEAP_SIZE"
export ROCSHMEM_FORCE_FUNCTIONAL_HEAP=0

unset MPICH_NEMESIS_ASYNC_PROGRESS
unset FI_MR_CACHE_MONITOR

mkdir -p "$HELPER_DIR" "$LOG_DIR"

cat > "$LAUNCHER" <<'LAUNCHER_EOF'
#!/bin/bash
set -euo pipefail

tasks=1

while (($#)); do
    case "$1" in
        -n|-np|--np)
            (($# >= 2)) || { echo "frontier-mpiexec: $1 requires a value" >&2; exit 2; }
            tasks="$2"
            shift 2
            ;;
        -mca|--mca)
            (($# >= 3)) || { echo "frontier-mpiexec: $1 requires a key and value" >&2; exit 2; }
            shift 3
            ;;
        -x)
            (($# >= 2)) || { echo "frontier-mpiexec: -x requires a value" >&2; exit 2; }
            spec="$2"
            if [[ "$spec" == ROCSHMEM_HEAP_SIZE=* && \
                  "${ROCSHMEM_FORCE_FUNCTIONAL_HEAP:-0}" == 1 ]]; then
                spec="ROCSHMEM_HEAP_SIZE=${ROCSHMEM_FUNCTIONAL_HEAP_SIZE:?}"
            fi
            if [[ "$spec" == *=* ]]; then
                export "$spec"
            else
                export "$spec=${!spec-}"
            fi
            shift 2
            ;;
        --timeout|--map-by|--bind-to|--distribution|-m)
            (($# >= 2)) || { echo "frontier-mpiexec: $1 requires a value" >&2; exit 2; }
            shift 2
            ;;
        --oversubscribe)
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "frontier-mpiexec: unsupported launcher option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

(($#)) || { echo "frontier-mpiexec: no executable was supplied" >&2; exit 2; }
[[ "$tasks" =~ ^[1-9][0-9]*$ ]] || { echo "frontier-mpiexec: invalid task count: $tasks" >&2; exit 2; }

command_args=("$@")
volume_size=0
has_batch=0
for ((i = 1; i < ${#command_args[@]}; i++)); do
    case "${command_args[i]}" in
        -v)
            if ((i + 1 < ${#command_args[@]})); then
                volume_size="${command_args[i + 1]}"
            fi
            ;;
        -b|-batch)
            has_batch=1
            ;;
    esac
done

if [[ "$volume_size" =~ ^[0-9]+$ ]] && \
   ((volume_size >= 1073741824)) && ((has_batch == 0)); then
    command_args+=( -b "${ROCSHMEM_LARGE_VOLUME_BATCH_SIZE:-1}" )
    set -- "${command_args[@]}"
    echo "Frontier launch: limiting ${volume_size}-byte volume test to buffer batch ${ROCSHMEM_LARGE_VOLUME_BATCH_SIZE:-1}"
fi

requested_nodes="${ROCSHMEM_TEST_NODES:-1}"
allocated_nodes="${SLURM_NNODES:-1}"
nodes="$requested_nodes"

((nodes > allocated_nodes)) && nodes="$allocated_nodes"
((nodes > tasks)) && nodes="$tasks"
((nodes < 1)) && nodes=1

tasks_per_node=$(((tasks + nodes - 1) / nodes))
if ((tasks_per_node > 8)); then
    required_nodes=$(((tasks + 7) / 8))
    echo "frontier-mpiexec: $tasks tasks need at least $required_nodes nodes at 8 GPUs per node; selected nodes=$nodes" >&2
    exit 2
fi

echo "Frontier launch: tasks=$tasks nodes=$nodes tasks_per_node=$tasks_per_node heap=${ROCSHMEM_HEAP_SIZE:-default} executable=$1"

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
trap restore_ctest_files EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while IFS= read -r file; do
    cp -p "$file" "$file.frontier-backup"
    PATCHED_FILES+=("$file")
    sed -i "s|/usr/bin/srun|$LAUNCHER|g" "$file"
done < <(grep -RIl --include='CTestTestfile.cmake' '/usr/bin/srun' "$UNIT_DIR" "$FUNCTIONAL_DIR")

echo
echo "rocSHMEM test configuration"
echo "  Allocation:          $SLURM_JOB_ID"
echo "  Allocated nodes:     $ALLOCATED_NODES"
echo "  Functional backend: $FUNCTIONAL_BACKEND"
echo "  Functional layout:  $FUNCTIONAL_LAYOUT"
echo "  Functional heap/PE: $FUNCTIONAL_HEAP_SIZE bytes"
echo "  RO disables IPC:    $RO_DISABLE_MIXED_IPC"
echo "  1 GiB volume batch:  $LARGE_VOLUME_BATCH_SIZE"
echo "  Launcher:            $LAUNCHER"
echo "  Logs:                $LOG_DIR"

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
    if ! ldd "$1" | grep -q "libmpi_gtl_hsa"; then
        echo "GPU-aware Cray MPICH library libmpi_gtl_hsa.so is not linked." >&2
        exit 1
    fi
    echo "Runtime-library check passed."
' bash "$PREFLIGHT_EXE"; then
    echo "Error: required runtime libraries are not visible inside the Slurm task."
    exit 1
fi

overall_rc=0
unit_rc=0
functional_rc=0

declare -a CTEST_COMMON_ARGS=(--output-on-failure -j1)
if [[ "$STOP_ON_FAILURE" == 1 ]]; then
    CTEST_COMMON_ARGS+=(--stop-on-failure)
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
    ((unit_rc == 0)) || overall_rc=1
fi

if [[ "$UNIT_ONLY" == 1 ]]; then
    echo "Unit-test log: $LOG_DIR/unit-tests.log"
    exit "$unit_rc"
fi

if ((unit_rc != 0)) && [[ "$STOP_ON_FAILURE" == 1 ]]; then
    echo "Stopping before functional tests because unit tests failed."
    exit "$unit_rc"
fi

echo
echo "============================================================"
echo "Preparing backend-aware functional tests"
echo "============================================================"

if [[ "$FUNCTIONAL_BACKEND" == ro ]]; then
    export ROCSHMEM_BACKEND=ro
    export ROCSHMEM_BACKEND_TYPE=ro
    if [[ "$FUNCTIONAL_LAYOUT" == auto ]]; then
        export ROCSHMEM_TEST_NODES="$ROCSHMEM_RO_TEST_NODES"
        BACKEND_DESCRIPTION="RO-compatible tests across up to $ROCSHMEM_RO_TEST_NODES nodes"
    else
        export ROCSHMEM_TEST_NODES="$LAYOUT_NODES"
        BACKEND_DESCRIPTION="RO-compatible ${LAYOUT_RANKS}-PE tests using $LAYOUT_NODES node(s) and $LAYOUT_GPUS_PER_NODE GPU(s) per node"
    fi
    export ROCSHMEM_DISABLE_MIXED_IPC="$RO_DISABLE_MIXED_IPC"
    BACKEND_LABEL_REGEX='^(backend_all|backend_ro)$'
else
    export ROCSHMEM_BACKEND=ipc
    export ROCSHMEM_BACKEND_TYPE=ipc
    export ROCSHMEM_TEST_NODES=1
    export ROCSHMEM_DISABLE_MIXED_IPC=0
    BACKEND_LABEL_REGEX='^backend_ipc$'
    if [[ "$FUNCTIONAL_LAYOUT" == auto ]]; then
        BACKEND_DESCRIPTION='IPC-required tests on one node'
    else
        BACKEND_DESCRIPTION="IPC-required ${LAYOUT_RANKS}-PE tests using one node and $LAYOUT_GPUS_PER_NODE GPUs"
    fi
fi
export ROCSHMEM_FORCE_FUNCTIONAL_HEAP=1
export ROCSHMEM_HEAP_SIZE="$FUNCTIONAL_HEAP_SIZE"

declare -a FUNCTIONAL_CTEST_ARGS=(
    "${CTEST_COMMON_ARGS[@]}"
    --label-regex "$BACKEND_LABEL_REGEX"
)

echo "Gathering $BACKEND_DESCRIPTION"
mapfile -t ALL_FUNCTIONAL_TESTS < <(
    ctest --test-dir "$FUNCTIONAL_DIR" -N \
        --label-regex "$BACKEND_LABEL_REGEX" |
        awk -F': ' -v ranks="$LAYOUT_RANKS" '
            /Test *#/ {
                name=$2
                gsub(/[[:blank:]]/, "", name)
                if (ranks == 0 || index(name, "_n" ranks "_") != 0) print name
            }'
)
TOTAL_TESTS=${#ALL_FUNCTIONAL_TESTS[@]}

if ((TOTAL_TESTS == 0)); then
    echo "Error: no tests carry a label matching $BACKEND_LABEL_REGEX."
    exit 2
fi

if [[ -n "$FUNCTIONAL_REGEX" ]]; then
    MATCHED_FUNCTIONAL_TESTS=()
    for test_name in "${ALL_FUNCTIONAL_TESTS[@]}"; do
        if [[ "$test_name" =~ $FUNCTIONAL_REGEX ]]; then
            MATCHED_FUNCTIONAL_TESTS+=("$test_name")
        fi
    done
    if ((${#MATCHED_FUNCTIONAL_TESTS[@]} == 0)); then
        echo "Error: no $BACKEND_DESCRIPTION match FUNCTIONAL_REGEX=$FUNCTIONAL_REGEX."
        exit 2
    fi
    regex_body="$(IFS='|'; echo "${MATCHED_FUNCTIONAL_TESTS[*]}")"
    FUNCTIONAL_CTEST_ARGS+=(--tests-regex "^(${regex_body})$")
    SELECTION_LABEL="${FUNCTIONAL_BACKEND} REGEX ${FUNCTIONAL_REGEX}"
    echo "Backend selection: $BACKEND_DESCRIPTION"
    echo "Test regex: $FUNCTIONAL_REGEX"
else
    start_offset=$((START_INDEX - 1))
    if ((start_offset >= TOTAL_TESTS)); then
        echo "Error: START_INDEX=$START_INDEX exceeds $TOTAL_TESTS compatible tests."
        exit 2
    fi

    BATCH_TESTS=("${ALL_FUNCTIONAL_TESTS[@]:start_offset:BATCH_SIZE}")
    BATCH_COUNT=${#BATCH_TESTS[@]}
    regex_body="$(IFS='|'; echo "${BATCH_TESTS[*]}")"
    INCREMENT_REGEX="^(${regex_body})$"
    FUNCTIONAL_CTEST_ARGS+=(--tests-regex "$INCREMENT_REGEX")
    END_INDEX=$((START_INDEX + BATCH_COUNT - 1))
    SELECTION_LABEL="${FUNCTIONAL_BACKEND} INDEX ${START_INDEX} TO ${END_INDEX}"

    echo "Backend selection: $BACKEND_DESCRIPTION"
    echo "Running backend-relative index $START_INDEX to $END_INDEX ($BATCH_COUNT of $TOTAL_TESTS tests)"
fi

if ((FUNCTIONAL_REPEAT > 1)); then
    FUNCTIONAL_CTEST_ARGS+=(--repeat "until-fail:$FUNCTIONAL_REPEAT")
    echo "Each selected test must pass $FUNCTIONAL_REPEAT consecutive runs."
fi

if [[ "$APPEND_LOG" == 1 ]]; then
    echo -e "\n--- BATCH START: ${SELECTION_LABEL} ---" >> "$LOG_DIR/functional-tests.log"
    ctest --test-dir "$FUNCTIONAL_DIR" "${FUNCTIONAL_CTEST_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/functional-tests.log"
else
    ctest --test-dir "$FUNCTIONAL_DIR" "${FUNCTIONAL_CTEST_ARGS[@]}" 2>&1 | tee "$LOG_DIR/functional-tests.log"
fi
functional_rc=${PIPESTATUS[0]}
((functional_rc == 0)) || overall_rc=1

echo
echo "============================================================"
if ((overall_rc == 0)); then
    echo "All selected rocSHMEM tests passed."
else
    echo "One or more selected rocSHMEM tests failed."
fi
echo "Logs: $LOG_DIR"
echo "============================================================"

exit "$overall_rc"
