#!/bin/bash
set -uo pipefail

# Usage:
#   UNIT_ONLY=1 ./tests.sh   # unit suite only
#   FUNCTIONAL_ONLY=1 FUNCTIONAL_REGEX='^put_n2_w1_z1024_512B(_uuid)?$' \
#     FUNCTIONAL_REPEAT=20 ./tests-with-ompi.sh
#
# Incremental Testing Usage:
#   START_INDEX=1 BATCH_SIZE=15 ./tests-with-ompi.sh
#   START_INDEX=235 BATCH_SIZE=350 APPEND_LOG=1 FUNCTIONAL_ONLY=1 ./tests-with-ompi.sh

ROCM_VERSION="${ROCM_VERSION:-6.2.4}"
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
module load cce/18.0.1
module load "rocm/$ROCM_VERSION"
module load craype-accel-amd-gfx90a
module load ums/default
module load ums024/default
module load cmake
module unload darshan-runtime >/dev/null 2>&1 || true

# Force Open MPI's one-sided communication engine to use point-to-point emulation.
# This prevents native RDMA/shared-memory registration failures during MPI_Win_create.
export OMPI_MCA_osc="pt2pt"

# Ensure the point-to-point engine doesn't attempt unsupported acceleration paths
export OMPI_MCA_osc_pt2pt_no_p2p=1

# Ensure stable PML and BTL configurations
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="ofi,self,sm"

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
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$ROCM_ROOT/lib:$ROCM_ROOT/lib64:${MPI_DIR:-}/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PATH="$ROCM_ROOT/bin:$ROCM_ROOT/llvm/bin:$PATH"

mkdir -p "$HELPER_DIR" "$LOG_DIR"

# Write out the updated libfabric-only MPI launcher wrapper
cat > "/ccs/home/angeloej/eze-p/test-helpers/frontier-mpiexec" <<'LAUNCHER_EOF'
#!/usr/bin/env bash

OMPI_DIR="/sw/frontier/ums/ums024/cce/18.0.1/install/openmpi-5.0.9-20251111"
OFI_DIR="/sw/frontier/ums/ums024/cce/18.0.1/install/libfabric-2.3.1-20251111"
ROCM_DIR="/opt/rocm-6.2.4"

REAL_MPIEXEC="${OMPI_DIR}/bin/mpiexec"

export LD_LIBRARY_PATH="${OFI_DIR}/lib:${OFI_DIR}/lib64:${OMPI_DIR}/lib:${OMPI_DIR}/lib64:${ROCM_DIR}/lib:${ROCM_DIR}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${OMPI_DIR}/bin:${ROCM_DIR}/bin:${PATH}"

# Clear out conflicting/broken environment variables
unset OMPI_MCA_pml OMPI_MCA_osc OMPI_MCA_btl OMPI_MCA_mtl

# Native Slingshot 11 CXI Transport Setup
export FI_PROVIDER="cxi"
export FI_CRAY_SHARE_GNI=1
export FI_MR_CACHE_MONITOR=madvise

# Force Pt2Pt Window Emulation (prevents MPI_Win_create crashes)
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="ofi,self,sm"
export OMPI_MCA_osc="pt2pt"
export OMPI_MCA_osc_pt2pt_no_p2p=1

# GPU awareness & IPC settings
export OMPI_MCA_btl_smcuda_have_cuda_support=1
export ROCSHMEM_DISABLE_MIXED_IPC=1

# Strict filtering of incoming CTest arguments
args=()
tasks=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        -mca)
            # Purge any incoming PML, OSC, or BTL parameters injected by CTest
            key="$2"
            if [[ "$key" == "pml" || "$key" == "osc" || "$key" == "btl" || "$key" == "pml ^ucx" || "$key" == "osc ucx" ]]; then
                shift 2
            else
                args+=("$1" "$2" "$3")
                shift 3
            fi
            ;;
        -n)
            tasks="$2"
            shift 2
            ;;
        --timeout|--map-by|--distribution|-m)
            # Discard formatting/scheduling overrides
            shift 2
            ;;
        -x)
            # Filter out UCX variable overrides
            if [[ "$2" =~ "UCX_" ]]; then
                shift 2
            else
                IFS='=' read -r var val <<< "$2"
                export "$var"="$val"
                shift 2
            fi
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

# Execute real mpiexec forwarding only the stabilized engine settings
exec "$REAL_MPIEXEC" \
    -n "$tasks" \
    --oversubscribe \
    --bind-to core \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x FI_PROVIDER \
    -x FI_CRAY_SHARE_GNI \
    -x FI_MR_CACHE_MONITOR \
    -x OMPI_MCA_osc \
    -x OMPI_MCA_osc_pt2pt_no_p2p \
    -x OMPI_MCA_pml \
    -x OMPI_MCA_btl \
    -x ROCSHMEM_DISABLE_MIXED_IPC \
    "${args[@]}"
LAUNCHER_EOF

chmod +x /ccs/home/angeloej/eze-p/test-helpers/frontier-mpiexec

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

# Set native Libfabric/OFI configurations globally for the execution environment
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="ofi,self,sm"
export OMPI_MCA_osc="rdma"
export OMPI_MCA_osc_rdma_no_p2p=1

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

# Slingshot CXI / OFI setup variables
export FI_PROVIDER="cxi"
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_DEFAULT_TX_SIZE=524288
export HSA_ENABLE_IPC_MODE_LEGACY=1
export FI_MR_CACHE_MONITOR=madvise

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