#!/bin/bash
set -euo pipefail

# Apply the Frontier/Cray-MPICH RO visibility experiment, rebuild rocSHMEM,
# request a two-node allocation, and stress the previously failing tests.
#
# Usage:
#   bash ./fix-ro-sync.sh all       # patch, build, allocate, and test
#   bash ./fix-ro-sync.sh build     # patch and build only
#   bash ./fix-ro-sync.sh test      # test inside an existing 2-node allocation
#

MODE="${1:-all}"
ACCOUNT="${ACCOUNT:-csc607}"
ALLOC_TIME="${ALLOC_TIME:-00:15:00}"
REPEAT="${REPEAT:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$ROOT/src/rocm-systems/projects/rocshmem"
BACKEND_HEADER="$SOURCE_ROOT/src/reverse_offload/backend_ro.hpp"
CONTEXT_HEADER="$SOURCE_ROOT/src/reverse_offload/context_ro_host.hpp"
CONTEXT_SOURCE="$SOURCE_ROOT/src/reverse_offload/context_ro_host.cpp"
FUNCTIONAL_EXE="$ROOT/install/share/rocshmem/rocshmem_functional_tests"

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "required file not found: $1"
}

apply_source_patch() {
    require_file "$BACKEND_HEADER"
    require_file "$CONTEXT_HEADER"
    require_file "$CONTEXT_SOURCE"

    command -v python3 >/dev/null 2>&1 || fail "python3 is required to apply the source patch"

    python3 - "$BACKEND_HEADER" "$CONTEXT_HEADER" "$CONTEXT_SOURCE" <<'PY'
from pathlib import Path
import re
import sys

backend_header = Path(sys.argv[1])
context_header = Path(sys.argv[2])
context_source = Path(sys.argv[3])


def replace_once(text, old, new, description):
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"Cannot apply {description}: expected one matching source block, found {count}"
        )
    return text.replace(old, new, 1)


# Keep one device RMA window while validating the synchronization fix.
text = backend_header.read_text()
text, count = re.subn(
    r"size_t num_windows_\{\d+\};",
    "size_t num_windows_{1};",
    text,
    count=1,
)
if count != 1:
    raise SystemExit("Cannot locate num_windows_ in backend_ro.hpp")
backend_header.write_text(text)


text = context_header.read_text()
if "class ROBackend;" not in text:
    text = replace_once(
        text,
        "namespace rocshmem {\n",
        "namespace rocshmem {\n\nclass ROBackend;\n",
        "ROBackend forward declaration",
    )

if "ROBackend *ro_backend" not in text:
    text = replace_once(
        text,
        "  HostInterface *host_interface = nullptr;\n",
        "  HostInterface *host_interface = nullptr;\n\n"
        "  /* Backend owning the GPU-backed RO MPI windows. */\n"
        "  ROBackend *ro_backend = nullptr;\n",
        "ROBackend member",
    )
context_header.write_text(text)


text = context_source.read_text()
if "ro_backend = b;" not in text:
    text = replace_once(
        text,
        "  ROBackend *b{static_cast<ROBackend *>(backend)};\n",
        "  ROBackend *b{static_cast<ROBackend *>(backend)};\n"
        "  ro_backend = b;\n",
        "ROBackend constructor assignment",
    )

if "synchronizes the exact GPU-backed MPI windows" not in text:
    replacement = r'''__host__ void ROHostContext::barrier_all() {
  DPRINTF("Function: ro_net_host_barrier_all\n");

  host_interface->fence(context_window_info);

  // All origins must finish their device-side RO operations before each
  // target synchronizes the exact GPU-backed MPI windows used by those ops.
  host_interface->barrier_for_sync();

  WindowInfoMPI **window_info = ro_backend->ro_window_proxy_->get();
  size_t num_windows =
      ro_backend->ro_window_proxy_->get_num_MPI_windows();

  for (size_t i = 0; i < num_windows; ++i) {
    host_interface->sync_all(window_info[i]);
  }
}'''
    pattern = r"__host__ void ROHostContext::barrier_all\(\) \{.*?\n\}"
    text, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit("Cannot locate ROHostContext::barrier_all()")

context_source.write_text(text)
PY

    grep -q 'size_t num_windows_{1};' "$BACKEND_HEADER" || \
        fail "num_windows_ was not set to 1"
    grep -q 'ROBackend \*ro_backend' "$CONTEXT_HEADER" || \
        fail "ROBackend member was not installed"
    grep -q 'synchronizes the exact GPU-backed MPI windows' "$CONTEXT_SOURCE" || \
        fail "barrier synchronization patch was not installed"

    echo "RO synchronization source patch is present."
}

build_and_verify() {
    apply_source_patch

    cd "$ROOT"
    ./build.sh -DUSE_WF_COAL=ON

    grep -q '^USE_WF_COAL:BOOL=ON$' "$ROOT/build/CMakeCache.txt" || \
        fail "USE_WF_COAL is not ON in CMakeCache.txt"
    [[ -x "$FUNCTIONAL_EXE" ]] || \
        fail "installed functional-test executable is missing"

    echo
    echo "Build verification passed:"
    echo "  USE_WF_COAL: ON"
    echo "  RO MPI windows: 1"
    echo "  Synchronization patch: installed"
    echo "  Functional tests: installed"
}

verify_test_prerequisites() {
    require_file "$BACKEND_HEADER"
    require_file "$CONTEXT_HEADER"
    require_file "$CONTEXT_SOURCE"
    [[ -x "$FUNCTIONAL_EXE" ]] || fail "run the build mode first"
    [[ -x "$ROOT/tests.sh" ]] || fail "tests.sh is missing or not executable"

    grep -q 'size_t num_windows_{1};' "$BACKEND_HEADER" || \
        fail "source is not configured for one RO MPI window"
    grep -q 'synchronizes the exact GPU-backed MPI windows' "$CONTEXT_SOURCE" || \
        fail "RO synchronization patch is absent"
    grep -q '^USE_WF_COAL:BOOL=ON$' "$ROOT/build/CMakeCache.txt" || \
        fail "installed build does not have USE_WF_COAL=ON"

    local nodes="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-0}}"
    [[ "$nodes" =~ ^[0-9]+$ ]] || nodes=0
    ((nodes >= 2)) || fail "test mode requires an active two-node allocation"
}

run_targeted_tests() {
    verify_test_prerequisites
    cd "$ROOT"

    MPICH_ASYNC_PROGRESS=1 \
    ROCSHMEM_CPUS_PER_TASK=8 \
    FUNCTIONAL_ONLY=1 \
    FUNCTIONAL_REGEX='^put_n2_w1_z1024_512B(_uuid)?$' \
    FUNCTIONAL_REPEAT="$REPEAT" \
    ./tests.sh
}

request_allocation_and_test() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        run_targeted_tests
        return
    fi

    local quoted_root
    printf -v quoted_root '%q' "$ROOT"

    echo
    echo "Requesting two Frontier nodes for the targeted stress test."
    salloc \
        -A "$ACCOUNT" \
        -N2 \
        -t "$ALLOC_TIME" \
        bash -lc "cd $quoted_root && exec bash ./fix-ro-sync.sh test"
}

case "$MODE" in
    build)
        build_and_verify
        ;;
    test)
        run_targeted_tests
        ;;
    all)
        build_and_verify
        request_allocation_and_test
        ;;
    *)
        fail "usage: $0 [all|build|test]"
        ;;
esac
