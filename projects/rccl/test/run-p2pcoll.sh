#!/usr/bin/env bash
# Build and run the standalone p2pcoll reproducer against build/debug/librccl.so.
#
# p2pcoll exercises the NCCL GH#1859 sequence (CommInitAll -> CommRegister ->
# Send/Recv -> AllReduce on the same registered buffer) as a plain program, so
# RCCL's NCCL_DEBUG=INFO trace is visible (the unit-test harness swallows it).
#
# Assumes a librccl.so already built WITH coverage in build/debug
# (configure with -DENABLE_CODE_COVERAGE=ON). The coverage instrumentation in
# the library is what lets you confirm whether ipcRegisterBuffer in p2p.cc is
# actually executed.
#
# Usage:
#   test/run-p2pcoll.sh                 # build + run with REG,P2P trace
#   BUILD_DIR=build/release test/run-p2pcoll.sh
#   COVERAGE=1 test/run-p2pcoll.sh      # also write/merge a profraw and report
#                                       # ipcRegisterBuffer coverage for p2p.cc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${RCCL_ROOT}/build/debug}"
LIB="${BUILD_DIR}/librccl.so"
INCLUDE_DIR="${BUILD_DIR}/include"
SRC="${SCRIPT_DIR}/p2pcoll.cpp"
BIN="${BUILD_DIR}/test/p2pcoll"

HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
ROCM_INCLUDE="${ROCM_INCLUDE:-/opt/rocm/include}"

if [[ ! -f "${LIB}" ]]; then
  echo "error: ${LIB} not found." >&2
  echo "       Build librccl.so first (with -DENABLE_CODE_COVERAGE=ON for coverage)." >&2
  exit 1
fi
if [[ ! -f "${INCLUDE_DIR}/rccl/rccl.h" ]]; then
  echo "error: ${INCLUDE_DIR}/rccl/rccl.h not found." >&2
  exit 1
fi

mkdir -p "$(dirname "${BIN}")"

echo "==> Building ${BIN}"
"${HIPCC}" -D__HIP_PLATFORM_AMD__ -O2 -std=c++17 \
  -o "${BIN}" "${SRC}" \
  -I"${INCLUDE_DIR}" -I"${ROCM_INCLUDE}" \
  -L"${BUILD_DIR}" -lrccl

RUN_ENV=(
  NCCL_LOCAL_REGISTER=1
  NCCL_DEBUG=INFO
  NCCL_DEBUG_SUBSYS=REG,P2P
  # Force NCCL_PROTO_SIMPLE for P2P operations.
  # addP2pToPlan (enqueue.cc) selects LL protocol when
  #   bytes <= nChannels * NCCL_P2P_LL_THRESHOLD  (default 8192)
  # The IPC registration path (ncclRegisterP2pIpcBuffer -> ipcRegisterBuffer)
  # is only taken when protocol == NCCL_PROTO_SIMPLE, so without this env var
  # small payloads silently bypass it.  Setting the threshold to 0 disables LL
  # unconditionally, guaranteeing SIMPLE is chosen regardless of payload size.
  NCCL_P2P_LL_THRESHOLD=0
  # Allow legacy cudaIpcGetMemHandle registration for hipMalloc'd buffers.
  # ipcRegisterBuffer's non-cuMem path (else if legacyIpcCap) calls
  # ncclParamLegacyCudaRegister() and goto fail if it is 0 (the default),
  # which leaves *regBufFlag=0 and bypasses lines 1188-1222 entirely.
  # NCCL_CUMEM_ENABLE=1 + cuMem allocations would be the alternative path.
  NCCL_LEGACY_CUDA_REGISTER=1
  "LD_LIBRARY_PATH=${BUILD_DIR}:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)

if [[ "${COVERAGE:-0}" == "1" ]]; then
  COV_DIR="${BUILD_DIR}/test/coverage-p2pcoll"
  PROFDATA="${COV_DIR}/merged.profdata"
  LLVM_BIN="/opt/rocm/llvm/bin"
  PROFDATA_TOOL="${LLVM_BIN}/llvm-profdata"
  COV_TOOL="${LLVM_BIN}/llvm-cov"
  if [[ ! -x "${PROFDATA_TOOL}" ]]; then PROFDATA_TOOL="$(command -v llvm-profdata)"; fi
  if [[ ! -x "${COV_TOOL}" ]];      then COV_TOOL="$(command -v llvm-cov)";           fi

  mkdir -p "${COV_DIR}"
  rm -f "${COV_DIR}"/*.profraw

  echo "==> Running ${BIN} (coverage)"
  env "${RUN_ENV[@]}" \
      LLVM_PROFILE_FILE="${COV_DIR}/p2pcoll_%p_%m.profraw" \
      "${BIN}" || true

  PROFRAW_FILES=( "${COV_DIR}"/*.profraw )
  echo "==> Merging ${#PROFRAW_FILES[@]} profraw file(s) -> ${PROFDATA}"
  "${PROFDATA_TOOL}" merge -sparse "${PROFRAW_FILES[@]}" -o "${PROFDATA}"

  echo ""
  echo "==> ipcRegisterBuffer coverage in p2p.cc"
  "${COV_TOOL}" report "${BIN}" \
    -object "${LIB}" \
    -instr-profile="${PROFDATA}" \
    --name=ipcRegisterBuffer \
    "${BUILD_DIR}/hipify/src/transport/p2p_tmp.cc"
else
  echo "==> Running ${BIN}"
  env "${RUN_ENV[@]}" "${BIN}"
fi
