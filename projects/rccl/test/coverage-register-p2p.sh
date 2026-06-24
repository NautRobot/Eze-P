#!/usr/bin/env bash
# Run Register.ProcessIsolatedRegisterTests (including P2pThenCollective_SameBuffer)
# under llvm source-based coverage and emit a text summary + HTML report scoped
# to src/transport/p2p.cc.
#
# The ProcessIsolatedTestRunner re-execs the binary once per sub-test; each
# child writes its own .profraw file (via RCCL_TEST_CODE_COVERAGE / the
# __llvm_profile_write_file() call in ProcessIsolatedTestRunner.cpp before
# _exit()). The parent profraw and all child profraw files are merged before
# reporting.
#
# Requirements:
#   - Configure with -DENABLE_CODE_COVERAGE=ON so rccl-UnitTests (and the
#     RCCL library itself) are built with -fprofile-instr-generate
#     -fcoverage-mapping and RCCL_TEST_CODE_COVERAGE is defined. Unlike the
#     top-level CMakeLists, the test CMakeLists does not gate this behind
#     Debug; it applies to any build type.
#   - llvm-profdata and llvm-cov on PATH or under /opt/rocm/llvm/bin.
#   - At least 2 AMD GPUs with fine-grained memory support (MI200+) so the
#     P2pThenCollective_SameBuffer sub-test does not auto-skip.
#
# Usage:
#   test/coverage-register-p2p.sh                           # default dirs
#   BUILD_DIR=build/debug  test/coverage-register-p2p.sh
#   HTML_DIR=out/cov-html  test/coverage-register-p2p.sh
#   FUNC=ipcRegisterBuffer test/coverage-register-p2p.sh   # scope to one function
#
# HTML output is always produced. Open HTML_DIR/index.html when done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${RCCL_ROOT}/build/release}"
BIN="${BUILD_DIR}/test/rccl-UnitTests"
COV_DIR="${BUILD_DIR}/test/coverage-register-p2p"
HTML_DIR="${HTML_DIR:-${COV_DIR}/html}"
PROFDATA="${COV_DIR}/merged.profdata"

# Prefer ROCm's bundled llvm tooling, fall back to system PATH.
LLVM_BIN="/opt/rocm/llvm/bin"
PROFDATA_TOOL="${LLVM_BIN}/llvm-profdata"
COV_TOOL="${LLVM_BIN}/llvm-cov"
if [[ ! -x "${PROFDATA_TOOL}" ]]; then PROFDATA_TOOL="$(command -v llvm-profdata)"; fi
if [[ ! -x "${COV_TOOL}" ]];      then COV_TOOL="$(command -v llvm-cov)";           fi

if [[ ! -x "${BIN}" ]]; then
  echo "error: ${BIN} not found." >&2
  echo "       Build with: cmake -DBUILD_TESTS=ON -DENABLE_CODE_COVERAGE=ON ..." >&2
  exit 1
fi

mkdir -p "${COV_DIR}"
# Wipe stale profraw files from a previous run so we don't accidentally
# merge old data with the new run.
rm -f "${COV_DIR}"/*.profraw

# ---------------------------------------------------------------------------
# 1. Run the test binary.
#
# LLVM_PROFILE_FILE is set with %p (PID) in the filename so that the parent
# process and every re-exec'd child process each write a distinct profraw file.
# The ProcessIsolatedTestRunner also sets LLVM_PROFILE_FILE in the child
# environment (with no-overwrite), but we set it here first (with overwrite
# disabled) so the parent's own profile ends up in our COV_DIR too. Because
# the runner uses setenv(..., 0) (no-overwrite), the value we set here is
# inherited unchanged by every child.
#
# --gtest_filter scopes to the Register suite only.  Additional gtest flags
# (e.g. --gtest_repeat, --gtest_also_run_disabled_tests) can be appended
# on the command line and are forwarded to the binary.
# ---------------------------------------------------------------------------
echo "==> Running Register.ProcessIsolatedRegisterTests"
LLVM_PROFILE_FILE="${COV_DIR}/rccl_tests_%p_%m.profraw" \
  "${BIN}" --gtest_filter="Register.ProcessIsolatedRegisterTests" "$@" || true

# ---------------------------------------------------------------------------
# 2. Merge all profraw files (parent + all child re-exec processes).
# ---------------------------------------------------------------------------
PROFRAW_FILES=( "${COV_DIR}"/*.profraw )
if [[ ${#PROFRAW_FILES[@]} -eq 0 || ! -f "${PROFRAW_FILES[0]}" ]]; then
  echo "error: no .profraw files found under ${COV_DIR}" >&2
  echo "       Was the binary built with -DENABLE_CODE_COVERAGE=ON?" >&2
  exit 1
fi

echo "==> Merging ${#PROFRAW_FILES[@]} profraw file(s) -> ${PROFDATA}"
"${PROFDATA_TOOL}" merge -sparse "${PROFRAW_FILES[@]}" -o "${PROFDATA}"

# ---------------------------------------------------------------------------
# 3. Source files to include in the coverage report.
#
# p2p_tmp.cc is the hipified version of src/transport/p2p.cc — it is what
# is actually compiled and instrumented, and it contains ipcRegisterBuffer
# where the NCCL GH#1859 bug (and its fix) live.
#
# RegisterTests.cpp is compiled directly (not hipified) and shows which
# test-side branches executed.
# ---------------------------------------------------------------------------
SOURCES=(
  "${BUILD_DIR}/hipify/src/transport/p2p_tmp.cc"
  "${RCCL_ROOT}/test/RegisterTests.cpp"
)

# Optional: scope both the annotated terminal output and the HTML report to a
# single function.  Exact match against the mangled or unmangled symbol name.
# Set via environment, e.g.: FUNC=ipcRegisterBuffer ./coverage-register-p2p.sh
NAME_FLAGS=()
if [[ -n "${FUNC:-}" ]]; then
  NAME_FLAGS+=(--name="${FUNC}")
  echo "==> Scoped to function: ${FUNC}"
fi

# ---------------------------------------------------------------------------
# 4. Text summary (always emitted to stdout).
# ---------------------------------------------------------------------------
echo ""
echo "==> Coverage summary"
"${COV_TOOL}" report "${BIN}" \
  -instr-profile="${PROFDATA}" \
  --show-branch-summary \
  --show-region-summary \
  "${NAME_FLAGS[@]}" \
  "${SOURCES[@]}"

# When scoped to a specific function, also print the annotated source with
# inline branch counts so hit/miss is immediately visible in the terminal.
if [[ -n "${FUNC:-}" ]]; then
  echo ""
  echo "==> Annotated source for ${FUNC}"
  "${COV_TOOL}" show "${BIN}" \
    -instr-profile="${PROFDATA}" \
    --name="${FUNC}" \
    --show-branches=count \
    "${SOURCES[@]}"
fi

# ---------------------------------------------------------------------------
# 5. HTML report (always produced).
#
# --show-branches=count  : inline hit/miss counts next to each branch point.
# --show-regions         : surface region boundaries for multi-condition lines.
# --show-line-counts-or-regions: use region counts where available.
# ---------------------------------------------------------------------------
mkdir -p "${HTML_DIR}"
echo ""
echo "==> Writing HTML report to ${HTML_DIR}"
"${COV_TOOL}" show "${BIN}" \
  -instr-profile="${PROFDATA}" \
  -format=html \
  -output-dir="${HTML_DIR}" \
  -show-line-counts-or-regions \
  -show-regions \
  -show-branches=count \
  --show-region-summary \
  --show-branch-summary \
  "${NAME_FLAGS[@]}" \
  "${SOURCES[@]}"

echo "    Open: ${HTML_DIR}/index.html"
