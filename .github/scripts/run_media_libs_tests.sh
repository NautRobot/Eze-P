#!/bin/bash
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-./build}"
ROCM_PATH="$(cd "${ROCM_PATH}" && pwd)"

echo "=== Media Libs Test ==="
echo "ROCM_PATH: ${ROCM_PATH}"

if [ ! -d "${ROCM_PATH}" ]; then
  echo "ERROR: ROCM_PATH not found at ${ROCM_PATH}"
  exit 1
fi

export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
if [ -d "${ROCM_PATH}/lib/rocm_sysdeps/lib" ]; then
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}"
fi

echo "=== Library core dependency check ==="
for lib in librocdecode.so librocjpeg.so; do
  LIB_PATH="${ROCM_PATH}/lib/${lib}"
  if [ -f "${LIB_PATH}" ]; then
    echo "--- ldd ${lib} ---"
    readelf -d "${LIB_PATH}"
  else
    echo "ERROR: ${lib} not found at ${LIB_PATH}"
    exit 1
  fi
done

echo "=== Library dependency check ==="
for lib in librocdecode.so librocjpeg.so; do
  LIB_PATH="${ROCM_PATH}/lib/${lib}"
  if [ -f "${LIB_PATH}" ]; then
    echo "--- ldd ${lib} ---"
    ldd "${LIB_PATH}"
  else
    echo "ERROR: ${lib} not found at ${LIB_PATH}"
    exit 1
  fi
done

ROCDECODE_TEST_DIR="${ROCM_PATH}/share/rocdecode/test"
ROCJPEG_TEST_DIR="${ROCM_PATH}/share/rocjpeg/test"

RESULT=0

if [ -d "${ROCDECODE_TEST_DIR}" ]; then
  echo "=== Configuring rocdecode CTests ==="
  cmake -B rocdecode-test \
    -DROCM_PATH="${ROCM_PATH}" \
    "${ROCDECODE_TEST_DIR}"

  echo "=== Building and running rocdecode CTests ==="
  # TBD - Currently blocked by driver upgrade on CI machines
  #if ! ctest --test-dir rocdecode-test -VV --output-on-failure --timeout 600; then
  if ! ctest --test-dir rocdecode-test -N; then
    echo "ERROR: rocdecode CTests failed"
    RESULT=1
  fi
else
  echo "WARNING: rocdecode test directory not found at ${ROCDECODE_TEST_DIR}"
  RESULT=1
fi

if [ -d "${ROCJPEG_TEST_DIR}" ]; then
  echo "=== Configuring rocjpeg CTests ==="
  cmake -B rocjpeg-test \
    -DROCM_PATH="${ROCM_PATH}" \
    "${ROCJPEG_TEST_DIR}"

  echo "=== Building and running rocjpeg CTests ==="
  # TBD - Currently blocked by driver upgrade on CI machines
  #if ! ctest --test-dir rocjpeg-test -VV --output-on-failure --timeout 600; then
  if ! ctest --test-dir rocjpeg-test -N; then
    echo "ERROR: rocjpeg CTests failed"
    RESULT=1
  fi
else
  echo "WARNING: rocjpeg test directory not found at ${ROCJPEG_TEST_DIR}"
  RESULT=1
fi

exit ${RESULT}
