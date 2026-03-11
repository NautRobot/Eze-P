#!/bin/bash
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-./build}"

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

ROCDECODE_TEST_DIR="${ROCM_PATH}/share/rocdecode/test"
ROCJPEG_TEST_DIR="${ROCM_PATH}/share/rocjpeg/test"

RESULT=0

if [ -d "${ROCDECODE_TEST_DIR}" ]; then
  echo "=== Configuring rocdecode CTests ==="
  cmake -B rocdecode-test \
    -DROCM_PATH="${ROCM_PATH}" \
    "${ROCDECODE_TEST_DIR}"

  echo "=== Building and running rocdecode CTests ==="
  if ! ctest --test-dir rocdecode-test --output-on-failure --timeout 600; then
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
  if ! ctest --test-dir rocjpeg-test --output-on-failure --timeout 600; then
    echo "ERROR: rocjpeg CTests failed"
    RESULT=1
  fi
else
  echo "WARNING: rocjpeg test directory not found at ${ROCJPEG_TEST_DIR}"
  RESULT=1
fi

exit ${RESULT}
