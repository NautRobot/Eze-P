#!/bin/bash
#SBATCH --job-name=rccl-register-baseline
#SBATCH --nodes=1
#SBATCH --nodelist=ctr-navi4x-aj53-ws08
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

RCCL_DIR="$HOME/Work/ROCm/rocm-systems/projects/rccl"
BRANCH="laptop/users/mahughes/aicomrccl-1100-p2p-userbuffer-test"
JOBS=$(nproc)

echo "=== $(date) ==="
echo "Node: $(hostname)"
echo "GPUs:"
rocm-smi --showid 2>/dev/null | grep -E 'GPU\[|Device Name'
echo ""

# --- Checkout ---
echo "=== Checking out branch: $BRANCH ==="
cd "$HOME/Work/ROCm/rocm-systems"
git fetch alola2
git checkout -B "$BRANCH" "alola2/$BRANCH"
git log --oneline -3
echo ""

# --- Build ---
echo "=== Building (local arch, with tests, Release) ==="
cd "$RCCL_DIR"
./install.sh -l -t -j "$JOBS" 2>&1

echo ""
echo "=== Build complete ==="
ls -lh build/release/test/rccl-UnitTests

# --- Baseline run ---
echo ""
echo "=== Running Register suite (baseline) ==="
NCCL_DEBUG=INFO \
NCCL_LOCAL_REGISTER=1 \
  build/release/test/rccl-UnitTests \
    --gtest_filter="Register.*" \
    --gtest_color=yes \
    2>&1

echo ""
echo "=== $(date) — done ==="
