/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Standalone reproducer for the NCCL GH#1859 sequence:
 *   ncclCommInitRank -> ncclCommRegister -> ncclSend/ncclRecv -> ncclAllReduce
 * all on the same registered user buffer.
 *
 * This mirrors the Register.ProcessIsolatedRegisterTests/P2pThenCollective_SameBuffer
 * unit test, but as a plain program linked directly against build/debug/librccl.so.
 * The unit-test harness (ProcessIsolatedTestRunner) re-execs each sub-test and
 * pipes its output back, which swallows RCCL's NCCL_DEBUG=INFO trace lines.
 * Running standalone lets you actually see the REG/P2P trace and confirm whether
 * the IPC registration path (ipcRegisterBuffer in p2p.cc) is taken.
 *
 * Drive it with test/run-p2pcoll.sh which sets all required env vars, or:
 *   NCCL_LOCAL_REGISTER=1 NCCL_P2P_LL_THRESHOLD=0 NCCL_LEGACY_CUDA_REGISTER=1 \
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG,P2P ./p2pcoll
 *
 * --- Why fork() + ncclCommInitRank instead of ncclCommInitAll ---
 *
 * comm->directMode is set true by ncclTransportCheckP2pType when any two local
 * ranks share the same pid. ipcRegisterBuffer's legacy IPC path then hits:
 *   if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
 * and returns *regBufFlag=0, bypassing lines 1188-1222 (the bug fix) entirely.
 * ncclCommInitAll runs all ranks in one process, so directMode is always true.
 *
 * We therefore fork() one child per rank. Each child is its own process so
 * directMode is false. This matches what ProcessIsolatedTestRunner does.
 *
 * --- Why the parent must not touch HIP/NCCL before fork() ---
 *
 * hipGetDeviceCount and ncclGetUniqueId both initialize the HIP/HSA runtime.
 * The HSA runtime is not fork-safe: children inherit broken internal state
 * (locks, file descriptors, memory mappings) that causes hipHostMalloc to
 * fail with "out of memory" on the very first allocation inside
 * ncclCommInitRank. To avoid this we do NO HIP or NCCL calls in the parent.
 *
 * The ncclUniqueId is shared from rank-0's child to the others via a
 * mmap(MAP_SHARED|MAP_ANONYMOUS) region set up before any fork().
 *
 * --- Protocol selection ---
 *
 * addP2pToPlan (enqueue.cc) selects LL protocol when:
 *   bytes <= nChannels * NCCL_P2P_LL_THRESHOLD  (default 8192)
 * ncclRegisterP2pIpcBuffer (-> ipcRegisterBuffer) is only called when
 * protocol == NCCL_PROTO_SIMPLE. We use ne = 16*1024 floats (65536 bytes)
 * to exceed the threshold for any plausible channel count, and also set
 * NCCL_P2P_LL_THRESHOLD=0 in run-p2pcoll.sh for an explicit guarantee.
 *
 * --- Hardware requirement ---
 *
 * The IPC registration path requires direct GPU-to-GPU P2P. On PCIe-only
 * hardware RCCL falls back to the SHM transport and ipcRegisterBuffer is
 * never called.
 */

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// Shared memory layout written by rank-0 child before others proceed.
struct SharedBootstrap {
    ncclUniqueId id;
    volatile int state;  // 0 = not ready, 1 = ready, -1 = fatal error in rank 0
};

#define HC(x)                                                              \
    do {                                                                   \
        hipError_t _e = (x);                                               \
        if (_e != hipSuccess) {                                            \
            printf("[rank %d] HIP error %d (%s) @ %s:%d\n", rank, _e,    \
                   hipGetErrorString(_e), __FILE__, __LINE__);             \
            return 1;                                                      \
        }                                                                  \
    } while (0)

#define NC(x)                                                              \
    do {                                                                   \
        ncclResult_t _e = (x);                                             \
        if (_e != ncclSuccess) {                                           \
            printf("[rank %d] NCCL error %d (%s) @ %s:%d\n", rank, _e,   \
                   ncclGetErrorString(_e), __FILE__, __LINE__);            \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static int run_rank(int rank, int nranks, SharedBootstrap* shared)
{
    // Must exceed nChannels * NCCL_P2P_LL_THRESHOLD (default 8192 bytes) so
    // that addP2pToPlan selects NCCL_PROTO_SIMPLE rather than NCCL_PROTO_LL.
    // Only with SIMPLE does it enter the ncclRegisterP2pIpcBuffer branch that
    // leads to ipcRegisterBuffer.
    const size_t ne    = 16 * 1024;
    const size_t bytes = ne * sizeof(float);

    if (rank == 0) {
        // First HIP/NCCL calls happen here, inside the child — no inherited
        // HIP state from the parent.
        int numDevices = 0;
        HC(hipGetDeviceCount(&numDevices));
        if (numDevices < nranks) {
            printf("Requires %d GPUs (detected %d).\n", nranks, numDevices);
            shared->state = -1;
            return 0;
        }

        int canAccess01 = 0, canAccess10 = 0;
        HC(hipDeviceCanAccessPeer(&canAccess01, 0, 1));
        HC(hipDeviceCanAccessPeer(&canAccess10, 1, 0));
        printf("hipDeviceCanAccessPeer 0->1=%d 1->0=%d%s\n", canAccess01, canAccess10,
               (canAccess01 && canAccess10)
                   ? ""
                   : "  (no direct P2P: SHM transport will be used, ipcRegisterBuffer NOT exercised)");

        // Generate the unique ID and publish it to the other children.
        NC(ncclGetUniqueId(&shared->id));
        __sync_synchronize();
        shared->state = 1;
    } else {
        // Wait for rank 0 to publish the ID.
        while (shared->state == 0) { /* spin */ }
        __sync_synchronize();
        if (shared->state < 0) return 0;  // rank 0 reported a fatal error
    }

    ncclUniqueId id = shared->id;

    HC(hipSetDevice(rank));

    ncclComm_t comm;
    NC(ncclCommInitRank(&comm, nranks, id, rank));

    hipStream_t s;
    HC(hipStreamCreate(&s));

    void *in, *out;
    HC(hipMalloc(&in,  bytes));
    HC(hipMalloc(&out, bytes));

    std::vector<float> hi(ne, static_cast<float>(rank + 1));
    HC(hipMemcpy(in, hi.data(), bytes, hipMemcpyHostToDevice));
    HC(hipMemset(out, 0, bytes));

    void* h = nullptr;
    NC(ncclCommRegister(comm, in, bytes, &h));
    printf("[rank %d] reg handle %p\n", rank, h);

    printf("[rank %d] === P2P phase ===\n", rank);
    NC(ncclGroupStart());
    if (rank == 0) NC(ncclSend(in, ne, ncclFloat, 1, comm, s));
    if (rank == 1) NC(ncclRecv(in, ne, ncclFloat, 0, comm, s));
    NC(ncclGroupEnd());
    HC(hipStreamSynchronize(s));

    printf("[rank %d] === AllReduce phase ===\n", rank);
    NC(ncclGroupStart());
    NC(ncclAllReduce(in, out, ne, ncclFloat, ncclSum, comm, s));
    NC(ncclGroupEnd());
    HC(hipStreamSynchronize(s));

    // After the P2P send, in[1] == 1.0 (received from rank 0), so both ranks
    // have in[r] == 1.0. The AllReduce sum is 1.0 + 1.0 = 2.0.
    const float expected = static_cast<float>(nranks);
    bool ok = true;
    std::vector<float> host(ne, -1.0f);
    HC(hipMemcpy(host.data(), out, bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < ne; i++) {
        if (host[i] != expected) {
            printf("[rank %d] MISMATCH elem %zu: expected %f got %f\n",
                   rank, i, expected, host[i]);
            ok = false;
            break;
        }
    }
    printf("[rank %d] === done (result %s) ===\n", rank, ok ? "OK" : "WRONG");

    NC(ncclCommDeregister(comm, h));
    HC(hipFree(in));
    HC(hipFree(out));
    HC(hipStreamDestroy(s));
    NC(ncclCommDestroy(comm));

    return ok ? 0 : 1;
}

int main()
{
    const int n = 2;

    // Set up shared memory BEFORE fork and BEFORE any HIP/NCCL call.
    // The HIP/HSA runtime is not fork-safe: initializing it in the parent
    // leaves the children with broken internal state (locks, fd's, mappings),
    // causing hipHostMalloc to fail inside ncclCommInitRank.
    // Rank-0's child generates the ncclUniqueId here and signals the others.
    SharedBootstrap* shared = (SharedBootstrap*)mmap(
        nullptr, sizeof(SharedBootstrap),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED) { perror("mmap"); return 1; }
    shared->state = 0;

    pid_t pids[n];
    for (int r = 0; r < n; r++) {
        pids[r] = fork();
        if (pids[r] < 0) { perror("fork"); return 1; }
        if (pids[r] == 0) {
            int rc = run_rank(r, n, shared);
            _exit(rc);
        }
    }

    int overall = 0;
    for (int r = 0; r < n; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            overall = 1;
    }
    munmap(shared, sizeof(SharedBootstrap));
    printf("=== overall result: %s ===\n", overall == 0 ? "OK" : "WRONG");
    return overall;
}
