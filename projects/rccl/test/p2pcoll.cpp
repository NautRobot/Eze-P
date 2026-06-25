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
 * Drive it with:
 *   NCCL_LOCAL_REGISTER=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG,P2P ./p2pcoll
 * or use test/run-p2pcoll.sh which sets all required env vars.
 *
 * IMPORTANT - each rank must run in its own process.
 * ncclTransportCheckP2pType sets comm->directMode=true when any two local ranks
 * share the same pid. ipcRegisterBuffer then hits:
 *   if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
 * and returns *regBufFlag=0, bypassing the IPC registration path entirely.
 * We therefore fork() one child per rank and use ncclCommInitRank with a
 * shared bootstrap address, mirroring what ProcessIsolatedTestRunner does.
 *
 * Protocol selection (enqueue.cc addP2pToPlan):
 *   protoLL &= bytes <= nChannels * NCCL_P2P_LL_THRESHOLD   (default 8192)
 *   protocol = protoLL ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE
 *
 * ncclRegisterP2pIpcBuffer (and therefore ipcRegisterBuffer) is only called when
 * protocol == NCCL_PROTO_SIMPLE.  A small payload (e.g. 1024 floats = 4096 bytes)
 * stays below the 8192-byte LL threshold and always selects LL, bypassing the
 * IPC registration branch entirely.  We must use a payload that exceeds
 * nChannels * 8192.  ne = 16*1024 floats (65536 bytes) is well above the
 * threshold for any plausible channel count.  We also set NCCL_P2P_LL_THRESHOLD=0
 * in run-p2pcoll.sh to make the intent explicit and guard against future
 * threshold changes.
 *
 * Note: the IPC registration path is only reached when the two GPUs have direct
 * GPU-to-GPU P2P. On PCIE-only hardware RCCL falls back to the SHM transport and
 * ipcRegisterBuffer is never called.
 */

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#define HC(x)                                                            \
    do {                                                                 \
        hipError_t e = (x);                                              \
        if (e != hipSuccess) {                                           \
            printf("[rank %d] HIP error %d (%s) @ %s:%d\n", rank, e,   \
                   hipGetErrorString(e), __FILE__, __LINE__);            \
            return 1;                                                    \
        }                                                                \
    } while (0)

#define NC(x)                                                            \
    do {                                                                 \
        ncclResult_t e = (x);                                            \
        if (e != ncclSuccess) {                                          \
            printf("[rank %d] NCCL error %d (%s) @ %s:%d\n", rank, e,  \
                   ncclGetErrorString(e), __FILE__, __LINE__);           \
            return 1;                                                    \
        }                                                                \
    } while (0)

static int run_rank(int rank, int nranks, const char* id_str)
{
    const size_t ne    = 16 * 1024;  // see protocol-selection note in file header
    const size_t bytes = ne * sizeof(float);

    HC(hipSetDevice(rank));

    // Report direct-P2P capability on rank 0.
    if (rank == 0) {
        int canAccess01 = 0, canAccess10 = 0;
        HC(hipDeviceCanAccessPeer(&canAccess01, 0, 1));
        HC(hipDeviceCanAccessPeer(&canAccess10, 1, 0));
        printf("hipDeviceCanAccessPeer 0->1=%d 1->0=%d%s\n", canAccess01, canAccess10,
               (canAccess01 && canAccess10)
                   ? ""
                   : "  (no direct P2P: collective will use SHM, ipcRegisterBuffer NOT exercised)");
    }

    ncclUniqueId id;
    memcpy(&id, id_str, sizeof(id));

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

    int numDevices = 0;
    hipGetDeviceCount(&numDevices);
    if (numDevices < n) {
        printf("This reproducer requires at least %d GPUs (detected %d).\n", n, numDevices);
        return 0;
    }

    // Generate the unique ID in the parent so all children share it.
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    // Pass it to children as a raw byte string via a shared pipe-free approach:
    // simply embed it in a char array and let each child inherit it via fork().
    char id_buf[sizeof(id)];
    memcpy(id_buf, &id, sizeof(id));

    // Fork one child per rank. Each child runs in its own process so that
    // comm->directMode is false (directMode is set when two local ranks share
    // the same pid, which would block the IPC registration path).
    pid_t pids[n];
    for (int r = 0; r < n; r++) {
        pids[r] = fork();
        if (pids[r] < 0) {
            perror("fork");
            return 1;
        }
        if (pids[r] == 0) {
            // Child: run this rank and exit with its return code.
            int rc = run_rank(r, n, id_buf);
            _exit(rc);
        }
    }

    // Parent: wait for all children and report overall result.
    int overall = 0;
    for (int r = 0; r < n; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            overall = 1;
    }
    printf("=== overall result: %s ===\n", overall == 0 ? "OK" : "WRONG");
    return overall;
}
