/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Standalone reproducer for the NCCL GH#1859 sequence:
 *   ncclCommInitAll -> ncclCommRegister -> ncclSend/ncclRecv -> ncclAllReduce
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
 *
 * Note: the IPC registration path is only reached when the two GPUs have direct
 * GPU-to-GPU P2P. On PCIE-only hardware RCCL falls back to the SHM transport and
 * ipcRegisterBuffer is never called.
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
 */

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define HC(x)                                                            \
    do {                                                                 \
        hipError_t e = (x);                                              \
        if (e != hipSuccess) {                                           \
            printf("HIP error %d (%s) @ %s:%d\n", e, hipGetErrorString(e), \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

#define NC(x)                                                            \
    do {                                                                 \
        ncclResult_t e = (x);                                            \
        if (e != ncclSuccess) {                                          \
            printf("NCCL error %d (%s) @ %s:%d\n", e,                    \
                   ncclGetErrorString(e), __FILE__, __LINE__);          \
            return 1;                                                    \
        }                                                                \
    } while (0)

int main()
{
    const int    n     = 2;
    // Must exceed nChannels * NCCL_P2P_LL_THRESHOLD (default 8192 bytes) so that
    // addP2pToPlan selects NCCL_PROTO_SIMPLE rather than NCCL_PROTO_LL.  Only
    // with SIMPLE does it enter the ncclRegisterP2pIpcBuffer branch that leads
    // to ipcRegisterBuffer.  16k floats = 65536 bytes is safely above any
    // realistic channel-count multiple of the 8192-byte threshold.
    const size_t ne    = 16 * 1024;
    const size_t bytes = ne * sizeof(float);

    int numDevices = 0;
    HC(hipGetDeviceCount(&numDevices));
    if (numDevices < n) {
        printf("This reproducer requires at least %d GPUs (detected %d).\n", n, numDevices);
        return 0;
    }

    // Report direct-P2P capability between the two GPUs. If either direction is
    // 0, RCCL will fall back to SHM and ipcRegisterBuffer will not be exercised.
    int canAccess01 = 0, canAccess10 = 0;
    HC(hipDeviceCanAccessPeer(&canAccess01, 0, 1));
    HC(hipDeviceCanAccessPeer(&canAccess10, 1, 0));
    printf("hipDeviceCanAccessPeer 0->1=%d 1->0=%d%s\n", canAccess01, canAccess10,
           (canAccess01 && canAccess10)
               ? ""
               : "  (no direct P2P: collective will use SHM, ipcRegisterBuffer NOT exercised)");

    std::vector<ncclComm_t> comms(n);
    NC(ncclCommInitAll(comms.data(), n, nullptr));

    std::vector<hipStream_t> s(n);
    std::vector<void*>       in(n), out(n), h(n);

    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        HC(hipStreamCreate(&s[r]));

        HC(hipMalloc(&in[r], bytes));
        std::vector<float> hi(ne, static_cast<float>(r + 1));
        HC(hipMemcpy(in[r], hi.data(), bytes, hipMemcpyHostToDevice));

        HC(hipMalloc(&out[r], bytes));
        HC(hipMemset(out[r], 0, bytes));

        NC(ncclCommRegister(comms[r], in[r], bytes, &h[r]));
        printf("rank %d reg handle %p\n", r, h[r]);
    }

    printf("=== P2P phase ===\n");
    NC(ncclGroupStart());
    HC(hipSetDevice(0));
    NC(ncclSend(in[0], ne, ncclFloat, 1, comms[0], s[0]));
    HC(hipSetDevice(1));
    NC(ncclRecv(in[1], ne, ncclFloat, 0, comms[1], s[1]));
    NC(ncclGroupEnd());
    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        HC(hipStreamSynchronize(s[r]));
    }

    printf("=== AllReduce phase ===\n");
    NC(ncclGroupStart());
    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        NC(ncclAllReduce(in[r], out[r], ne, ncclFloat, ncclSum, comms[r], s[r]));
    }
    NC(ncclGroupEnd());
    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        HC(hipStreamSynchronize(s[r]));
    }

    // Verify the AllReduce result: in[0]=1.0, in[1] received 1.0 from rank 0
    // (P2P send clobbers in[1] with rank-0's value of 1.0), so the sum is
    // 1.0 + 1.0 = 2.0 per element on every rank.
    const float expected = static_cast<float>(n);  // 1.0f * n
    bool        ok       = true;
    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        std::vector<float> host(ne, -1.0f);
        HC(hipMemcpy(host.data(), out[r], bytes, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < ne; i++) {
            if (host[i] != expected) {
                printf("MISMATCH rank %d elem %zu: expected %f got %f\n", r, i, expected, host[i]);
                ok = false;
                break;
            }
        }
    }
    printf("=== done (result %s) ===\n", ok ? "OK" : "WRONG");

    for (int r = 0; r < n; r++) {
        HC(hipSetDevice(r));
        NC(ncclCommDeregister(comms[r], h[r]));
        HC(hipFree(in[r]));
        HC(hipFree(out[r]));
        HC(hipStreamDestroy(s[r]));
        NC(ncclCommDestroy(comms[r]));
    }

    return ok ? 0 : 1;
}
