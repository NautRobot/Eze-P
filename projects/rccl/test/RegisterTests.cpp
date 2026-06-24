/*************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

// Helper to check GPU availability
static bool hasGpuAvailable() {
    int numDevices = 0;
    hipError_t err = hipGetDeviceCount(&numDevices);
    return (err == hipSuccess && numDevices >= 1);
}

// Macro to skip test if no GPU is available
#define SKIP_IF_NO_GPU() \
    do { \
        if (!hasGpuAvailable()) { \
            GTEST_SKIP() << "This test requires at least 1 GPU device."; \
            return; \
        } \
    } while(0)

// Helper to initialize a single-rank communicator
static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

//==============================================================================
// Test implementation functions - parameterized by registration expectation
//==============================================================================

/**
 * @brief Test basic register/deregister of a single buffer
 * @param expectNonNull If true, expect non-NULL handle (registration enabled)
 */
static void testCommRegisterDeregister(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create buffer on device
    const size_t bufferSize = 1024 * 1024; // 1 MB
    void* deviceBuffer = nullptr;
    HIPCALL(hipMalloc(&deviceBuffer, bufferSize));
    ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate device buffer";

    // Register buffer with ncclCommRegister
    void* regHandle = nullptr;
    NCCLCHECK(ncclCommRegister(comm, deviceBuffer, bufferSize, &regHandle));

    // Verify handle based on expected behavior
    if (expectNonNull) {
        EXPECT_NE(regHandle, nullptr)
            << "Buffer registration failed: regHandle is NULL even though NCCL_LOCAL_REGISTER=1";
    } else {
        EXPECT_EQ(regHandle, nullptr)
            << "Expected NULL handle when NCCL_LOCAL_REGISTER is disabled";
    }

    // Deregister and clean up
    NCCLCHECK(ncclCommDeregister(comm, regHandle));
    HIPCALL(hipFree(deviceBuffer));
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering multiple buffers simultaneously
 * @param expectNonNull If true, expect non-NULL handles and verify uniqueness
 */
static void testMultipleBufferRegistration(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create and register multiple buffers
    const int numBuffers = 4;
    const size_t bufferSize = 64 * 1024; // 64 KB each
    void* deviceBuffers[numBuffers] = {nullptr};
    void* regHandles[numBuffers] = {nullptr};

    for (int i = 0; i < numBuffers; i++) {
        HIPCALL(hipMalloc(&deviceBuffers[i], bufferSize));
        ASSERT_NE(deviceBuffers[i], nullptr) << "Failed to allocate buffer " << i;

        NCCLCHECK(ncclCommRegister(comm, deviceBuffers[i], bufferSize, &regHandles[i]));

        if (expectNonNull) {
            EXPECT_NE(regHandles[i], nullptr) << "Registration failed for buffer " << i;
        } else {
            EXPECT_EQ(regHandles[i], nullptr) << "Expected NULL handle for buffer " << i;
        }
    }

    // Verify all handles are unique (only when registration is enabled)
    if (expectNonNull) {
        for (int i = 0; i < numBuffers; i++) {
            for (int j = i + 1; j < numBuffers; j++) {
                if (regHandles[i] != nullptr && regHandles[j] != nullptr) {
                    EXPECT_NE(regHandles[i], regHandles[j])
                        << "Buffers " << i << " and " << j << " have the same registration handle";
                }
            }
        }
    }

    // Deregister and clean up
    for (int i = 0; i < numBuffers; i++) {
        NCCLCHECK(ncclCommDeregister(comm, regHandles[i]));
        HIPCALL(hipFree(deviceBuffers[i]));
    }
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering buffers of various sizes
 * @param expectNonNull If true, expect non-NULL handles for all sizes
 */
static void testVariableSizeBuffers(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Test various buffer sizes: 4KB, 64KB, 1MB, 4MB
    const size_t sizes[] = {4096, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        void* deviceBuffer = nullptr;
        void* regHandle = nullptr;

        HIPCALL(hipMalloc(&deviceBuffer, sizes[i]));
        ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate buffer of size " << sizes[i];

        NCCLCHECK(ncclCommRegister(comm, deviceBuffer, sizes[i], &regHandle));

        if (expectNonNull) {
            EXPECT_NE(regHandle, nullptr)
                << "Registration failed for buffer size " << sizes[i] << " bytes";
        } else {
            EXPECT_EQ(regHandle, nullptr)
                << "Expected NULL handle for buffer size " << sizes[i] << " bytes";
        }

        NCCLCHECK(ncclCommDeregister(comm, regHandle));
        HIPCALL(hipFree(deviceBuffer));
    }

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Regression test for NCCL GH#1859 / v2.29.2-1 fix:
 *        "Fixes crash that can happen when calling p2p and then collectives
 *         while using the same user buffer."
 *
 * Root cause: in p2p.cc the NCCL_IPC_COLLECTIVE branch allocates
 * devPeerRmtAddrs but only copies hostPeerRmtAddrs → devPeerRmtAddrs when
 * needUpdate==true.  If P2P ran first on the same regRecord, needUpdate is
 * already false by the time the collective resolves addresses, so
 * devPeerRmtAddrs is left zeroed and the GPU kernel dereferences a null
 * remote pointer → illegal memory access / crash.
 *
 * Sequence that triggers the bug (requires ≥ 2 intra-node GPUs):
 *   1. ncclCommInitAll across 2 ranks (one per GPU).
 *   2. ncclCommRegister the same device buffer on each rank
 *      (NCCL_LOCAL_REGISTER=1 ensures real IPC registration).
 *   3. ncclSend / ncclRecv group call on the registered buffer → sync.
 *      (populates hostPeerRmtAddrs; leaves devPeerRmtAddrs NULL for the
 *       collective branch because only the P2P branch ran.)
 *   4. ncclAllReduce group call on the same registered buffer → sync.
 *      (buggy: allocates devPeerRmtAddrs, skips memcpy, crashes;
 *       fixed:  allocates devPeerRmtAddrs and always copies on first alloc.)
 *   5. Verify ncclSuccess and correct AllReduce output.
 *
 * Skipped when fewer than 2 GPUs are visible (intra-node IPC path requires
 * at least 2 ranks on the same node).
 */
static void testP2pThenCollectiveSameBuffer()
{
    // Need at least 2 GPUs for an intra-node IPC communicator.
    int numDevices = 0;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 2) {
        GTEST_SKIP() << "This test requires at least 2 GPU devices (detected "
                     << numDevices << ").";
        return;
    }
    const int numRanks = 2;

    // --- Communicator setup ---
    std::vector<ncclComm_t> comms(numRanks, nullptr);
    NCCLCHECK(ncclCommInitAll(comms.data(), numRanks, nullptr));

    // --- Per-rank resources ---
    const size_t numElements = 1024;                        // 4 KB of float
    const size_t bufBytes    = numElements * sizeof(float);

    std::vector<hipStream_t> streams(numRanks);
    std::vector<void*>       devBufs(numRanks, nullptr);    // device buffers
    std::vector<void*>       regHandles(numRanks, nullptr); // ncclCommRegister handles
    std::vector<void*>       devOut(numRanks, nullptr);     // separate output for AllReduce

    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamCreate(&streams[r]));

        // Input buffer: filled with rank+1 so AllReduce sum = numRanks*(numRanks+1)/2 per element.
        HIPCALL(hipMalloc(&devBufs[r], bufBytes));
        std::vector<float> hostIn(numElements, static_cast<float>(r + 1));
        HIPCALL(hipMemcpy(devBufs[r], hostIn.data(), bufBytes, hipMemcpyHostToDevice));

        // Separate output buffer for AllReduce (out-of-place keeps the test readable).
        HIPCALL(hipMalloc(&devOut[r], bufBytes));
        HIPCALL(hipMemset(devOut[r], 0, bufBytes));

        // Register the input buffer — this is the buffer that will be used for
        // both the P2P and the subsequent AllReduce.
        NCCLCHECK(ncclCommRegister(comms[r], devBufs[r], bufBytes, &regHandles[r]));
        EXPECT_NE(regHandles[r], nullptr)
            << "Rank " << r << ": ncclCommRegister returned NULL handle "
               "(is NCCL_LOCAL_REGISTER=1 set?).";
    }

    // --- Step 1: P2P send/recv on the registered buffer ---
    // Rank 0 sends devBufs[0] → rank 1.  Rank 1 receives into devBufs[1]
    // (overwriting it, which is fine — we only care about the AllReduce result).
    // This populates regRecord->regIpcAddrs.hostPeerRmtAddrs for the P2P path
    // but leaves the NCCL_IPC_COLLECTIVE devPeerRmtAddrs uninitialised.
    NCCLCHECK(ncclGroupStart());
    HIPCALL(hipSetDevice(0));
    NCCLCHECK(ncclSend(devBufs[0], numElements, ncclFloat, /*peer=*/1, comms[0], streams[0]));
    HIPCALL(hipSetDevice(1));
    NCCLCHECK(ncclRecv(devBufs[1], numElements, ncclFloat, /*peer=*/0, comms[1], streams[1]));
    NCCLCHECK(ncclGroupEnd());

    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamSynchronize(streams[r]));
    }

    // Reset the output buffers before the AllReduce so we can verify the result.
    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipMemset(devOut[r], 0, bufBytes));
    }

    // --- Step 2: AllReduce on the *same* registered buffer (bug trigger) ---
    // Without the fix this crashes with an illegal memory access because
    // devPeerRmtAddrs is allocated but not populated (skipped memcpy).
    // With the fix devPeerRmtAddrs is populated on first allocation regardless
    // of needUpdate, so the GPU kernel receives valid remote addresses.
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclAllReduce(devBufs[r], devOut[r], numElements,
                                ncclFloat, ncclSum, comms[r], streams[r]));
    }
    NCCLCHECK(ncclGroupEnd());

    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        HIPCALL(hipStreamSynchronize(streams[r]));
    }

    // --- Verify AllReduce result ---
    // After the P2P, devBufs[0] still holds 1.0f (rank 0 sent but did not
    // overwrite its own buffer) and devBufs[1] now holds 1.0f (received from
    // rank 0).  Sum across both ranks = 2.0f per element.
    const float expectedSum = static_cast<float>(numRanks); // 1.0f * numRanks
    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        std::vector<float> hostOut(numElements, -1.0f);
        HIPCALL(hipMemcpy(hostOut.data(), devOut[r], bufBytes, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < numElements; i++) {
            EXPECT_FLOAT_EQ(hostOut[i], expectedSum)
                << "Rank " << r << " element [" << i << "]: "
                << "expected " << expectedSum << " got " << hostOut[i];
            if (hostOut[i] != expectedSum) break; // report first mismatch only
        }
    }

    // --- Cleanup ---
    for (int r = 0; r < numRanks; r++) {
        HIPCALL(hipSetDevice(r));
        NCCLCHECK(ncclCommDeregister(comms[r], regHandles[r]));
        HIPCALL(hipFree(devBufs[r]));
        HIPCALL(hipFree(devOut[r]));
        HIPCALL(hipStreamDestroy(streams[r]));
        NCCLCHECK(ncclCommDestroy(comms[r]));
    }
}

/**
 * @brief Test deregistering NULL handle (should succeed as no-op)
 */
static void testDeregisterNullHandle() {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Deregister NULL handle - should be a no-op
    NCCLCHECK(ncclCommDeregister(comm, nullptr));

    NCCLCHECK(ncclCommDestroy(comm));
}

//==============================================================================
// Test configuration helpers
//==============================================================================

// Environment configuration for explicitly disabled local registration
static ProcessIsolatedTestRunner::TestConfig
makeDisabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "0"}});
}

// Environment configuration for enabled registration
static ProcessIsolatedTestRunner::TestConfig
makeEnabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "1"}});
}

/**
 * @brief Test ncclCommRegister and ncclCommDeregister APIs with process isolation
 *
 * This test suite verifies that:
 * 1. A device buffer can be registered with ncclCommRegister (API returns success)
 * 2. When NCCL_LOCAL_REGISTER=1, the registration returns a valid (non-NULL) handle
 * 3. When NCCL_LOCAL_REGISTER=0, NULL handle is expected (local registration off)
 * 4. The buffer can be deregistered with ncclCommDeregister
 *
 * Note: On newer HIP, NCCL_LOCAL_REGISTER defaults to 1 when unset; the "disabled"
 * cases therefore set NCCL_LOCAL_REGISTER=0 explicitly rather than clearing the var.
 */
TEST(Register, ProcessIsolatedRegisterTests)
{
    RUN_ISOLATED_TESTS(
        // CommRegisterDeregister tests
        makeDisabledConfig("CommRegisterDeregister_Disabled",
            []() { testCommRegisterDeregister(false); }),
        makeEnabledConfig("CommRegisterDeregister_Enabled",
            []() { testCommRegisterDeregister(true); }),

        // MultipleBufferRegistration tests
        makeDisabledConfig("MultipleBufferRegistration_Disabled",
            []() { testMultipleBufferRegistration(false); }),
        makeEnabledConfig("MultipleBufferRegistration_Enabled",
            []() { testMultipleBufferRegistration(true); }),

        // VariableSizeBuffers tests
        makeDisabledConfig("VariableSizeBuffers_Disabled",
            []() { testVariableSizeBuffers(false); }),
        makeEnabledConfig("VariableSizeBuffers_Enabled",
            []() { testVariableSizeBuffers(true); }),

        // DeregisterNullHandle test (no enable/disable variants needed)
        ProcessIsolatedTestRunner::TestConfig("DeregisterNullHandle", testDeregisterNullHandle),

        // Regression test: NCCL GH#1859 — p2p followed by collective on the same
        // registered user buffer must not crash (requires ≥ 2 GPUs).
        ProcessIsolatedTestRunner::TestConfig(
            "P2pThenCollective_SameBuffer", testP2pThenCollectiveSameBuffer)
            .withEnvironment({{"NCCL_LOCAL_REGISTER", "1"}})
            .withNumGpus(2)
            .withTimeout(std::chrono::seconds(120))
    );
}

} // namespace RcclUnitTesting
