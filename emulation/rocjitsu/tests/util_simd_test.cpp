// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_simd_test.cpp
/// @brief Generic-core unit tests for util::simd primitives. Exercises the
/// header-only helpers (`load_or_broadcast`, `masked_store`, `blit_to_buffer`,
/// `force_scalar`) without any rocjitsu Operand/Wavefront fixture.

#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <thread>

#if !UTIL_HAS_STDX_SIMD

TEST(UtilSimd, ToolchainMissingStdxSimd) {
  GTEST_SKIP() << "<experimental/simd> unavailable; util::simd helpers skipped.";
}

#else

namespace {

constexpr std::size_t kW = util::simd::native_width_v<uint32_t>;

TEST(UtilSimd, LoadOrBroadcast_Contiguous_U32) {
  alignas(util::simd::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = static_cast<uint32_t>(0xDEAD'0000u + i);
  auto v = util::simd::load_or_broadcast<uint32_t>(src, 0u);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], src[i]) << "lane " << i;
}

TEST(UtilSimd, LoadOrBroadcast_Contiguous_F32) {
  alignas(util::simd::native<float>) uint32_t src[kW];
  std::array<float, kW> expected{};
  for (std::size_t i = 0; i < kW; ++i) {
    expected[i] = 1.5f * static_cast<float>(i) - 7.25f;
    src[i] = std::bit_cast<uint32_t>(expected[i]);
  }
  auto v = util::simd::load_or_broadcast<float>(src, 0u);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
}

TEST(UtilSimd, LoadOrBroadcast_NullBroadcast_U32) {
  constexpr uint32_t kBits = 0xCAFEBABEu;
  auto v = util::simd::load_or_broadcast<uint32_t>(nullptr, kBits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kBits) << "lane " << i;
}

TEST(UtilSimd, LoadOrBroadcast_NullBroadcast_F32) {
  constexpr float kVal = -3.14159f;
  const uint32_t bits = std::bit_cast<uint32_t>(kVal);
  auto v = util::simd::load_or_broadcast<float>(nullptr, bits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kVal) << "lane " << i;
}

TEST(UtilSimd, MaskedStore_FullMask) {
  alignas(util::simd::native<uint32_t>) uint32_t dst[kW] = {};
  alignas(util::simd::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) src[i] = 0xA5A5'0000u + i;
  auto v = util::simd::load_or_broadcast<uint32_t>(src, 0u);
  const uint64_t full = (kW >= 64) ? ~0ULL : ((1ULL << kW) - 1ULL);
  util::simd::masked_store<uint32_t>(dst, v, full);
  EXPECT_EQ(0, std::memcmp(dst, src, sizeof(dst)));
}

TEST(UtilSimd, MaskedStore_PartialMask) {
  alignas(util::simd::native<uint32_t>) uint32_t dst[kW];
  alignas(util::simd::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0x1111'1111u;
    src[i] = 0x2222'0000u + i;
  }
  auto v = util::simd::load_or_broadcast<uint32_t>(src, 0u);
  // 0xA... pattern: enable every other lane starting from bit 1.
  uint64_t mask = 0;
  for (std::size_t i = 1; i < kW; i += 2) mask |= (1ULL << i);
  util::simd::masked_store<uint32_t>(dst, v, mask);
  for (std::size_t i = 0; i < kW; ++i) {
    if (mask & (1ULL << i))
      EXPECT_EQ(dst[i], src[i]) << "lane " << i << " should be written";
    else
      EXPECT_EQ(dst[i], 0x1111'1111u) << "lane " << i << " should be preserved";
  }
}

TEST(UtilSimd, MaskedStore_EmptyMask) {
  alignas(util::simd::native<uint32_t>) uint32_t dst[kW];
  alignas(util::simd::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0xDEAD'BEEFu;
    src[i] = 0x0BAD'F00Du;
  }
  auto v = util::simd::load_or_broadcast<uint32_t>(src, 0u);
  util::simd::masked_store<uint32_t>(dst, v, 0ULL);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(dst[i], 0xDEAD'BEEFu) << "lane " << i;
}

TEST(UtilSimd, BlitToBuffer_F32) {
  alignas(util::simd::native<float>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = std::bit_cast<uint32_t>(2.0f * static_cast<float>(i) + 0.5f);
  auto v = util::simd::load_or_broadcast<float>(src, 0u);
  alignas(util::simd::native<float>) uint32_t buf[util::simd::native<float>::size()] = {};
  util::simd::blit_to_buffer<float>(buf, v);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(buf[i], src[i]) << "lane " << i;
}

TEST(UtilSimd, ForceScalar_ThreadLocal) {
  util::simd::force_scalar() = true;
  ASSERT_TRUE(util::simd::force_scalar());

  std::atomic<bool> other_thread_saw_true{true};
  std::thread t([&]() {
    other_thread_saw_true.store(util::simd::force_scalar());
  });
  t.join();

  EXPECT_FALSE(other_thread_saw_true.load())
      << "force_scalar() must be thread-local";
  EXPECT_TRUE(util::simd::force_scalar())
      << "this thread's flag must survive the other thread";
  util::simd::force_scalar() = false;
}

} // namespace

#endif // UTIL_HAS_STDX_SIMD
