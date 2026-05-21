// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_simd_test.cpp
/// @brief Generic-core unit tests for util SIMD primitives. Header-only,
/// no rocjitsu Operand/Wavefront fixture.

#include "util/simd.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <thread>

#if !__has_include(<experimental/simd>)

TEST(UtilSimd, ToolchainMissingStdxSimd) {
  GTEST_SKIP() << "<experimental/simd> unavailable; util SIMD helpers skipped.";
}

#else

namespace {

constexpr std::size_t kW = util::native_width_v<uint32_t>;

TEST(UtilSimd, Load_Contiguous_U32) {
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = static_cast<uint32_t>(0xDEAD'0000u + i);
  auto v = util::load<uint32_t>(src);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], src[i]) << "lane " << i;
}

TEST(UtilSimd, Load_Contiguous_F32) {
  alignas(util::native<float>) uint32_t src[kW];
  std::array<float, kW> expected{};
  for (std::size_t i = 0; i < kW; ++i) {
    expected[i] = 1.5f * static_cast<float>(i) - 7.25f;
    src[i] = std::bit_cast<uint32_t>(expected[i]);
  }
  auto v = util::load<float>(src);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], expected[i]) << "lane " << i;
}

TEST(UtilSimd, Broadcast_U32) {
  constexpr uint32_t kBits = 0xCAFEBABEu;
  auto v = util::broadcast<uint32_t>(kBits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kBits) << "lane " << i;
}

TEST(UtilSimd, Broadcast_F32) {
  constexpr float kVal = -3.14159f;
  const uint32_t bits = std::bit_cast<uint32_t>(kVal);
  auto v = util::broadcast<float>(bits);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(v[i], kVal) << "lane " << i;
}

TEST(UtilSimd, MaskedStore_FullMask) {
  alignas(util::native<uint32_t>) uint32_t dst[kW] = {};
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = 0xA5A5'0000u + i;
  auto v = util::load<uint32_t>(src);
  const uint64_t full = util::mask<uint64_t>(static_cast<int>(kW));
  util::masked_store<uint32_t>(dst, v, full);
  EXPECT_EQ(0, std::memcmp(dst, src, sizeof(dst)));
}

TEST(UtilSimd, MaskedStore_PartialMask) {
  alignas(util::native<uint32_t>) uint32_t dst[kW];
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0x1111'1111u;
    src[i] = 0x2222'0000u + i;
  }
  auto v = util::load<uint32_t>(src);
  // 0xA... pattern: enable every other lane starting from bit 1.
  uint64_t mask = 0;
  for (std::size_t i = 1; i < kW; i += 2)
    mask |= (1ULL << i);
  util::masked_store<uint32_t>(dst, v, mask);
  for (std::size_t i = 0; i < kW; ++i) {
    if (mask & (1ULL << i))
      EXPECT_EQ(dst[i], src[i]) << "lane " << i << " should be written";
    else
      EXPECT_EQ(dst[i], 0x1111'1111u) << "lane " << i << " should be preserved";
  }
}

TEST(UtilSimd, MaskedStore_EmptyMask) {
  alignas(util::native<uint32_t>) uint32_t dst[kW];
  alignas(util::native<uint32_t>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i) {
    dst[i] = 0xDEAD'BEEFu;
    src[i] = 0x0BAD'F00Du;
  }
  auto v = util::load<uint32_t>(src);
  util::masked_store<uint32_t>(dst, v, 0ULL);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(dst[i], 0xDEAD'BEEFu) << "lane " << i;
}

TEST(UtilSimd, BlitToBuffer_F32) {
  alignas(util::native<float>) uint32_t src[kW];
  for (std::size_t i = 0; i < kW; ++i)
    src[i] = std::bit_cast<uint32_t>(2.0f * static_cast<float>(i) + 0.5f);
  auto v = util::load<float>(src);
  alignas(util::native<float>) uint32_t buf[util::native<float>::size()] = {};
  util::blit_to_buffer<float>(buf, v);
  for (std::size_t i = 0; i < kW; ++i)
    EXPECT_EQ(buf[i], src[i]) << "lane " << i;
}

TEST(UtilSimd, ForceScalar_ThreadLocal) {
  util::force_scalar() = true;
  ASSERT_TRUE(util::force_scalar());

  std::atomic<bool> other_thread_saw_true{true};
  std::thread t([&]() { other_thread_saw_true.store(util::force_scalar()); });
  t.join();

  EXPECT_FALSE(other_thread_saw_true.load()) << "force_scalar() must be thread-local";
  EXPECT_TRUE(util::force_scalar()) << "this thread's flag must survive the other thread";
  util::force_scalar() = false;
}

} // namespace

#endif // __has_include(<experimental/simd>)
