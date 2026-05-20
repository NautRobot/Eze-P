// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_SIMD_H_
#define UTIL_SIMD_H_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if __has_include(<experimental/simd>)
#  include <experimental/simd>
#  define UTIL_HAS_STDX_SIMD 1
#else
#  define UTIL_HAS_STDX_SIMD 0
#endif

namespace util::simd {

#if UTIL_HAS_STDX_SIMD
namespace stdx = std::experimental;

/// Per-thread switch that callers check before taking the SIMD fast
/// path. Used by tests/benchmarks to A/B SIMD vs scalar in one process.
inline bool &force_scalar() {
  static thread_local bool flag = false;
  return flag;
}

template <class T>
using native = stdx::native_simd<T>;

/// Native-SIMD width measured in 32-bit lanes. Convenience constant.
template <class T>
inline constexpr std::size_t native_width_v = native<T>::size();

/// Load `native<T>` from contiguous uint32_t storage if `p != nullptr`;
/// otherwise broadcast `broadcast_bits` (bit-cast to T) to every lane.
/// T must be a 32-bit trivially-copyable type.
template <class T>
inline native<T> load_or_broadcast(const uint32_t *p, uint32_t broadcast_bits) {
  using Bits = stdx::native_simd<uint32_t>;
  using Val  = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  static_assert(sizeof(Val) == sizeof(Bits));
  if (p) {
    Bits bits(p, stdx::element_aligned);
    if constexpr (std::is_same_v<T, uint32_t>) return bits;
    else                                       return std::bit_cast<Val>(bits);
  }
  if constexpr (std::is_same_v<T, uint32_t>) return Val(broadcast_bits);
  else                                       return Val(std::bit_cast<T>(broadcast_bits));
}

/// Store `v` into contiguous uint32_t storage at `dst`, blending in only
/// the lanes whose bit is set in `mask`. If `mask` covers the full SIMD
/// width, falls through to a straight contiguous store.
template <class T>
inline void masked_store(uint32_t *dst, native<T> v, uint64_t mask) {
  using Bits = stdx::native_simd<uint32_t>;
  using Val  = native<T>;
  static_assert(sizeof(T) == sizeof(uint32_t));
  static_assert(sizeof(Val) == sizeof(Bits));
  constexpr std::size_t W = Bits::size();
  uint64_t full = (W >= 64) ? ~0ULL : ((1ULL << W) - 1ULL);
  Bits bits = [&] {
    if constexpr (std::is_same_v<T, uint32_t>) return v;
    else                                       return std::bit_cast<Bits>(v);
  }();
  if ((mask & full) == full) {
    bits.copy_to(dst, stdx::element_aligned);
    return;
  }
  alignas(Bits) uint32_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i)
    if (mask & (1ULL << i)) dst[i] = buf[i];
}

/// Same as masked_store, but writes to a caller-supplied uint32_t buffer
/// instead of contiguous lane storage. For operands whose dst is not a
/// contiguous VGPR (rocjitsu falls back to write_lane_chunk in that case).
template <class T>
inline void blit_to_buffer(uint32_t (&buf)[native<T>::size()], native<T> v) {
  using Bits = stdx::native_simd<uint32_t>;
  Bits bits = [&] {
    if constexpr (std::is_same_v<T, uint32_t>) return v;
    else                                       return std::bit_cast<Bits>(v);
  }();
  bits.copy_to(buf, stdx::vector_aligned);
}
#endif // UTIL_HAS_STDX_SIMD

} // namespace util::simd

#endif // UTIL_SIMD_H_
