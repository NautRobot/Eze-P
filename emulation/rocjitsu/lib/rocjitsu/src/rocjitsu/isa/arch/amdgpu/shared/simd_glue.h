// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Operand-aware SIMD glue for the auto-generated execute_<mnemonic>
// kernels in execute_shared.h. Hand-maintained; lives separately from
// the generated header so the same code is not duplicated in
// simd_codegen.py (raw string) and execute_shared.h (emitted output).
//
// Layering: this header sees rocjitsu types (Wavefront, plus the Op /
// Inst template parameters) and bridges to the generic util::simd
// primitives in util/simd.h. The generic util layer never depends on
// rocjitsu; only this direction is permitted.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <cstddef>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

#if UTIL_HAS_STDX_SIMD
namespace stdx = std::experimental;
#endif
#define ROCJITSU_HAS_STDX_SIMD UTIL_HAS_STDX_SIMD

/// Explicit-width alias for IEEE-754 binary32. C++23 has std::float32_t
/// in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// Per-thread runtime override that disables the SIMD fast path in
/// kernels that have one. Forwards to util::simd::force_scalar().
inline bool &simd_force_scalar() { return util::simd::force_scalar(); }

#if UTIL_HAS_STDX_SIMD
template <typename T, typename Op>
inline util::simd::native<T> read_simd(const Op &op, const Wavefront &wf,
                                       uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint32_t),
                "read_simd: T must be a 32-bit lane type");
  const uint32_t *p = SimdAccess::lane_ptr(op, wf, lane_base);
  return util::simd::load_or_broadcast<T>(p, p ? 0u : op.read_scalar(wf));
}

template <typename T, typename Op>
inline void write_simd(const Op &op, Wavefront &wf, uint32_t lane_base,
                       util::simd::native<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = util::simd::native_width_v<T>;
  if (uint32_t *p = SimdAccess::dst_ptr(op, wf, lane_base)) {
    util::simd::masked_store<T>(p, v, mask);
    return;
  }
  alignas(util::simd::native<T>) uint32_t buf[W];
  util::simd::blit_to_buffer<T>(buf, v);
  op.write_lane_chunk(wf, lane_base, static_cast<uint32_t>(W), buf, mask);
}

template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &inst, Wavefront &wf,
                                                       BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  uint64_t exec = wf.exec();
  constexpr std::size_t W = util::simd::native_width_v<T>;
  uint64_t chunk_full = (W >= 64) ? ~0ULL : ((1ULL << W) - 1ULL);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = read_simd<T>(inst.src0, wf, base);
    auto b = read_simd<T>(inst.vsrc1, wf, base);
    write_simd<T>(inst.vdst, wf, base, bin_op(a, b), chunk);
  }
  return true;
}
#else
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &, Wavefront &, BinOp) {
  return false;
}
#endif

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
