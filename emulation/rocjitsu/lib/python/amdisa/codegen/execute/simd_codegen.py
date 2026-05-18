# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SIMD specialization codegen for AMDGPU VOP2 execute kernels.

Emits a `<experimental/simd>`-based fast path on top of the generated
scalar per-lane bodies. The scalar body is preserved verbatim as a
fallback; the SIMD probe is one line at the start of the kernel:

    if (try_execute_binary_vop2_simd<T>(inst, wf, op_functor)) return;

Eligible kernels are listed in :data:`SIMD_VOP2_BINARY` — only those
whose host SIMD result is bit-identical to the scalar generated body
(IEEE-754 single-rounded fp arithmetic, wrap-around integer arithmetic,
elementwise bitwise ops). NaN-sensitive ops (min/max), VCC-writing ops
(add_co), and modifier-bearing forms (VOP3 with abs/neg/clamp/omod) are
excluded — those need their own helpers.
"""

from __future__ import annotations

# template_name -> (cpp_element_type, cpp_binary_op_functor)
#
# template_name matches the symbol emitted by _generator.gen_shared_execute:
#   f"{inst.mnemonic}_{enc_key}"  (e.g. "v_add_f32_vop2").
#
# The functor is invoked as `bin_op(simd<T>, simd<T>) -> simd<T>` inside
# try_execute_binary_vop2_simd. Use std::*<> for stateless ops.
SIMD_VOP2_BINARY: dict[str, tuple[str, str]] = {
    'v_add_f32_vop2': ('float32_t', 'std::plus<>{}'),
    'v_add_u32_vop2': ('uint32_t',  'std::plus<>{}'),
}


def simd_probe_line(template_name: str) -> str | None:
    """Return the SIMD fast-path probe line for a kernel, or None."""
    spec = SIMD_VOP2_BINARY.get(template_name)
    if spec is None:
        return None
    cpp_t, cpp_op = spec
    return (f'  if (try_execute_binary_vop2_simd<{cpp_t}>(inst, wf, {cpp_op}))\n'
            f'    return;')


def simd_preamble_top() -> str:
    """SIMD-related `#include` / `#define` block. Emitted BEFORE the
    `namespace rocjitsu` / `namespace amdgpu` open."""
    return r'''#if __has_include(<experimental/simd>)
#include <experimental/simd>
#define ROCJITSU_HAS_STDX_SIMD 1
#else
#define ROCJITSU_HAS_STDX_SIMD 0
#endif
'''


def simd_preamble_in_namespace() -> str:
    """SIMD helper declarations / inline definitions. Assumed to be placed
    INSIDE `namespace rocjitsu { namespace amdgpu { ... } }` so the helpers
    sit in the same namespace as `Wavefront` and the per-instruction
    templates that reference them.
    """
    return r'''#if ROCJITSU_HAS_STDX_SIMD
namespace stdx = std::experimental;
#endif

/// @brief Explicit-width alias for IEEE-754 binary32. C++23 has
/// std::float32_t in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// @brief Per-thread runtime override that disables the SIMD fast path
/// in kernels that have one. Used by tests/v_add_simd_benchmark.cpp.
inline bool &simd_force_scalar() {
  static thread_local bool flag = false;
  return flag;
}

#if ROCJITSU_HAS_STDX_SIMD
template <typename T, typename Op>
inline stdx::native_simd<T> read_simd(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  using BitsSimd = stdx::native_simd<uint32_t>;
  using ValSimd = stdx::native_simd<T>;
  static_assert(sizeof(T) == sizeof(uint32_t), "read_simd: T must be a 32-bit lane type");
  static_assert(sizeof(ValSimd) == sizeof(BitsSimd));
  if (const uint32_t *p = op.simd_lane_ptr(wf, lane_base)) {
    BitsSimd bits(p, stdx::element_aligned);
    if constexpr (std::is_same_v<T, uint32_t>)
      return bits;
    else
      return std::bit_cast<ValSimd>(bits);
  }
  uint32_t scalar = op.read_scalar(wf);
  if constexpr (std::is_same_v<T, uint32_t>)
    return ValSimd(scalar);
  else
    return ValSimd(std::bit_cast<T>(scalar));
}

template <typename T, typename Op>
inline void write_simd(const Op &op, Wavefront &wf, uint32_t lane_base, stdx::native_simd<T> v,
                       uint64_t mask) {
  using BitsSimd = stdx::native_simd<uint32_t>;
  using ValSimd = stdx::native_simd<T>;
  static_assert(sizeof(T) == sizeof(uint32_t), "write_simd: T must be a 32-bit lane type");
  static_assert(sizeof(ValSimd) == sizeof(BitsSimd));
  constexpr std::size_t W = BitsSimd::size();
  uint64_t full = (W >= 64) ? ~0ULL : ((1ULL << W) - 1ULL);
  BitsSimd bits;
  if constexpr (std::is_same_v<T, uint32_t>)
    bits = v;
  else
    bits = std::bit_cast<BitsSimd>(v);
  if (uint32_t *p = op.simd_dst_ptr(wf, lane_base)) {
    if ((mask & full) == full) {
      bits.copy_to(p, stdx::element_aligned);
      return;
    }
    alignas(BitsSimd) uint32_t buf[W];
    bits.copy_to(buf, stdx::vector_aligned);
    for (std::size_t i = 0; i < W; ++i)
      if (mask & (1ULL << i))
        p[i] = buf[i];
    return;
  }
  alignas(BitsSimd) uint32_t buf[W];
  bits.copy_to(buf, stdx::vector_aligned);
  op.write_lane_chunk(wf, lane_base, static_cast<uint32_t>(W), buf, mask);
}

template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  uint64_t exec = wf.exec();
  constexpr std::size_t W = stdx::native_simd<T>::size();
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
'''


def simd_extra_includes() -> list[str]:
    """Extra `#include` lines required by the SIMD preamble."""
    return [
        '#include <cstring>',
        '#include <functional>',
    ]
