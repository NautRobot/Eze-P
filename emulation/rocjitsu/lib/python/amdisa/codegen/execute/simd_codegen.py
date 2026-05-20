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

The platform-probe boilerplate (`<experimental/simd>` include, namespace
alias, thread-local force-scalar knob, 32-bit-lane load / masked-store
helpers) lives in the generic `util/simd.h` header. This module emits
only the operand-aware glue that references rocjitsu `Operand` /
`Wavefront` / `Inst` types.
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


def simd_preamble_in_namespace() -> str:
    """SIMD helper declarations / inline definitions. Assumed to be placed
    INSIDE `namespace rocjitsu { namespace amdgpu { ... } }` so the helpers
    sit in the same namespace as `Wavefront` and the per-instruction
    templates that reference them.

    The generic platform probe + lane load/store primitives live in
    `util/simd.h`; this block carries only the operand-aware glue.
    """
    return r'''#if UTIL_HAS_STDX_SIMD
namespace stdx = std::experimental;
#endif
#define ROCJITSU_HAS_STDX_SIMD UTIL_HAS_STDX_SIMD

/// @brief Explicit-width alias for IEEE-754 binary32. C++23 has
/// std::float32_t in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// @brief Per-thread runtime override that disables the SIMD fast path
/// in kernels that have one. Forwards to util::simd::force_scalar().
inline bool &simd_force_scalar() { return util::simd::force_scalar(); }

#if UTIL_HAS_STDX_SIMD
template <typename T, typename Op>
inline util::simd::native<T> read_simd(const Op &op, const Wavefront &wf,
                                       uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint32_t),
                "read_simd: T must be a 32-bit lane type");
  return util::simd::load_or_broadcast<T>(op.simd_lane_ptr(wf, lane_base),
                                          op.read_scalar(wf));
}

template <typename T, typename Op>
inline void write_simd(const Op &op, Wavefront &wf, uint32_t lane_base,
                       util::simd::native<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = util::simd::native_width_v<T>;
  if (uint32_t *p = op.simd_dst_ptr(wf, lane_base)) {
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
'''


def simd_extra_includes() -> list[str]:
    """Extra `#include` lines required by the SIMD preamble.

    Returns the single relative include for the generic util layer.
    `<functional>` (used by `std::plus<>{}` in probe lines) lives in the
    main angle-include block of `_generator.gen_shared_execute`.
    """
    return ['#include "util/simd.h"']
