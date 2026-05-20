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

The SIMD helper templates themselves (`read_simd`, `write_simd`,
`try_execute_binary_vop2_simd`, plus the `ROCJITSU_HAS_STDX_SIMD` macro
and `stdx` alias) live in the hand-maintained header
``rocjitsu/isa/arch/amdgpu/shared/simd_glue.h``. This module emits only
the probe-line call sites and the `#include` that pulls the helpers in.
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


def simd_extra_includes() -> list[str]:
    """Extra `#include` lines required by the SIMD probe call sites.

    The helper templates (and the ``ROCJITSU_HAS_STDX_SIMD`` macro that
    callers may depend on) live in ``simd_glue.h``; that header pulls in
    ``util/simd.h`` transitively, so this is the only SIMD-specific include
    the generated shared header needs.
    """
    return ['#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"']
