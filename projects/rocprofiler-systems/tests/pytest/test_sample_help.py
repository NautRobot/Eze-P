# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Automated coverage for the ``rocprof-sys-sample`` command-line help surface.

Validates every help / informational invocation of ``rocprof-sys-sample``: the
help entry points (``--version`` / ``--help`` / ``--help=all`` / compact help),
every ``--help=<topic>`` / ``--help=<domain>`` page, the preset listing and the
``--explain=<preset>`` pages.

Each case is a ``pytest.param`` of ``(run_args, pass_regex)`` consumed by the
single parametrized test below; add or adjust coverage by editing the
``HELP_CASES`` list.
"""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.sampling]

TARGET = "rocprof-sys-sample"


# ----------------------------------------------------------------------------
# Shared pattern groups (kept here to avoid repeating them across cases)
# ----------------------------------------------------------------------------

# Sections printed by the compact help screens (--help / -h / -?).
_COMPACT = [
    r"Usage: rocprof-sys-sample",
    r"QUICK START",
    r"DOMAIN FLAGS",
    r"COMMON OPTIONS",
    r"HELP TOPICS",
    r"EXAMPLES",
]

# Common scaffolding printed by every --explain=<preset> page.
_EXPLAIN = [
    r"Description:",
    r"Use case:",
    r"Environment Variables:",
    r"ROCPROFSYS_\w+ = ",
]


def _explain(preset: str, category: str) -> list[str]:
    """pass_regex for an ``--explain=<preset>`` page (preset-specific + shared)."""
    return [
        rf"Preset: {preset}",
        rf"Category:\s+{category}",
        rf"Usage: rocprof-sys-sample --preset={preset}",
    ] + _EXPLAIN


# ----------------------------------------------------------------------------
# Expected-output cases: (run_args, pass_regex)
# ----------------------------------------------------------------------------

HELP_CASES = [
    # --- Domain help pages (--help=<domain>) ------------------------------
    pytest.param(
        ["--help=cpu"],
        [
            r"CPU OPTIONS \(",
            r"-H, --host",
            r"-f, --sampling-freq",
            r"-t, --tids",
            r"--sampling-wait",
            r"--sampling-duration",
            r"--sample-cputime",
            r"--sample-realtime",
            r"--sample-overflow",
            r"-C, --cpu-events",
            r"-G, --gpu-events",
            r"--cpu",
            r"See also",
        ],
        id="domain_cpu",
    ),
    pytest.param(
        ["--help=gpu"],
        [
            r"GPU OPTIONS \(",
            r"-D, --device",
            r"--use-amd-smi",
            r"--amd-smi-metrics",
            r"--process-freq",
            r"--process-wait",
            r"--process-duration",
            r"--gpus",
            r"--ai-nics",
            r"-G, --gpu-events",
            r"--gpu",
            r"See also",
        ],
        id="domain_gpu",
    ),
    pytest.param(
        ["--help=parallel"],
        [
            r"PARALLEL OPTIONS \(",
            r"-I, --include",
            r"-E, --exclude",
            r"--parallel",
            r"See also",
        ],
        id="domain_parallel",
    ),
    pytest.param(
        ["--help=rocm"],
        [
            r"ROCM OPTIONS \(",
            r"-T, --trace",
            r"--use-amd-smi",
            r"--selected-regions",
            r"--gpus",
            r"--ai-nics",
            r"--hsa-interrupt",
            r"--rocm",
            r"See also",
        ],
        id="domain_rocm",
    ),
    # --- Preset explanations (--explain=<preset>) -------------------------
    pytest.param(
        ["--explain=balanced"], _explain("balanced", "general"), id="explain_balanced"
    ),
    pytest.param(
        ["--explain=detailed"], _explain("detailed", "general"), id="explain_detailed"
    ),
    pytest.param(
        ["--explain=profile-mpi"],
        _explain("profile-mpi", "hpc"),
        id="explain_profile-mpi",
    ),
    pytest.param(
        ["--explain=profile-only"],
        _explain("profile-only", "general"),
        id="explain_profile-only",
    ),
    pytest.param(
        ["--explain=runtime-trace"],
        _explain("runtime-trace", "tracing"),
        id="explain_runtime-trace",
    ),
    pytest.param(
        ["--explain=sys-trace"], _explain("sys-trace", "tracing"), id="explain_sys-trace"
    ),
    pytest.param(
        ["--explain=trace-gpu"], _explain("trace-gpu", "gpu"), id="explain_trace-gpu"
    ),
    pytest.param(
        ["--explain=trace-hpc"], _explain("trace-hpc", "hpc"), id="explain_trace-hpc"
    ),
    pytest.param(
        ["--explain=trace-hw-counters"],
        _explain("trace-hw-counters", "gpu"),
        id="explain_trace-hw-counters",
    ),
    pytest.param(
        ["--explain=trace-openmp"],
        _explain("trace-openmp", "hpc"),
        id="explain_trace-openmp",
    ),
    pytest.param(
        ["--explain=workload-trace"],
        _explain("workload-trace", "gpu"),
        id="explain_workload-trace",
    ),
    # --- Full option dump (--help=all): every section header + flag -------
    pytest.param(
        ["--help=all"],
        [
            r"Options:",
            r"-h, -\?, --help",
            r"--version(?![-\w])",
            r"\[DEBUG OPTIONS\]",
            r"--log-level(?![-\w])",
            r"--monochrome(?![-\w])",
            r"--debug(?![-\w])",
            r"-v, --verbose(?![-\w])",
            r"--ci(?![-\w])",
            r"--log-file(?![-\w])",
            r"--dl-verbose(?![-\w])",
            r"--perfetto-annotations(?![-\w])",
            r"--kokkosp-kernel-logger(?![-\w])",
            r"--ci-skip-push-pop-check(?![-\w])",
            r"--sampling-allocator-size(?![-\w])",
            r"--kokkosp-name-length-max(?![-\w])",
            r"--kokkosp-prefix(?![-\w])",
            r"\[MODE OPTIONS\]",
            r"--mode(?![-\w])",
            r"\[GENERAL OPTIONS\]",
            r"-c, --config(?![-\w])",
            r"-o, --output(?![-\w])",
            r"-T, --trace(?![-\w])",
            r"-L, --trace-legacy(?![-\w])",
            r"-P, --profile(?![-\w])",
            r"-F, --flat-profile(?![-\w])",
            r"-H, --host(?![-\w])",
            r"-D, --device(?![-\w])",
            r"-w, --wait(?![-\w])",
            r"-d, --duration(?![-\w])",
            r"--periods(?![-\w])",
            r"--rank-filter-id(?![-\w])",
            r"--rank-filter-output(?![-\w])",
            r"--rank-filter-logs(?![-\w])",
            r"\[BACKEND OPTIONS\]",
            r"-I, --include(?![-\w])",
            r"-E, --exclude(?![-\w])",
            r"--use-amd-smi(?![-\w])",
            r"--use-causal(?![-\w])",
            r"--amd-smi-metrics(?![-\w])",
            r"--use-code-coverage(?![-\w])",
            r"--use-kokkosp(?![-\w])",
            r"--use-mpip(?![-\w])",
            r"--use-rcclp(?![-\w])",
            r"--use-rocpd(?![-\w])",
            r"--use-sampling(?![-\w])",
            r"--use-shmem(?![-\w])",
            r"--use-ucx(?![-\w])",
            r"--trace-thread-barriers(?![-\w])",
            r"--trace-thread-join(?![-\w])",
            r"--trace-thread-locks(?![-\w])",
            r"--use-process-sampling(?![-\w])",
            r"--trace-thread-rw-locks(?![-\w])",
            r"--trace-thread-spin-locks(?![-\w])",
            r"--use-unified-memory-profiling(?![-\w])",
            r"\[PARALLELISM OPTIONS\]",
            r"--thread-pool-size(?![-\w])",
            r"--num-threads-hint(?![-\w])",
            r"\[TRACING OPTIONS\]",
            r"--trace-file(?![-\w])",
            r"--trace-buffer-size(?![-\w])",
            r"--trace-fill-policy(?![-\w])",
            r"--trace-wait(?![-\w])",
            r"--trace-duration(?![-\w])",
            r"--trace-periods(?![-\w])",
            r"--selected-regions(?![-\w])",
            r"--trace-clock-id(?![-\w])",
            r"\[PROFILE OPTIONS\]",
            r"--profile-format(?![-\w])",
            r"--profile-diff(?![-\w])",
            r"\[HOST/DEVICE \(PROCESS SAMPLING\) OPTIONS\]",
            r"--process-freq(?![-\w])",
            r"--process-wait(?![-\w])",
            r"--process-duration(?![-\w])",
            r"--cpus(?![-\w])",
            r"--gpus(?![-\w])",
            r"--ai-nics(?![-\w])",
            r"\[GENERAL SAMPLING OPTIONS\]",
            r"-f, --sampling-freq(?![-\w])",
            r"-t, --tids(?![-\w])",
            r"--sampling-wait(?![-\w])",
            r"--sampling-duration(?![-\w])",
            r"--sample-cputime(?![-\w])",
            r"--sample-realtime(?![-\w])",
            r"--sample-overflow(?![-\w])",
            r"--sampling-ainics(?![-\w])",
            r"\[SAMPLING TIMER OPTIONS\]",
            r"--sampling-cputime-delay(?![-\w])",
            r"--sampling-cputime-freq(?![-\w])",
            r"--sampling-cputime-signal(?![-\w])",
            r"--sampling-cputime-tids(?![-\w])",
            r"--sampling-realtime-delay(?![-\w])",
            r"--sampling-realtime-freq(?![-\w])",
            r"--sampling-realtime-signal(?![-\w])",
            r"--sampling-realtime-tids(?![-\w])",
            r"\[ADVANCED SAMPLING OPTIONS\]",
            r"--sampling-include-inlines(?![-\w])",
            r"--sampling-keep-internal(?![-\w])",
            r"--sampling-overflow-event(?![-\w])",
            r"--sampling-overflow-freq(?![-\w])",
            r"--sampling-overflow-signal(?![-\w])",
            r"--sampling-overflow-tids(?![-\w])",
            r"\[HARDWARE COUNTER OPTIONS\]",
            r"-C, --cpu-events(?![-\w])",
            r"-G, --gpu-events(?![-\w])",
            r"\[CATEGORY OPTIONS\]",
            r"--enable-categories(?![-\w])",
            r"--disable-categories(?![-\w])",
            r"\[IO OPTIONS\]",
            r"--tmpdir(?![-\w])",
            r"--use-pid(?![-\w])",
            r"--causal-file(?![-\w])",
            r"--time-output(?![-\w])",
            r"--causal-file-reset(?![-\w])",
            r"--use-temporary-files(?![-\w])",
            r"\[PERFETTO OPTIONS\]",
            r"--perfetto-backend(?![-\w])",
            r"--rocm-group-by-queue(?![-\w])",
            r"--merge-perfetto-files(?![-\w])",
            r"--perfetto-flush-period-ms(?![-\w])",
            r"--perfetto-shmem-size-hint-kb(?![-\w])",
            r"\[TIMEMORY OPTIONS\]",
            r"--timemory-components(?![-\w])",
            r"\[ROCM OPTIONS\]",
            r"--rocm-domains(?![-\w])",
            r"--gpu-perf-counters(?![-\w])",
            r"--rocm-hip-runtime-api-operations(?![-\w])",
            r"--rocm-marker-api-operations(?![-\w])",
            r"--rocm-memory-copy-operations(?![-\w])",
            r"--rocm-kfd-page-fault-operations(?![-\w])",
            r"\[MISCELLANEOUS OPTIONS\]",
            r"-i, --inlines(?![-\w])",
            r"--hsa-interrupt(?![-\w])",
            r"--use-ainic(?![-\w])",
            r"--kill-delay(?![-\w])",
            r"--causal-backend(?![-\w])",
            r"--causal-delay(?![-\w])",
            r"--causal-duration(?![-\w])",
            r"--causal-mode(?![-\w])",
            r"--cpu-freq-enabled(?![-\w])",
            r"--cpu-metrics(?![-\w])",
            r"--causal-binary-exclude(?![-\w])",
            r"--causal-binary-scope(?![-\w])",
            r"--causal-end-to-end(?![-\w])",
            r"--kokkosp-deep-copy(?![-\w])",
            r"--causal-fixed-speedup(?![-\w])",
            r"--causal-function-exclude(?![-\w])",
            r"--causal-function-exclude-defaults(?![-\w])",
            r"--causal-function-scope(?![-\w])",
            r"--causal-random-seed(?![-\w])",
            r"--causal-source-exclude(?![-\w])",
            r"--causal-source-scope(?![-\w])",
            r"\[LIBROCPROF-SYS OPTIONS\]",
            r"\[PRESET OPTIONS\]",
            r"--preset(?![-\w])",
            r"--list-presets(?![-\w])",
            r"--explain(?![-\w])",
            r"\[DOMAIN OPTIONS\]",
            r"--gpu(?![-\w])",
            r"--rocm(?![-\w])",
            r"--cpu(?![-\w])",
            r"--parallel(?![-\w])",
            r"\[EXPORT OPTIONS\]",
            r"--export-config(?![-\w])",
        ],
        id="help_all",
    ),
    # --- Compact help screens (--help / -h / -?) --------------------------
    pytest.param(["--help"], _COMPACT, id="help_compact_long"),
    pytest.param(["-h"], _COMPACT, id="help_compact_short_h"),
    pytest.param(["-?"], _COMPACT, id="help_compact_short_q"),
    # --- Preset listing (--list-presets) ----------------------------------
    pytest.param(
        ["--list-presets"],
        [
            r"Available Presets:",
            r"Usage: rocprof-sys-sample --preset=<name>",
            r"rocprof-sys-sample --explain=<name>",
            r"general:",
            r"gpu:",
            r"hpc:",
            r"tracing:",
            r"\bbalanced\b",
            r"\bdetailed\b",
            r"\bprofile-only\b",
            r"\btrace-gpu\b",
            r"\btrace-hw-counters\b",
            r"\bworkload-trace\b",
            r"\bprofile-mpi\b",
            r"\btrace-hpc\b",
            r"\btrace-openmp\b",
            r"\bruntime-trace\b",
            r"\bsys-trace\b",
        ],
        id="list_presets",
    ),
    # --- Topic help pages (--help=<topic>) --------------------------------
    pytest.param(
        ["--help=backend"],
        [
            r"rocprof-sys-sample --help=backend",
            r"-I, --include",
            r"-E, --exclude",
            r"--use-amd-smi",
            r"--use-causal",
            r"--amd-smi-metrics",
            r"--use-code-coverage",
            r"--use-kokkosp",
            r"--use-mpip",
            r"--use-rcclp",
            r"--use-rocpd",
            r"--use-sampling",
            r"--use-shmem",
            r"--use-ucx",
            r"--trace-thread-barriers",
            r"--trace-thread-join",
            r"--trace-thread-locks",
            r"--use-process-sampling",
            r"--trace-thread-rw-locks",
            r"--trace-thread-spin-locks",
            r"--use-unified-memory-profiling",
            r"See also",
        ],
        id="topic_backend",
    ),
    pytest.param(
        ["--help=counters"],
        [
            r"rocprof-sys-sample --help=counters",
            r"-C, --cpu-events",
            r"-G, --gpu-events",
            r"See also",
        ],
        id="topic_counters",
    ),
    pytest.param(
        ["--help=debug"],
        [
            r"rocprof-sys-sample --help=debug",
            r"--log-level",
            r"--monochrome",
            r"--debug",
            r"-v, --verbose",
            r"--log-file",
            r"--dl-verbose",
            r"--perfetto-annotations",
            r"--kokkosp-kernel-logger",
            r"--ci-skip-push-pop-check",
            r"--sampling-allocator-size",
            r"--kokkosp-name-length-max",
            r"--kokkosp-prefix",
            r"--ci",
        ],
        id="topic_debug",
    ),
    pytest.param(
        ["--help=general"],
        [
            r"rocprof-sys-sample --help=general",
            r"-c, --config",
            r"-o, --output",
            r"-T, --trace",
            r"-L, --trace-legacy",
            r"-P, --profile",
            r"-F, --flat-profile",
            r"-H, --host",
            r"-D, --device",
            r"-w, --wait",
            r"-d, --duration",
            r"--periods",
            r"--rank-filter-id",
            r"--rank-filter-output",
            r"--rank-filter-logs",
            r"See also",
        ],
        id="topic_general",
    ),
    pytest.param(
        ["--help=misc"],
        [
            r"rocprof-sys-sample --help=misc",
            r"-i, --inlines",
            r"--hsa-interrupt",
            r"See also",
        ],
        id="topic_misc",
    ),
    pytest.param(
        # --help=preset is curated (get_help_topic_map) to render PRESET +
        # DOMAIN + EXPORT sections together; assert the three section headers
        # plus every left-column flag they list.
        ["--help=preset"],
        [
            r"rocprof-sys-sample --help=preset",
            r"\[PRESET OPTIONS\]",
            r"\[DOMAIN OPTIONS\]",
            r"\[EXPORT OPTIONS\]",
            r"--list-presets",
            r"--explain",
            r"--preset",
            r"--gpu",
            r"--rocm",
            r"--cpu",
            r"--parallel",
            r"--export-config",
            r"See also",
        ],
        id="topic_preset",
    ),
    pytest.param(
        ["--help=process"],
        [
            r"rocprof-sys-sample --help=process",
            r"--process-freq",
            r"--process-wait",
            r"--process-duration",
            r"--cpus",
            r"--gpus",
            r"--ai-nics",
            r"See also",
        ],
        id="topic_process",
    ),
    pytest.param(
        ["--help=profiling"],
        [
            r"rocprof-sys-sample --help=profiling",
            r"--profile-format",
            r"--profile-diff",
            r"See also",
        ],
        id="topic_profiling",
    ),
    pytest.param(
        ["--help=sampling"],
        [
            r"rocprof-sys-sample --help=sampling",
            r"-f, --sampling-freq",
            r"-t, --tids",
            r"--sampling-wait",
            r"--sampling-duration",
            r"--sample-cputime",
            r"--sample-realtime",
            r"--sample-overflow",
            r"--sampling-ainics",
            r"--sampling-cputime-delay",
            r"--sampling-cputime-freq",
            r"--sampling-cputime-signal",
            r"--sampling-cputime-tids",
            r"--sampling-include-inlines",
            r"--sampling-keep-internal",
            r"--sampling-overflow-event",
            r"--sampling-overflow-freq",
            r"--sampling-overflow-signal",
            r"--sampling-overflow-tids",
            r"--sampling-realtime-delay",
            r"--sampling-realtime-freq",
            r"--sampling-realtime-signal",
            r"--sampling-realtime-tids",
            r"See also",
        ],
        id="topic_sampling",
    ),
    pytest.param(
        ["--help=tracing"],
        [
            r"rocprof-sys-sample --help=tracing",
            r"--trace-file",
            r"--trace-buffer-size",
            r"--trace-fill-policy",
            r"--trace-wait",
            r"--trace-duration",
            r"--trace-periods",
            r"--selected-regions",
            r"--trace-clock-id",
            r"See also",
        ],
        id="topic_tracing",
    ),
    # --- Error / version handling -----------------------------------------
    pytest.param(
        ["--help=does-not-exist"],
        [
            r"Unknown help topic 'does-not-exist'",
            r"Available topics",
        ],
        id="unknown_topic",
    ),
    pytest.param(
        ["--version"],
        [r"rocprof-sys-sample v\d+\.\d+\.\d+"],
        id="version",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("sample-help")
class TestSampleHelp(RocprofsysTest):
    """Validate every help / informational invocation of rocprof-sys-sample."""

    @pytest.mark.parametrize("run_args, pass_regex", HELP_CASES)
    def test_help(self, run_args, pass_regex):
        result = self.run_test(
            "baseline",
            target=TARGET,
            run_args=run_args,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)
