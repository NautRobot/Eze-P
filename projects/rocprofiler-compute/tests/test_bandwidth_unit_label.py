# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Verify that unit strings preserve case through the display pipeline.

The YAML metric definitions use case-sensitive units like "GB/s",
"FLOPs", "IOPS". The expression evaluation must not mangle them.
"""

import pytest


class TestUnitStringPreservesCase:
    """Verify update_normal_unit_string preserves case-sensitive units."""

    @pytest.mark.parametrize(
        "unit, expected",
        [
            ("GB/s", "GB/s"),
            ("Percent", "Percent"),
            ("Conflicts per Access", "Conflicts per Access"),
            ("Wavefronts", "Wavefronts"),
        ],
    )
    def test_unit_case_preserved(self, unit, expected):
        """Unit strings must not be mangled by .capitalize().

        .capitalize() turns 'GB/s' into 'Gb/s' (gigabits, not
        gigabytes). The function should pass through unit strings
        without altering their case.
        """
        try:
            from utils.metrics.expression import (
                update_normal_unit_string,
            )
        except ImportError:
            pytest.skip("vendored deps not available")

        result = update_normal_unit_string(unit, "per_kernel")
        assert result == expected, (
            f"'{unit}' was mangled to '{result}'. "
            "Likely using .capitalize() instead of returning "
            "the string unchanged."
        )
