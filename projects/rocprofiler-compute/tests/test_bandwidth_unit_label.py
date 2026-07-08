# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests verifying that bandwidth labels use "GB/s" (gigabytes), not "Gb/s" (gigabits).

The YAML metric definitions use "GB/s" and the underlying values are in gigabytes
per second. The display layer must match.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.tui


class TestGuiBandwidthUnitFilter:
    """Verify create_sol_charts filters bandwidth rows by 'GB/s', not 'Gb/s'."""

    def _make_sol_df(self):
        """Build a DataFrame matching the L2 Cache SOL panel (table_id=1701)."""
        return pd.DataFrame({
            "Metric": [
                "GPU L2 Util",
                "L2-Fabric Read BW",
                "L2-Fabric Write and Atomic BW",
                "HBM Bandwidth",
            ],
            "Avg": ["45.0", "416.5", "293.9", "1638.4"],
            "Unit": ["Percent", "GB/s", "GB/s", "GB/s"],
        })

    def test_bandwidth_chart_uses_correct_unit_filter(self):
        """The HBM bandwidth chart must include GB/s rows from the dataframe.

        If the filter uses 'Gb/s' instead of 'GB/s', no rows match and
        the bandwidth chart is silently empty.
        """
        from utils.gui import create_sol_charts

        df = self._make_sol_df()
        charts = create_sol_charts(df, table_id=1701)

        # table_id=1701 should produce 2 charts: percent chart + bandwidth chart
        assert len(charts) == 2, (
            f"Expected 2 charts (percent + bandwidth), got {len(charts)}. "
            "The bandwidth chart is missing — likely filtering on wrong unit string."
        )

        bw_chart = charts[1]
        bw_data = bw_chart.data[0]
        # The bandwidth chart should contain the 3 GB/s metrics
        assert len(bw_data.y) == 3, (
            f"Expected 3 bandwidth metrics, got {len(bw_data.y)}. "
            "Unit filter may be using 'Gb/s' instead of 'GB/s'."
        )


class TestTuiChartBandwidthLabel:
    """Verify px_simple_bar uses 'GB/s' label for bandwidth chart (id=1701.2)."""

    def _make_chart_df(self):
        """Build a minimal DataFrame for px_simple_bar.

        The Value column must use a numpy dtype so that .astype(int) works
        on individual elements (px_simple_bar iterates and calls x.astype).
        """
        return pd.DataFrame({
            "Metric": ["L2-Fabric Read BW", "HBM Bandwidth"],
            "Value": pd.array([416, 1638], dtype="Int64"),
        })

    def test_bandwidth_axis_label_is_GB_not_Gb(self):
        """The x-axis label for id=1701.2 must be 'GB/s', not 'Gb/s'.

        Values are in gigabytes per second; 'Gb/s' would mean gigabits,
        which is an 8x misrepresentation.
        """
        from rocprof_compute_tui.widgets.charts import px_simple_bar

        df = self._make_chart_df()
        fig = px_simple_bar(df, title="test", id=1701.2)

        x_label = fig.layout.xaxis.title.text
        assert x_label == "GB/s", (
            f"x-axis label is '{x_label}', expected 'GB/s'. "
            "The label_txt override for id=1701.2 uses the wrong unit."
        )
