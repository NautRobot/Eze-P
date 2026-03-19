##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""Tracks debug output for metric expression evaluation (``-g`` / debug mode)."""

from collections.abc import Hashable


class DebugRowTracker:
    """Tracks which (df_id, row_id) pairs have already been debugged."""

    def __init__(self) -> None:
        self._debugged_rows: set[tuple[int, Hashable]] = set()

    def should_show_inputs(self, df_id: int, row_id: Hashable) -> bool:
        """Check if this row's inputs should be displayed and mark it as seen."""
        key = (df_id, row_id)
        if key in self._debugged_rows:
            return False
        self._debugged_rows.add(key)
        return True
