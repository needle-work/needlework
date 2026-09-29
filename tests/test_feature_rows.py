"""Lazy feature rows over several memory-mapped parts: reads splice the parts correctly
and an out-of-range row raises instead of returning uninitialized memory."""

import numpy as np
import pytest

from needlework.data.features import Rows


def _rows() -> Rows:
    first = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    second = 100 + np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    return Rows([first, second])


def test_reads_splice_across_parts() -> None:
    rows = _rows()
    np.testing.assert_array_equal(
        rows[np.array([[2, 3], [6, 0]])],
        np.array([[[4, 5], [100, 101]], [[106, 107], [0, 1]]], dtype=np.float32),
    )


@pytest.mark.parametrize("bad", [-1, 7])
def test_out_of_range_rows_raise(bad: int) -> None:
    with pytest.raises(IndexError, match=r"\[0, 7\)"):
        _rows()[np.array([0, bad])]
