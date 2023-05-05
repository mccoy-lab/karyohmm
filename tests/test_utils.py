import pytest
from karyohmm_utils import *


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 0),
        ([0, 1], (1, -1, 1, -1), 1),
        ([1, 1], (0, 1, 1, -1), 2),
        ([1, 1], (-1, -1, -1, -1), -1),
    ],
)
def test_mat_dosage(hap, state, expected):
    """Test the maternal dosage function."""
    assert mat_dosage(hap, state) == expected


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 1),
        ([0, 1], (1, -1, 0, -1), 0),
        ([0, 1], (-1, -1, 0, -1), 0),
        ([0, 1], (1, -1, 0, 1), 1.0),
        ([1, 1], (1, -1, 0, 1), 2.0),
        ([1, 0], (1, -1, 0, 0), 2.0),
        ([1, 1], (-1, -1, -1, -1), -1),
    ],
)
def test_pat_dosage(hap, state, expected):
    """Test the paternal dosage function"""
    assert pat_dosage(hap, state) == expected
