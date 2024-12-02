"""Test suite to make sure that input I/O is correct."""

import numpy as np
import pandas as pd
import pytest

from karyohmm import DataReader

# -------- Creating fake test data -------- #
test_dict = {
    "chrom": ["chr1", "chr1"],
    "pos": [1, 2],
    "ref": ["R", "R"],
    "alt": ["A", "A"],
    "baf": [0.1, 0.1],
    "mat_haps": [[0, 0], [0, 1]],
    "pat_haps": [[1, 1], [1, 0]],
}

test_df = pd.DataFrame(
    {
        "chrom": ["chr1", "chr1"],
        "pos": [1, 2],
        "ref": ["R", "R"],
        "alt": ["A", "A"],
        "baf": [0.1, 0.1],
        "mat_hap0": [0, 0],
        "mat_hap1": [0, 1],
        "pat_hap0": [1, 1],
        "pat_hap1": [1, 0],
    }
)


@pytest.mark.parametrize("mode, duo_maternal", [("Meta", None), ("Duo", False)])
def test_correct_mode(mode, duo_maternal):
    """Test that only correct modes are accepted for data reading."""
    DataReader(mode=mode, duo_maternal=duo_maternal)


@pytest.mark.parametrize(
    "mode, duo_maternal", [("Duo", None), ("Duo", "X"), ("Duo", 2)]
)
def test_incorrect_duo(mode, duo_maternal):
    """Test incorrect duo-mode parameters."""
    with pytest.raises(ValueError):
        DataReader(mode=mode, duo_maternal=duo_maternal)


@pytest.mark.parametrize("mode", ["", "X", None])
def test_bad_mode(mode):
    """Test incorrect modes."""
    with pytest.raises(AssertionError):
        DataReader(mode=mode)


@pytest.mark.parametrize("df", [test_df])
def test_read_tsv(df, tmp_path):
    """Test reading in a good CSV."""
    p = tmp_path / "x.tsv"
    df.to_csv(p, sep="\t", index=None)
    data_reader = DataReader()
    data_reader.read_data(input_fp=str(p))


@pytest.mark.parametrize("d", [test_dict])
def test_read_npz(d, tmp_path):
    """Test reading in NPZ formatted files."""
    p = tmp_path / "x.npz"
    np.savez_compressed(p, **d)
    data_reader = DataReader()
    data_reader.read_data(input_fp=str(p))
