"""Classes for reading in data for karyoHMM.

Implements methods for both MetaHMM and PoCHMM-based datasets.
"""

import numpy as np
import polars as pl


class DataReader:
    """Input / output class for reading in data for karyohmm."""

    def __init__(self, mode="Meta", duo_maternal=None):
        """Initialize the DataReader class for a given mode."""
        assert mode in ["Meta", "Duo"]
        self.mode = mode
        if (mode == "Duo") and (type(duo_maternal) is not bool):
            raise ValueError(
                "Need to specify whether a mother-child or father-child duo!"
            )
        if (duo_maternal is True) and (mode == "Duo"):
            self.duo_maternal = duo_maternal

    def read_data_np(self, input_fp):
        """Read data from an .npy or npz file and reformat for karyohmm.

        Args:
            input_fp (`str`): path to input NPZ or NPY file.

        Output:
            df (`pl.DataFrame`): polars dataframe of cleaned options.

        """
        data = np.load(input_fp, allow_pickle=True)
        for x in ["chrom", "pos", "ref", "alt", "baf"]:
            assert x in data
        df = pl.DataFrame(
            {
                "chrom": data["chrom"],
                "pos": data["pos"],
                "ref": data["ref"],
                "alt": data["alt"],
                "baf": data["baf"],
            }
        )
        if "mat_haps" in data:
            assert data["mat_haps"].ndim == 2
            df = df.with_columns(
                pl.Series(name="mat_hap0", values=data["mat_haps"][0, :]),
                pl.Series(name="mat_hap1", values=data["mat_haps"][1, :]),
            )
        if "pat_haps" in data:
            assert data["pat_haps"].ndim == 2
            df = df.with_columns(
                pl.Series(name="pat_hap0", values=data["pat_haps"][0, :]),
                pl.Series(name="pat_hap1", values=data["pat_haps"][1, :]),
            )
        if "af" in data:
            df = df.with_columns(pl.Series(name="af", values=data["af"]))
        if ("lrr" in data) and ("sigma" in data):
            df = df.with_columns(
                pl.Series(name="lrr", values=data["lrr"]),
                pl.Series(name="lrr", values=data["sigmas"]),
            )
        return df

    def read_data_df(self, input_fp):
        """Read in data from a pre-existing text-based dataset.

        Args:
            input_fp (`str`): path to input TSV/CSV/TXT file.

        Output:
            df (`pl.DataFrame`): polars dataframe of cleaned options.

        """
        sep = ","
        if ".tsv" in input_fp:
            sep = "\t"
        elif ".txt" in input_fp:
            sep = " "
        df = pl.read_csv(input_fp, separator=sep)
        for x in ["chrom", "pos", "ref", "alt", "baf"]:
            assert x in df.columns
        if self.mode == "Meta":
            for x in ["mat_hap0", "mat_hap1", "pat_hap0", "pat_hap1"]:
                assert x in df.columns
        if self.mode == "Duo":
            if self.duo_maternal:
                assert "mat_hap0" in df.columns
                assert "mat_hap1" in df.columns
            else:
                assert "pat_hap0" in df.columns
                assert "pat_hap1" in df.columns
        if "lrr" in df.columns:
            assert "sigma" in df.columns
        return df

    def read_data(self, input_fp):
        """Read in data in either pandas/numpy format.

        Args:
            input_fp (`str`): path to input file.

        Output:
            df (`pd.DataFrame`): pandas dataframe of cleaned options.

        """
        if (".npz" in input_fp) or (".npy" in input_fp):
            df = self.read_data_np(input_fp)
        else:
            df = self.read_data_df(input_fp)
        return df
