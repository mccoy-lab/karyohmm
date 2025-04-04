"""Classes for reading in data for karyoHMM.

Implements methods for both MetaHMM and DuoHMM-based datasets.
"""

import numpy as np
import pandas as pd


class DataReader:
    """Input / output class for reading in data for karyohmm."""

    def __init__(self, mode="Meta", duo_maternal=None):
        """Initialize the DataReader class for a given mode."""
        assert mode in ["Meta", "Duo", "Recomb"]
        karyo_dtypes = {
            "chrom": str,
            "pos": float,
            "ref": str,
            "alt": str,
            "baf": float,
            "af": float,
            "mat_hap0": int,
            "mat_hap1": int,
            "pat_hap0": int,
            "pat_hap1": int,
        }
        self.dtypes = karyo_dtypes
        self.mode = mode
        if (mode == "Duo") and (type(duo_maternal) is not bool):
            raise ValueError(
                "Need to specify whether a mother-child or father-child duo!"
            )
        if (duo_maternal is not None) and (mode == "Duo"):
            self.duo_maternal = duo_maternal

    def read_data_np(self, input_fp):
        """Read data from an .npy or npz file and reformat for karyohmm.

        Args:
            input_fp (`str`): path to input NPZ or NPY file.

        Output:
            df (`pd.DataFrame`): pandas dataframe of cleaned options.

        """
        data = np.load(input_fp, allow_pickle=True)
        for x in ["chrom", "pos", "ref", "alt", "baf"]:
            assert x in data
        if self.mode != "Duo":
            assert "mat_haps" in data
            assert "pat_haps" in data
            df = pd.DataFrame(
                {
                    "chrom": data["chrom"].tolist(),
                    "pos": data["pos"].tolist(),
                    "ref": data["ref"].tolist(),
                    "alt": data["alt"].tolist(),
                    "baf": data["baf"].tolist(),
                    "mat_hap0": data["mat_haps"][0, :].tolist(),
                    "mat_hap1": data["mat_haps"][1, :].tolist(),
                    "pat_hap0": data["pat_haps"][0, :].tolist(),
                    "pat_hap1": data["pat_haps"][1, :].tolist(),
                }
            )
            if "af" in data:
                df["af"] = data["af"]
            df = df.astype(dtype=self.dtypes)
            return df
        if self.mode == "Duo":
            if self.duo_maternal:
                assert "mat_haps" in data
                df = pd.DataFrame(
                    {
                        "chrom": data["chrom"],
                        "pos": data["pos"],
                        "ref": data["ref"],
                        "alt": data["alt"],
                        "baf": data["baf"],
                        "mat_hap0": data["mat_haps"][0, :],
                        "mat_hap1": data["mat_haps"][1, :],
                    },
                )
                if "af" in data:
                    df["af"] = data["af"]
                df = df.astype(dtype=self.dtypes)
                return df
            else:
                assert "pat_haps" in data
                df = pd.DataFrame(
                    {
                        "chrom": data["chrom"],
                        "pos": data["pos"],
                        "ref": data["ref"],
                        "alt": data["alt"],
                        "baf": data["baf"],
                        "pat_hap0": data["pat_haps"][0, :],
                        "pat_hap1": data["pat_haps"][1, :],
                    },
                )
                if "af" in data:
                    df["af"] = data["af"]
                df = df.astype(dtype=self.dtypes)
            return df

    def read_data_df(self, input_fp):
        """Read in data from a pre-existing text-based dataset.

        Args:
            input_fp (`str`): path to input TSV/CSV/TXT file.

        Output:
            df (`pd.DataFrame`): pandas dataframe of cleaned options.

        """
        sep = ","
        if ".tsv" in input_fp:
            sep = "\t"
        elif ".txt" in input_fp:
            sep = " "
        df = pd.read_csv(input_fp, dtype=self.dtypes, sep=sep)
        for x in [
            "chrom",
            "pos",
            "ref",
            "alt",
            "baf",
        ]:
            assert x in df.columns
        if self.mode != "Duo":
            for x in ["mat_hap0", "mat_hap1", "pat_hap0", "pat_hap1"]:
                assert x in df.columns
        if self.mode == "Duo":
            if self.duo_maternal:
                assert "mat_hap0" in df.columns
                assert "mat_hap1" in df.columns
            else:
                assert "pat_hap0" in df.columns
                assert "pat_hap1" in df.columns
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
