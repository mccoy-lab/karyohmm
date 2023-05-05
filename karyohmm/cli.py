import sys

import click
import numpy as np
import pandas as pd

from karyohmm import EuploidyHMM, MetaHMM


def read_data_np(input_fp):
    """Read data from an .npy or npz file and reformat for karyohmm."""
    data = np.load(input_fp)
    for x in ["chrom", "pos", "ref", "alt", "baf", "lrr"]:
        assert x in data
    df = pd.DataFrame(
        {
            "chrom": data["chrom"].values,
            "pos": data["pos"].values,
            "ref": data["ref"].values,
            "alt": data["alt"].values,
            "baf": data["baf"].values,
            "lrr": data["lrr"].values,
        }
    )
    return df


def read_data_df(input_fp):
    """Method to read in data from a pre-existing pandas dataset."""
    sep = ","
    if ".tsv" in input_fp:
        sep = "\t"
    if ".txt" in input_fp:
        sep = "\s+"
    df = pd.read_csv(input_fp, sep=sep, engine="python")
    for x in ["chrom", "pos", "ref", "alt", "baf", "lrr"]:
        assert x in df.columns
    return df


def read_data(input_fp):
    """Combining the two reading methods (numpy/pandas)."""
    try:
        df = read_data_df(input_fp)
    except:
        df = read_data_np(input_fp)
    return df


@click.command()
@click.option(
    "--input", "-i", required=True, type=str, help="Input data file for PGT Data."
)
@click.option("--out", "-o", required=True, type=str, help="Output file.")
@click.option(
    "--logr",
    "-l",
    is_flag=True,
    required=False,
    default=False,
    show_default=True,
    type=bool,
    help="Include LRR information.",
)
@click.option(
    "--viterbi",
    is_flag=True,
    required=False,
    default=False,
    show_default=True,
    type=bool,
    help="Apply a Viterbi algorithm for tracing ploidy.",
)
@click.option(
    "--mode",
    required=True,
    default="Meta",
    type=click.Choice(["Meta", "Euploid"]),
    case_sensitive=False,
    show_default=True,
)
def main(input, out, logr, viterbi, mode):
    """Main CLI entrypoint for calling karyohmm."""
    print(f"Reading in input data {input} ...", file=sys.stderr)
    data_df = read_data(input)
    assert data_df is not None
    print(f"Finished reading in {input}", file=sys.stderr)
    if mode == "Meta":
        hmm = MetaHMM(logr=logr)
    else:
        hmm = EuploidyHMM()
    print(f"Inference of HMM-parameters ...", file=sys.stderr)
    # TODO :place the actual inference here ...
    pass
