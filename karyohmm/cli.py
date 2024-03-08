"""CLI for karyohmm."""
import logging
import sys

import click
import numpy as np
import pandas as pd

from karyohmm import MetaHMM

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Shared type requirements for underlying data
karyo_dtypes = {
    "chrom": str,
    "pos": float,
    "ref": str,
    "alt": str,
    "baf": float,
    "mat_hap0": int,
    "mat_hap1": int,
    "pat_hap0": int,
    "pat_hap1": int,
}


def read_data_np(input_fp):
    """Read data from an .npy or npz file and reformat for karyohmm."""
    data = np.load(input_fp, allow_pickle=True)
    for x in ["chrom", "pos", "ref", "alt", "baf", "mat_haps", "pat_haps"]:
        assert x in data
    df = pd.DataFrame(
        {
            "chrom": data["chrom"],
            "pos": data["pos"],
            "ref": data["ref"],
            "alt": data["alt"],
            "baf": data["baf"],
            "mat_hap0": data["mat_haps"][0, :],
            "mat_hap1": data["mat_haps"][1, :],
            "pat_hap0": data["pat_haps"][0, :],
            "pat_hap1": data["pat_haps"][1, :],
        },
        dtype=karyo_dtypes,
    )
    return df


def read_data_df(input_fp):
    """Read in data from a pre-existing text-based dataset."""
    sep = ","
    if ".tsv" in input_fp:
        sep = "\t"
    elif ".txt" in input_fp:
        sep = " "
    df = pd.read_csv(input_fp, dtype=karyo_dtypes, sep=sep)
    for x in [
        "chrom",
        "pos",
        "ref",
        "alt",
        "baf",
        "mat_hap0",
        "mat_hap1",
        "pat_hap0",
        "pat_hap1",
    ]:
        assert x in df.columns
    return df


def read_data(input_fp):
    """Read in data in either pandas/numpy format."""
    if (".npz" in input_fp) or (".npy" in input_fp):
        df = read_data_np(input_fp)
    else:
        df = read_data_df(input_fp)
    return df


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input data file for PGT-A array intensity data.",
)
@click.option(
    "--viterbi",
    is_flag=True,
    required=False,
    default=False,
    show_default=True,
    type=bool,
    help="Apply the Viterbi algorithm for tracing ploidy.",
)
@click.option(
    "--mode",
    required=True,
    default="Meta",
    type=click.Choice(["Meta"]),
    show_default=True,
)
@click.option(
    "--algo",
    required=False,
    default="Powell",
    type=click.Choice(["Nelder-Mead", "L-BFGS-B", "Powell"]),
    show_default=True,
    help="Method for parameter inference.",
)
@click.option(
    "--recomb_rate",
    "-r",
    required=False,
    default=1e-8,
    type=float,
    show_default=True,
    help="Recombination rate between SNPs.",
)
@click.option(
    "--aneuploidy_rate",
    "-a",
    required=False,
    default=1e-10,
    type=float,
    show_default=True,
    help="Probability of shifting between aneuploidy states between SNPs.",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    required=False,
    type=bool,
    default=True,
    help="Gzip output files",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm",
    help="Output file prefix.",
)
def main(
    input,
    viterbi,
    mode,
    algo="Powell",
    recomb_rate=1e-8,
    aneuploidy_rate=1e-2,
    gzip=True,
    out="karyohmm",
):
    """Karyohmm CLI."""
    logging.info(f"Starting to read input data {input}.")
    data_df = read_data(input)
    assert data_df is not None
    logging.info(f"Finished reading in {input}.")
    if mode == "Meta":
        hmm = MetaHMM()
    else:
        raise NotImplementedError(
            "Meta-HMM is currently the only supported model for karyohmm!"
        )
    # The unique chromosomes present in this dataset and the specific
    uniq_chroms = np.unique(data_df["chrom"])
    kar_dfs = []
    for c in uniq_chroms:
        logging.info(f"Starting inference of karyohmm emission parameters for {c}.")
        cur_df = data_df[data_df["chrom"] == c].sort_values("pos")
        # Defining the numpy objects to test out.
        mat_haps = np.vstack([cur_df.mat_hap0.values, cur_df.mat_hap1.values])
        pat_haps = np.vstack([cur_df.pat_hap0.values, cur_df.pat_hap1.values])
        bafs = cur_df.baf.values
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=bafs,
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            r=recomb_rate,
            a=aneuploidy_rate,
            algo=algo,
        )
        logging.info(f"Finished inference of HMM-parameters for {c}!")
        if viterbi:
            logging.info("Running Viterbi algorithm path tracing.")
            path, states, _, _ = hmm.viterbi(
                bafs=bafs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                r=recomb_rate,
                a=aneuploidy_rate,
            )
            state_lbls = [hmm.get_state_str(s) for s in states]
            n, ns = path.size, len(states)
            path_mat = np.zeros(shape=(n, ns), dtype=np.int32)
            for i, p in enumerate(path):
                path_mat[i, p] = 1
            path_df = pd.DataFrame(path_mat)
            path_df.columns = state_lbls
            path_df["pi0_hat"] = pi0_est
            path_df["sigma_hat"] = sigma_est
            path_df["chrom"] = cur_df.chrom.values
            path_df["pos"] = cur_df.pos.values
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            path_df = path_df[
                cols_to_move
                + [col for col in path_df.columns if col not in cols_to_move]
            ]
            out_fp = (
                f"{out}.{c}.meta.viterbi.tsv.gz"
                if gzip
                else f"{out}.{c}.meta.viterbi.tsv"
            )
            path_df.to_csv(out_fp, sep="\t", index=None)
            logging.info(f"Wrote Viterbi algorithm traceback to {out_fp}")
        else:
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=bafs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
            )
            kar_prob = hmm.posterior_karyotypes(gammas, karyotypes)
            kar_prob["pi0_hat"] = pi0_est
            kar_prob["sigma_hat"] = sigma_est
            kar_prob["chrom"] = c
            df = pd.DataFrame(kar_prob, index=[0])
            kar_dfs.append(df)
            state_lbls = [hmm.get_state_str(s) for s in states]
            gamma_df = pd.DataFrame(gammas.T)
            gamma_df.columns = state_lbls
            gamma_df["chrom"] = cur_df["chrom"].values
            gamma_df["pos"] = cur_df["pos"].values
            gamma_df["pi0_hat"] = pi0_est
            gamma_df["sigma_hat"] = sigma_est
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            gamma_df = gamma_df[
                cols_to_move
                + [col for col in gamma_df.columns if col not in cols_to_move]
            ]
            out_fp = (
                f"{out}.{c}.meta.gammas.tsv.gz"
                if gzip
                else f"{out}.{c}.meta.gammas.tsv"
            )
            gamma_df.to_csv(out_fp, sep="\t", index=None)
            logging.info(f"Wrote forward-backward algorithm results to {out_fp}")
    if not viterbi:
        out_fp = f"{out}.meta.posterior.tsv.gz" if gzip else f"{out}.meta.posterior.tsv"
        kar_df = pd.concat(kar_dfs)
        kar_df.to_csv(out_fp, sep="\t", index=None)
        logging.info(f"Wrote full posterior karyotypes to {out_fp}")
    logging.info("Finished karyohmm analysis!")
