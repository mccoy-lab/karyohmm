"""CLI for karyohmm."""
import sys

import click
import numpy as np
import pandas as pd

from karyohmm import MetaHMM


def read_data_np(input_fp):
    """Read data from an .npy or npz file and reformat for karyohmm."""
    data = np.load(input_fp)
    for x in ["chrom", "pos", "ref", "alt", "baf", "lrr", "mat_haps", "pat_haps"]:
        assert x in data
    df = pd.DataFrame(
        {
            "chrom": data["chrom"],
            "pos": data["pos"],
            "ref": data["ref"],
            "alt": data["alt"],
            "baf": data["baf"],
            "lrr": data["lrr"],
            "mat_hap0": data["mat_haps"][0, :],
            "mat_hap1": data["mat_haps"][1, :],
            "pat_hap0": data["pat_haps"][0, :],
            "pat_hap1": data["pat_haps"][1, :],
        }
    )
    return df


def read_data_df(input_fp):
    """Read in data from a pre-existing text-based dataset."""
    sep = ","
    if ".tsv" in input_fp:
        sep = "\t"
    elif ".txt" in input_fp:
        sep = " "
    df = pd.read_csv(input_fp, sep=sep)
    for x in [
        "chrom",
        "pos",
        "ref",
        "alt",
        "baf",
        "lrr",
        "mat_hap0",
        "mat_hap1",
        "pat_hap0",
        "pat_hap1",
    ]:
        assert x in df.columns
    return df


def read_data(input_fp):
    """Read in data in either pandas/numpy format."""
    try:
        df = read_data_df(input_fp)
    except Exception:
        df = read_data_np(input_fp)
    return df


@click.command()
@click.option(
    "--input", "-i", required=True, type=str, help="Input data file for PGT Data."
)
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
    "--unphased",
    required=False,
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--eps",
    required=False,
    default=1e-6,
    type=float,
    show_default=True,
)
@click.option(
    "--niter",
    required=False,
    default=50,
    type=int,
    show_default=True,
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm",
    help="Output file prefix.",
)
def main(input, logr, viterbi, mode, unphased, eps, niter, out):
    """Karyohmm CLI."""
    print(f"Reading in input data {input} ...", file=sys.stderr)
    data_df = read_data(input)
    assert data_df is not None
    print(f"Finished reading in {input}.", file=sys.stderr)
    if mode == "Meta":
        hmm = MetaHMM(logr=logr)
    else:
        raise NotImplementedError("Meta-HMM is currently the only supported mode!")
    print("Inference of HMM-parameters ...", file=sys.stderr)
    # Defining the numpy objects to test out.
    mat_haps = np.vstack([data_df.mat_hap0.values, data_df.mat_hap1.values])
    pat_haps = np.vstack([data_df.pat_hap0.values, data_df.pat_hap1.values])
    bafs = data_df.baf.values
    lrrs = data_df.lrr.values
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=bafs,
        lrrs=lrrs,
        mat_haps=mat_haps,
        pat_haps=pat_haps,
        unphased=unphased,
        logr=logr,
        eps=eps,
    )
    pi0_lrr = np.nan
    lrr_mu = np.nan
    lrr_sd = np.nan
    if logr:
        pi0_lrr, lrr_mu, lrr_sd, _ = hmm.est_lrr_sd(lrrs, niter=niter)
    print("Finished inference of HMM-parameters!", file=sys.stderr)
    print("Running analyses ... ", file=sys.stderr)
    if viterbi:
        if mode == "Meta":
            path, states, _, _ = hmm.viterbi(
                bafs=bafs,
                lrrs=lrrs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                pi0_lrr=pi0_lrr,
                lrr_mu=lrr_mu,
                lrr_sd=lrr_sd,
                eps=eps,
                unphased=unphased,
                logr=logr,
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
            path_df["chrom"] = data_df.chrom.values
            path_df["pos"] = data_df.pos.values
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            path_df = path_df[
                cols_to_move
                + [col for col in path_df.columns if col not in cols_to_move]
            ]
            path_df.to_csv(f"{out}.meta.viterbi.tsv", sep="\t", index=None)
            print(f"Wrote viterbi algorithm traceback to {out}.meta.viterbi.tsv")
        else:
            path, states, _, _ = hmm.viterbi(
                bafs=bafs,
                lrrs=lrrs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                eps=eps,
                unphased=unphased,
                logr=logr,
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
            path_df["chrom"] = data_df.chrom.values
            path_df["pos"] = data_df.pos.values
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            path_df = path_df[
                cols_to_move
                + [col for col in path_df.columns if col not in cols_to_move]
            ]
            path_df.to_csv(f"{out}.disomy.viterbi.tsv", sep="\t", index=None)
            print(f"Wrote viterbi algorithm traceback to {out}.disomy.viterbi.tsv")
    else:
        if mode == "Meta":
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=bafs,
                lrrs=lrrs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                pi0_lrr=pi0_lrr,
                lrr_mu=lrr_mu,
                lrr_sd=lrr_sd,
                eps=eps,
                unphased=unphased,
                logr=logr,
            )
            kar_prob = hmm.posterior_karyotypes(gammas, karyotypes)
            kar_prob["pi0_hat"] = pi0_est
            kar_prob["sigma_hat"] = sigma_est
            df = pd.DataFrame(kar_prob, index=[0])
            df.to_csv(f"{out}.meta.posterior.tsv", sep="\t", index=None)
            print(
                f"Wrote posterior karyotypes to {out}.meta.posterior.tsv",
                file=sys.stderr,
            )

            state_lbls = [hmm.get_state_str(s) for s in states]
            gamma_df = pd.DataFrame(gammas.T)
            gamma_df.columns = state_lbls
            gamma_df["chrom"] = data_df["chrom"].values
            gamma_df["pos"] = data_df["pos"].values
            gamma_df["pi0_hat"] = pi0_est
            gamma_df["sigma_hat"] = sigma_est
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            gamma_df = gamma_df[
                cols_to_move
                + [col for col in gamma_df.columns if col not in cols_to_move]
            ]
            gamma_df.to_csv(f"{out}.meta.gammas.tsv", sep="\t", index=None)
            print(
                f"Wrote forward-backward algorithm results to {out}.meta.gammas.tsv",
                file=sys.stderr,
            )

        else:
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=bafs,
                lrrs=lrrs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                eps=eps,
                unphased=unphased,
                logr=logr,
            )
            kar_prob = hmm.posterior_karyotypes(gammas, karyotypes)
            kar_prob["pi0_hat"] = pi0_est
            kar_prob["sigma_hat"] = sigma_est
            df = pd.DataFrame(kar_prob, index=[0])
            df.to_csv(f"{out}.disomy.posterior.tsv", sep="\t", index=None)
            print(
                f"Wrote posterior karyotypes to {out}.disomy.posterior.tsv",
                file=sys.stderr,
            )

            state_lbls = [hmm.get_state_str(s) for s in states]
            gamma_df = pd.DataFrame(gammas.T)
            gamma_df.columns = state_lbls
            gamma_df["chrom"] = data_df["chrom"].values
            gamma_df["pos"] = data_df["pos"].values
            gamma_df["pi0_hat"] = pi0_est
            gamma_df["sigma_hat"] = sigma_est
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            gamma_df = gamma_df[
                cols_to_move
                + [col for col in gamma_df.columns if col not in cols_to_move]
            ]
            gamma_df.to_csv(f"{out}.disomy.gammas.tsv", sep="\t", index=None)
            print(
                f"Wrote forward-backward algorithm results to {out}.meta.gammas.tsv",
                file=sys.stderr,
            )

    print("Finished karyohmm analysis!", file=sys.stderr)
