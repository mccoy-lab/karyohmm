"""CLI for karyohmm."""
import logging
import sys

import click
import numpy as np
import pandas as pd

from karyohmm import DataReader, DuoHMM, MetaHMM, RecombEst

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    help="Viterbi algorithm for tracing ploidy states.",
)
@click.option(
    "--mode",
    required=True,
    default="Meta",
    type=click.Choice(["Meta", "Duo"]),
    show_default=True,
)
@click.option(
    "--algo",
    required=False,
    default="Powell",
    type=click.Choice(["Nelder-Mead", "L-BFGS-B", "Powell"]),
    show_default=True,
    help="Optimization method for parameter inference.",
)
@click.option(
    "--thin",
    required=False,
    default=1,
    type=int,
    show_default=True,
    help="SNP thinning to improve optimization speed for parameter inference.",
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
    default=1e-2,
    type=float,
    show_default=True,
    help="Probability of shifting between aneuploidy states between SNPs.",
)
@click.option(
    "--duo_maternal",
    "-dm",
    required=False,
    default=None,
    type=bool,
    show_default=True,
    help="Indicator of duo being mother-child duo.",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    required=False,
    type=bool,
    default=False,
    help="Gzip output files.",
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
    viterbi=False,
    mode="Meta",
    algo="Powell",
    thin=1,
    recomb_rate=1e-8,
    aneuploidy_rate=1e-2,
    duo_maternal=None,
    gzip=False,
    out="karyohmm",
):
    """Karyohmm-Inference CLI."""
    logging.info(f"Starting to read input data {input}...")
    data_reader = DataReader(mode=mode, duo_maternal=duo_maternal)
    data_df = data_reader.read_data(input)
    assert data_df is not None
    logging.info(f"Finished reading in {input}.")
    if mode == "Meta":
        hmm = MetaHMM()
    elif mode == "Duo":
        hmm = DuoHMM()
    else:
        raise NotImplementedError(
            f"Mode {mode} is not currently supported  in karyoHMM!"
        )
    # Keep accumulators of the various dataframes to be output
    kar_dfs = []
    path_dfs = []
    gamma_dfs = []
    # The unique chromosomes present in this dataset and the specific
    uniq_chroms = np.unique(data_df["chrom"])
    for c in uniq_chroms:
        logging.info(f"Starting inference of karyohmm emission parameters for {c}.")
        cur_df = data_df[data_df["chrom"] == c].sort_values("pos")
        if mode == "Meta":
            # Defining the numpy objects to test out.
            mat_haps = np.vstack([cur_df.mat_hap0.values, cur_df.mat_hap1.values])
            pat_haps = np.vstack([cur_df.pat_hap0.values, cur_df.pat_hap1.values])
            bafs = cur_df.baf.values
            pos = cur_df.pos.values
            if thin > 1:
                pi0_est, sigma_est = hmm.est_sigma_pi0(
                    bafs=bafs[::thin],
                    pos=pos[::thin],
                    mat_haps=mat_haps[:, ::thin],
                    pat_haps=pat_haps[:, ::thin],
                    r=recomb_rate,
                    a=aneuploidy_rate,
                    algo=algo,
                )
            else:
                pi0_est, sigma_est = hmm.est_sigma_pi0(
                    bafs=bafs[::thin],
                    pos=pos[::thin],
                    mat_haps=mat_haps[:, ::thin],
                    pat_haps=pat_haps[:, ::thin],
                    r=recomb_rate,
                    a=aneuploidy_rate,
                    algo=algo,
                )
            logging.info(
                f"MetaHMM emission parameters are pi0={pi0_est:.3f}, sigma={sigma_est:.3f} for {c}."
            )
            logging.info(f"Finished inference of HMM-parameters for {c}!")
            logging.info(f"Starting Forward-Backward algorithm tracing for {c} ...")
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=bafs,
                pos=pos,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0_est,
                std_dev=sigma_est,
                r=recomb_rate,
                a=aneuploidy_rate,
            )
            logging.info(f"Finished  Forward-Backward algorithm for {c}!")
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
            gamma_dfs.append(gamma_df)
            if viterbi:
                logging.info(f"Starting Viterbi algorithm tracing for {c} ...")
                path, states, _, _ = hmm.viterbi(
                    bafs=bafs,
                    pos=pos,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pi0=pi0_est,
                    std_dev=sigma_est,
                    r=recomb_rate,
                    a=aneuploidy_rate,
                )
                logging.info(f"Finished Viterbi algorithm tracing for {c}!")
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
                path_dfs.append(path_df)
        if mode == "Duo":
            # Defining the numpy objects to test out.
            if data_reader.duo_maternal:
                haps = np.vstack([cur_df.mat_hap0.values, cur_df.mat_hap1.values])
            else:
                haps = np.vstack([cur_df.pat_hap0.values, cur_df.pat_hap1.values])
            bafs = cur_df.baf.values
            pos = cur_df.pos.values
            ps = None
            if "af" in cur_df.columns:
                if not np.any(np.isnan(cur_df["af"].values)):
                    ps = cur_df["af"].values
            if thin > 1:
                pi0_est, sigma_est = hmm.est_sigma_pi0(
                    bafs=bafs[::thin],
                    haps=haps[:, ::thin],
                    pos=pos[::thin],
                    freqs=ps if ps is None else ps[::thin],
                    r=recomb_rate,
                    a=aneuploidy_rate,
                    algo=algo,
                )
            else:
                pi0_est, sigma_est = hmm.est_sigma_pi0(
                    bafs=bafs,
                    haps=haps,
                    pos=pos,
                    freqs=ps,
                    r=recomb_rate,
                    a=aneuploidy_rate,
                    algo=algo,
                )
            logging.info(
                f"DuoHMM emission parameters are pi0={pi0_est:.3f}, sigma={sigma_est:.3f} for {c}."
            )
            logging.info(f"Finished inference of DuoHMM-parameters for {c}!")
            logging.info(f"Starting Forward-Backward algorithm tracing for {c} ...")
            gammas, states, karyotypes = hmm.forward_backward(
                bafs=bafs,
                pos=pos,
                haps=haps,
                freqs=ps,
                pi0=pi0_est,
                std_dev=sigma_est,
                r=recomb_rate,
                a=aneuploidy_rate,
            )
            logging.info(f"Finished  Forward-Backward algorithm for {c}!")
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
            gamma_dfs.append(gamma_df)

    if mode == "Meta":
        out_fp = f"{out}.meta.posterior.tsv.gz" if gzip else f"{out}.meta.posterior.tsv"
        kar_df = pd.concat(kar_dfs)
        kar_df.to_csv(out_fp, sep="\t", index=None)
        logging.info(f"Wrote full posterior karyotypes to {out_fp}!")
        gamma_df = pd.concat(gamma_dfs)
        out_fp = f"{out}.meta.gammas.tsv.gz" if gzip else f"{out}.meta.gammas.tsv"
        gamma_df.to_csv(out_fp, sep="\t", index=None)
        logging.info(f"Wrote per-site forward-backward algorithm results to {out_fp}")
        if viterbi:
            path_df = pd.concat(path_dfs)
            out_fp = f"{out}.meta.viterbi.tsv.gz" if gzip else f"{out}.meta.viterbi.tsv"
            path_df.to_csv(out_fp, sep="\t", index=None)
            logging.info(f"Wrote Viterbi algorithm traceback to {out_fp}")
    if mode == "Duo":
        out_fp = f"{out}.duo.posterior.tsv.gz" if gzip else f"{out}.duo.posterior.tsv"
        kar_df = pd.concat(kar_dfs)
        kar_df.to_csv(out_fp, sep="\t", index=None)
        logging.info(f"Wrote full posterior karyotypes to {out_fp}")
        gamma_df = pd.concat(gamma_dfs)
        out_fp = f"{out}.duo.gammas.tsv.gz" if gzip else f"{out}.duo.gammas.tsv"
        gamma_df.to_csv(out_fp, sep="\t", index=None)
        logging.info(f"Wrote per-site forward-backward algorithm results to {out_fp}")
    logging.info("Finished karyohmm analysis!")
