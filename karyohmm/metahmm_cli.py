"""CLI for karyohmm."""

import gzip as gz
import logging

import numpy as np
import polars as pl
import rich_click as click

from karyohmm import DataReader, MetaHMM

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
    algo="Powell",
    thin=1,
    recomb_rate=1e-8,
    aneuploidy_rate=1e-2,
    gzip=False,
    out="karyohmm",
):
    """Karyohmm-Inference CLI."""
    logging.info(f"Starting to read input data {input}...")
    data_reader = DataReader(mode="Meta")
    data_df = data_reader.read_data(input)
    assert data_df is not None
    logging.info(f"Finished reading in {input}.")
    hmm = MetaHMM()
    # Keep accumulators of the various dataframes to be output
    kar_dfs = []
    path_dfs = []
    gamma_dfs = []
    uniq_chroms = data_df["chrom"].unique().sort().to_list()
    for c in uniq_chroms:
        logging.info(f"Starting inference of karyohmm emission parameters for {c}.")
        cur_df = data_df.filter(pl.col("chrom") == c).sort("pos")
        mat_haps = np.vstack(
            [cur_df["mat_hap0"].to_numpy(), cur_df["mat_hap1"].to_numpy()]
        )
        pat_haps = np.vstack(
            [cur_df["pat_hap0"].to_numpy(), cur_df["pat_hap1"].to_numpy()]
        )
        bafs = cur_df["baf"].to_numpy()
        pos = cur_df["pos"].to_numpy()
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
                bafs=bafs,
                pos=pos,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
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
        kar_dfs.append(pl.DataFrame({k: [v] for k, v in kar_prob.items()}))
        state_lbls = [hmm.get_state_str(s) for s in states]
        gamma_df = pl.DataFrame({lbl: gammas[i, :] for i, lbl in enumerate(state_lbls)})
        gamma_df = gamma_df.with_columns(
            cur_df["chrom"],
            cur_df["pos"],
            pl.lit(pi0_est).alias("pi0_hat"),
            pl.lit(sigma_est).alias("sigma_hat"),
        )
        cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
        gamma_df = gamma_df.select(
            cols_to_move + [col for col in gamma_df.columns if col not in cols_to_move]
        )
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
            path_df = pl.DataFrame(
                {lbl: path_mat[:, i] for i, lbl in enumerate(state_lbls)}
            )
            path_df = path_df.with_columns(
                cur_df["chrom"],
                cur_df["pos"],
                pl.lit(pi0_est).alias("pi0_hat"),
                pl.lit(sigma_est).alias("sigma_hat"),
            )
            cols_to_move = ["chrom", "pos", "pi0_hat", "sigma_hat"]
            path_df = path_df.select(
                cols_to_move
                + [col for col in path_df.columns if col not in cols_to_move]
            )
            path_dfs.append(path_df)

    out_fp = f"{out}.meta.posterior.tsv.gz" if gzip else f"{out}.meta.posterior.tsv"
    kar_df = pl.concat(kar_dfs)
    if gzip:
        with gz.open(out_fp, "wb") as f:
            kar_df.write_csv(f, separator="\t")
    else:
        kar_df.write_csv(out_fp, separator="\t")
    logging.info(f"Wrote full posterior karyotypes to {out_fp}!")
    gamma_df = pl.concat(gamma_dfs)
    out_fp = f"{out}.meta.gammas.tsv.gz" if gzip else f"{out}.meta.gammas.tsv"
    if gzip:
        with gz.open(out_fp, "wb") as f:
            gamma_df.write_csv(f, separator="\t")
    else:
        gamma_df.write_csv(out_fp, separator="\t")
    logging.info(f"Wrote per-site forward-backward algorithm results to {out_fp}")
    if viterbi:
        path_df = pl.concat(path_dfs)
        out_fp = f"{out}.meta.viterbi.tsv.gz" if gzip else f"{out}.meta.viterbi.tsv"
        if gzip:
            with gz.open(out_fp, "wb") as f:
                path_df.write_csv(f, separator="\t")
        else:
            path_df.write_csv(out_fp, separator="\t")
        logging.info(f"Wrote Viterbi algorithm traceback to {out_fp}")
    logging.info("Finished karyohmm analysis!")
