"""CLI for mosaic cell-fraction estimation using MosaicEst."""

import gzip as gz
import logging

import numpy as np
import polars as pl
import rich_click as click
from scipy.stats import chi2

from karyohmm import DataReader, MosaicEst

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _parse_region(region_str):
    """Parse 'CHR:START-END' into (chrom, start, end) with 1-based inclusive positions.

    Raises click.BadParameter on malformed input.
    """
    try:
        chrom, coords = region_str.rsplit(":", 1)
        start_s, end_s = coords.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start < 1 or end < start:
            raise ValueError
        return chrom, start, end
    except (ValueError, AttributeError):
        raise click.BadParameter(
            f"Region must be CHR:START-END with 1-based inclusive positions "
            f"(e.g. chr21:15000000-46700000), got: {region_str!r}"
        )


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input data file (NPZ or TSV/CSV) with BAF, LRR, and parental haplotypes.",
)
@click.option(
    "--chrom",
    "-c",
    multiple=True,
    default=(),
    type=str,
    help=(
        "Restrict analysis to these chromosome(s). "
        "Repeatable: -c chr21 -c chr18. "
        "Default: all chromosomes in input (or inferred from --region)."
    ),
)
@click.option(
    "--region",
    "-r",
    multiple=True,
    default=(),
    type=str,
    help=(
        "Restrict analysis to a genomic region in CHR:START-END format "
        "(1-based inclusive positions). Repeatable for multiple regions. "
        "When supplied, only sites within the region(s) are passed to MosaicEst, "
        "giving an unbiased cell-fraction estimate for that interval rather than "
        "a chromosome-wide average. The output label reflects the region coordinates. "
        "Example: -r chr21:15000000-46700000"
    ),
)
@click.option(
    "--std-dev-baf",
    required=False,
    default=0.1,
    type=float,
    show_default=True,
    help="BAF emission noise standard deviation at expected-het sites.",
)
@click.option(
    "--switch-err",
    required=False,
    default=0.01,
    type=float,
    show_default=True,
    help="Within-type maternal<->paternal origin-switch probability.",
)
@click.option(
    "--t-rate",
    required=False,
    default=1e-4,
    type=float,
    show_default=True,
    help="Neutral<->aneuploid transition probability per site.",
)
@click.option(
    "--alpha",
    required=False,
    default=0.05,
    type=float,
    show_default=True,
    help="LRT significance threshold for flagging mosaicism.",
)
@click.option(
    "--min-het",
    required=False,
    default=10,
    type=int,
    show_default=True,
    help="Minimum expected-het sites required; regions below this are skipped.",
)
@click.option(
    "--no-lrr",
    is_flag=True,
    default=False,
    help="Ignore LRR data even when present in input (BAF-only mode).",
)
@click.option(
    "--gammas",
    is_flag=True,
    default=False,
    help="Write per-site log forward-variable file for each state.",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    default=False,
    help="Gzip all output files.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm_mosaic",
    help="Output file prefix.",
)
def main(
    input,
    chrom,
    region,
    std_dev_baf,
    switch_err,
    t_rate,
    alpha,
    min_het,
    no_lrr,
    gammas,
    gzip,
    out,
):
    """Estimate mosaic cell fraction and parental origin per chromosome or region.

    Runs a 5-state joint BAF + LRR HMM (MosaicEst) on each target and reports
    the MLE cell fraction, 95% CI, inferred parental origin, and a
    likelihood-ratio test for mosaicism.

    States: neutral | maternal-gain | paternal-gain | maternal-loss | paternal-loss

    Without --region, the full chromosome is analysed and mle_cf is a
    chromosome-wide average.  With --region, only sites within the specified
    interval are used, giving an unbiased estimate for that segment — suitable
    for quantifying mosaicism identified in a specific arm or interval by a
    prior MetaHMM run.
    """
    logging.info(f"Reading input data from {input} ...")
    data_reader = DataReader(mode="Meta")
    data_df = data_reader.read_data(input)
    logging.info("Finished reading input.")

    has_lrr = ("lrr" in data_df.columns) and ("sigmas" in data_df.columns) and (not no_lrr)
    if not has_lrr:
        logging.warning(
            "LRR/sigmas columns not found (or --no-lrr set). "
            "Running in BAF-only mode — detection power will be reduced."
        )

    # Parse region strings into (chrom, start, end) tuples
    parsed_regions = [_parse_region(r) for r in region]

    # Build the list of (chrom, start, end, label) jobs to run.
    # Each job is one call to MosaicEst and one row in the output.
    jobs = []
    if parsed_regions:
        # Regions drive the job list; --chrom is ignored if --region is given.
        if chrom:
            logging.warning(
                "--chrom is ignored when --region is supplied; "
                "target chromosomes are inferred from the region(s)."
            )
        for r_chrom, r_start, r_end in parsed_regions:
            jobs.append((r_chrom, r_start, r_end, f"{r_chrom}:{r_start}-{r_end}"))
    else:
        # Whole-chromosome mode
        target_chroms = list(chrom) if chrom else data_df["chrom"].unique().sort().to_list()
        for c in target_chroms:
            jobs.append((c, None, None, c))

    logging.info(f"Queued {len(jobs)} analysis job(s).")

    summary_rows = []
    gamma_dfs = []

    for job_chrom, job_start, job_end, job_label in jobs:
        logging.info(f"[{job_label}] Starting mosaic estimation ...")
        cur_df = data_df.filter(pl.col("chrom") == job_chrom).sort("pos")

        if cur_df.is_empty():
            logging.warning(f"[{job_label}] No data found on {job_chrom} — skipping.")
            continue

        # Apply positional filter when a region was specified
        if job_start is not None:
            cur_df = cur_df.filter(
                (pl.col("pos") >= job_start) & (pl.col("pos") <= job_end)
            )
            if cur_df.is_empty():
                logging.warning(
                    f"[{job_label}] No sites in {job_chrom}:{job_start}-{job_end} — skipping."
                )
                continue
            logging.info(
                f"[{job_label}] Region filter applied: "
                f"{len(cur_df)} sites retained between {job_start} and {job_end}."
            )

        mat_haps = np.vstack([cur_df["mat_hap0"].to_numpy(), cur_df["mat_hap1"].to_numpy()])
        pat_haps = np.vstack([cur_df["pat_hap0"].to_numpy(), cur_df["pat_hap1"].to_numpy()])
        bafs = cur_df["baf"].to_numpy()
        pos = cur_df["pos"].to_numpy()
        lrrs = cur_df["lrr"].to_numpy() if has_lrr else None
        sigmas = cur_df["sigmas"].to_numpy() if has_lrr else None

        try:
            m_est = MosaicEst(
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                bafs=bafs,
                pos=pos,
                lrrs=lrrs,
                sigmas=sigmas,
                switch_err=switch_err,
                t_rate=t_rate,
            )
        except ValueError as e:
            logging.warning(f"[{job_label}] Skipping — {e}")
            continue

        if m_est.n_het < min_het:
            logging.warning(
                f"[{job_label}] Only {m_est.n_het} expected-het sites "
                f"(< --min-het {min_het}) — skipping."
            )
            continue

        logging.info(
            f"[{job_label}] Phasing complete. "
            f"{m_est.n_het} expected-het sites, {pos.size} total sites."
        )
        logging.info(f"[{job_label}] Running MLE optimisation ...")
        m_est.est_mle_cf(std_dev_baf=std_dev_baf)

        if np.isnan(m_est.mle_cf):
            logging.warning(f"[{job_label}] MLE optimisation failed — recording NaN.")
            summary_rows.append({
                "chrom": job_chrom,
                "region_start": job_start,
                "region_end": job_end,
                "n_sites": pos.size,
                "n_het": m_est.n_het,
                "mle_cf": float("nan"),
                "cf_lower": float("nan"),
                "cf_upper": float("nan"),
                "origin": "unknown",
                "lrt": float("nan"),
                "lrt_pval": float("nan"),
                "significant": False,
                "std_dev_baf": std_dev_baf,
            })
            continue

        ci = m_est.ci_mle_cf(std_dev_baf=std_dev_baf)
        lrt_stat = m_est.lrt_cf(std_dev_baf=std_dev_baf)
        lrt_pval = float(chi2.sf(lrt_stat, df=1))
        origin = m_est.infer_origin(std_dev_baf=std_dev_baf)
        significant = lrt_pval < alpha

        logging.info(
            f"[{job_label}] mle_cf={m_est.mle_cf:.4f} "
            f"CI=[{ci[0]:.4f}, {ci[2]:.4f}] "
            f"origin={origin} "
            f"LRT={lrt_stat:.2f} p={lrt_pval:.2e} "
            f"{'*significant*' if significant else 'not significant'}"
        )

        summary_rows.append({
            "chrom": job_chrom,
            "region_start": job_start,
            "region_end": job_end,
            "n_sites": pos.size,
            "n_het": m_est.n_het,
            "mle_cf": m_est.mle_cf,
            "cf_lower": ci[0],
            "cf_upper": ci[2],
            "origin": origin,
            "lrt": lrt_stat,
            "lrt_pval": lrt_pval,
            "significant": significant,
            "std_dev_baf": std_dev_baf,
        })

        if gammas:
            alphas_arr, _, _ = m_est.forward_algo_full(
                cf=m_est.mle_cf, std_dev_baf=std_dev_baf
            )
            gamma_df = pl.DataFrame({
                state: alphas_arr[k, :]
                for k, state in enumerate(MosaicEst.STATE_NAMES)
            }).with_columns(
                pl.lit(job_chrom).alias("chrom"),
                pl.lit(job_label).alias("region"),
                pl.Series("pos", pos),
                pl.lit(m_est.mle_cf).alias("mle_cf"),
            )
            cols_first = ["chrom", "region", "pos", "mle_cf"]
            gamma_df = gamma_df.select(
                cols_first + [col for col in gamma_df.columns if col not in cols_first]
            )
            gamma_dfs.append(gamma_df)

    if not summary_rows:
        logging.warning("No jobs completed successfully. No output written.")
        return

    # Write per-job summary; cast region columns to Int64 so whole-chromosome
    # (null) and region runs share the same schema when outputs are combined.
    summary_df = pl.DataFrame(summary_rows).with_columns(
        pl.col("region_start").cast(pl.Int64),
        pl.col("region_end").cast(pl.Int64),
    )
    ext = ".tsv.gz" if gzip else ".tsv"
    summary_fp = f"{out}.mosaic.summary{ext}"
    if gzip:
        with gz.open(summary_fp, "wb") as f:
            summary_df.write_csv(f, separator="\t")
    else:
        summary_df.write_csv(summary_fp, separator="\t")
    logging.info(f"Wrote mosaic summary to {summary_fp}")

    # Write per-site gammas if requested
    if gammas and gamma_dfs:
        gamma_fp = f"{out}.mosaic.gammas{ext}"
        all_gammas = pl.concat(gamma_dfs)
        if gzip:
            with gz.open(gamma_fp, "wb") as f:
                all_gammas.write_csv(f, separator="\t")
        else:
            all_gammas.write_csv(gamma_fp, separator="\t")
        logging.info(f"Wrote per-site forward-variable log-probabilities to {gamma_fp}")

    logging.info("Mosaic estimation complete.")
