"""CLI for simulating synthetic aneuploidy data in karyoHMM."""
import gzip as gz
import logging
import sys

import click
import numpy as np
import pandas as pd

from karyohmm import PGTSim, PGTSimMosaic, PGTSimSegmental, PGTSimVCF

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--mode",
    required=True,
    default="Whole-Chromosome",
    type=click.Choice(["Whole-Chromosome", "Segmental", "Mosaic"]),
    show_default=True,
)
@click.option(
    "--chrom",
    "-c",
    required=False,
    default="chr1",
    type=str,
    show_default=True,
    help="Chromosome indicator.",
)
@click.option(
    "--afs",
    "-a",
    required=False,
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="Allele frequency file for variants (to mimic ascertainment-bias).",
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
    "--vcf",
    "-v",
    required=False,
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="VCF as input for parental haplotype data.",
)
@click.option(
    "--maternal_id",
    required=False,
    default=None,
    type=str,
    show_default=True,
    help="IID of maternal individual in VCF",
)
@click.option(
    "--paternal_id",
    required=False,
    default=None,
    type=str,
    show_default=True,
    help="IID of paternal individual in VCF.",
)
@click.option(
    "--length",
    "-l",
    required=False,
    default=50e6,
    type=float,
    show_default=True,
    help="Length of segment to simulate.",
)
@click.option(
    "--ploidy",
    "-p",
    required=True,
    default="2",
    type=click.Choice(["0", "1", "2", "3"]),
    show_default=True,
    help="Degree of aneuploidy to be simulated.",
)
@click.option(
    "--m",
    "-m",
    required=False,
    default=5000,
    type=int,
    show_default=True,
    help="Number of variants to simulate on chromosome.",
)
@click.option(
    "--std_dev",
    required=True,
    default=0.2,
    type=float,
    show_default=True,
    help="Standard deviation of BAF-distribution.",
)
@click.option(
    "--pi0",
    required=True,
    default=0.5,
    type=float,
    show_default=True,
    help="Point-mass for emission distribution of BAF.",
)
@click.option(
    "--mat_skew",
    required=True,
    default=0.5,
    type=float,
    show_default=True,
    help="Probability of being a maternal-origin aneuploidy.",
)
@click.option(
    "--mean_size",
    required=False,
    default=100,
    type=int,
    show_default=True,
    help="Mean size of a segmental aneuploidy on the chromosome.",
)
@click.option(
    "--switch_err_rate",
    "-se",
    required=False,
    default=1e-2,
    type=float,
    show_default=True,
    help="Switch error rate in parental haplotypes.",
)
@click.option(
    "--seed",
    required=True,
    default=42,
    type=int,
    show_default=True,
    help="Random seed for simulation.",
)
@click.option(
    "--threads",
    required=False,
    default=1,
    type=int,
    show_default=True,
    help="VCF reading threads.",
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
@click.option(
    "--format",
    "-fmt",
    required=True,
    type=click.Choice(["tsv", "npz"]),
    default="tsv",
    help="Output file format.",
)
def main(
    mode="Whole-Chromosome",
    chrom="chr1",
    afs=None,
    vcf=None,
    maternal_id=None,
    paternal_id=None,
    ploidy="2",
    recomb_rate=1e-8,
    aneuploidy_rate=1e-2,
    length=50e6,
    m=5000,
    std_dev=0.2,
    pi0=0.5,
    mat_skew=0.5,
    duo_maternal=None,
    mean_size=100,
    switch_err_rate=1e-2,
    seed=42,
    threads=1,
    gzip=False,
    out="karyohmm",
    format="tsv",
):
    """Karyohmm-Simulator CLI."""
    logging.info("Starting simulation ...")
    vcf_haps = False
    if vcf is not None:
        logging.info(f"Reading in parental haplotypes from VCF: {vcf}")
        if (maternal_id is None) or (paternal_id is None):
            raise ValueError(
                "Need to specify both `maternal_id` and `paternal_id` if simulating from a VCF!"
            )
        pgt_sim_vcf = PGTSimVCF()
        mat_haps, pat_haps, pos, ps = pgt_sim_vcf.gen_parental_haplotypes(
            vcf_fp=vcf,
            maternal_id=maternal_id,
            paternal_id=paternal_id,
            gts012=True,
            threads=threads,
        )
        logging.info(
            f"Finished extracting haplotypes from {maternal_id}, {paternal_id} in {vcf}!"
        )
        vcf_haps = True
    if mode == "Whole-Chromosome":
        logging.info("Simulating whole-chromosome aneuploidy ...")
        if vcf_haps:
            logging.info("Starting whole-chromosome aneuploidy simulation ...")
            pgt_sim = PGTSim()
            results = pgt_sim.sim_from_haps(
                mat_haps,
                pat_haps,
                pos,
                ploidy=int(ploidy),
                rec_rate=recomb_rate,
                mat_skew=mat_skew,
                std_dev=std_dev,
                mix_prop=pi0,
                alpha=1.0,
                switch_err_rate=switch_err_rate,
                seed=seed,
            )
            results["ps"] = ps
        else:
            pgt_sim = PGTSim()
            results = pgt_sim.full_ploidy_sim(
                afs=afs,
                ploidy=int(ploidy),
                m=m,
                length=length,
                rec_rate=recomb_rate,
                mat_skew=mat_skew,
                std_dev=std_dev,
                mix_prop=pi0,
                alpha=1.0,
                switch_err_rate=switch_err_rate,
                seed=seed,
            )
    elif mode == "Segmental":
        if vcf_haps:
            pgt_sim = PGTSimSegmental()
            results = pgt_sim.sim_from_haps(
                mat_haps,
                pat_haps,
                pos,
                ploidy=int(ploidy),
                rec_rate=recomb_rate,
                mat_skew=mat_skew,
                std_dev=std_dev,
                mix_prop=pi0,
                mean_size=mean_size,
                switch_err_rate=switch_err_rate,
                seed=seed,
            )
            results["ps"] = ps
        else:
            pgt_sim = PGTSimSegmental()
            results = pgt_sim.full_segmental_sim(
                afs=afs,
                ploidy=int(ploidy),
                m=m,
                length=length,
                rec_rate=recomb_rate,
                mat_skew=mat_skew,
                std_dev=std_dev,
                mix_prop=pi0,
                mean_size=mean_size,
                switch_err_rate=switch_err_rate,
                seed=seed,
            )
    elif mode == "Mosaic":
        raise NotImplementedError("Mosaic simulation not currently implemented!")
    if format == "tsv":
        logging.info(
            "Writing output in TSV format (note: not all intermediate data will be kept) ..."
        )
        if gzip:
            logging.info(f"Writing output to {out}.tsv.gz")
            out_fp = f"{out}.tsv.gz"
            with gz.open(out_fp, "wt") as outfile:
                outfile.write(
                    "chrom\tpos\tref\talt\taf\tploidy\tmat_hap0\tmat_hap1\tpat_hap0\tpat_hap1\tbaf\n"
                )
                if "aploid" in results:
                    for i in range(m):
                        outfile.write(
                            f"{chrom}\t{results['pos'][i]}\tA\tC\t{results['af']}\t{results['aploid']}\t{results['mat_haps_prime'][0, i]}\t{results['mat_haps_prime'][1, i]}\t{results['pat_haps_prime'][0, i]}\t{results['pat_haps_prime'][1, i]}\t{results['baf'][i]}\n"
                        )
                else:
                    for i in range(m):
                        outfile.write(
                            f"{chrom}\t{results['pos'][i]}\tA\tC\t{results['af']}\t{results['ploidies'][i]}\t{results['mat_haps_prime'][0, i]}\t{results['mat_haps_prime'][1, i]}\t{results['pat_haps_prime'][0, i]}\t{results['pat_haps_prime'][1, i]}\t{results['baf'][i]}\n"
                        )
        else:
            logging.info(f"Writing output to {out}.tsv")
            out_fp = f"{out}.tsv"
            with open(out_fp, "w+") as outfile:
                outfile.write(
                    "chrom\tpos\tref\talt\taf\tploidy\tmat_hap0\tmat_hap1\tpat_hap0\tpat_hap1\tbaf\n"
                )
                if "aploid" in results:
                    for i in range(m):
                        outfile.write(
                            f"{chrom}\t{results['pos'][i]}\tA\tC\t{results['af']}\t{results['aploid']}\t{results['mat_haps_prime'][0, i]}\t{results['mat_haps_prime'][1, i]}\t{results['pat_haps_prime'][0, i]}\t{results['pat_haps_prime'][1, i]}\t{results['baf'][i]}\n"
                        )
                else:
                    for i in range(m):
                        outfile.write(
                            f"{chrom}\t{results['pos'][i]}\tA\tC\t{results['af']}\t{results['ploidies'][i]}\t{results['mat_haps_prime'][0, i]}\t{results['mat_haps_prime'][1, i]}\t{results['pat_haps_prime'][0, i]}\t{results['pat_haps_prime'][1, i]}\t{results['baf'][i]}\n"
                        )

    else:
        # Append a chromosome name to the input
        results["chrom"] = np.repeat(f"{chrom}", m, dtype=str)
        results["ref"] = np.repeat("A", m, dtype=str)
        results["alt"] = np.repeat("C", m, dtype=str)
        logging.info("Writing output in NPZ format ...")
        out_fp = f"{out}.npz"
        logging.info(f"Writing output to {out}.npz ...")
        np.savez(out_fp, **results)

    logging.info("Finished data simulation!")
