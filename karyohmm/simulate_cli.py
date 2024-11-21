"""CLI for simulating synthetic aneuploidy data in karyoHMM."""
import logging
import sys

import click
import numpy as np
import pandas as pd

from karyohmm import PGTSim, PGTSimMosaic

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
    type=float,
    show_default=True,
    help="VCF as input for parental haplotype data.",
)
@click.option(
    "--vcf",
    "-v",
    required=False,
    default=None,
    type=float,
    show_default=True,
    help="VCF as input for parental haplotype data.",
)
@click.option(
    "--maternal_id",
    required=False,
    default=None,
    type=float,
    show_default=True,
    help="IID of maternal individual in VCF",
)
@click.option(
    "--paternal_id",
    required=False,
    default=None,
    type=float,
    show_default=True,
    help="IID of paternal individual in VCF.",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    required=False,
    type=bool,
    default=True,
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
    mode="Whole-Chromosome",
    recomb_rate=1e-8,
    aneuploidy_rate=1e-2,
    duo_maternal=None,
    gzip=True,
    out="karyohmm",
):
    """Karyohmm-Simulator CLI."""
    logging.info("Starting simulation ...")
    logging.info("Finished data simulation!")
