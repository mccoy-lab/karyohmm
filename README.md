# `karyohmm`

Karyohmm is a method to estimate copy number of chromosomes from large-scale genotyping intensity data (specifically for PGT data) when conditioning on parental haplotypes. Specifically, we estimate the posterior probability of a specific karyotypic state (e.g. disomy vs maternal trisomy) in the `MetaHMM` framework.

## Installation

Currently `karyohmm` is installable via a local `pip` install. Simply execute the following:

```
git clone git@github.com:mccoy-lab/karyohmm.git
cd karyohmm/
pip install .
```

While the majority of the interface uses `python`, many of the internal helper functions built using `Cython` (see the `karyohmm_utils.pyx` file)

## `MetaHMM`

The `MetaHMM` mode infers embryo copy-number using primarily B-allele Frequency (BAF) data. However, we highly suggest orienting this to the `REF/ALT` allelic state of the parental haplotypes. We suggest applying the following steps on each chromosome for maximum performance:

1. Estimate parameters ($\sigma, \pi_0$) via MLE using numerical optimization of the forward-algorithm.

2. Estimate the posterior probabilities of being in specific karyotype states using the Forward-Backward algorithm (using the inferred parameters from step 1).

### States in `MetaHMM`

The states in the `MetaHMM` model correspond to specific karyotypes for chromosomes or sections of a chromosome:

* `0` - nullisomy
* `1m` - only a single maternal chromosome is copied from (paternal chromosome loss / paternal monosomy)
* `1p` - only a single paternal chromosome is copied from (maternal chromosome loss/ maternal monosomy)
* `2` - disomy
* `3m` - extra maternal chromosome (maternal trisomy)
* `3p` - extra paternal chromosome (paternal trisomy)

## DuoHMM

The `DuoHMM` model similarly attempts to quantify the aneuploidy status of an embryo, but with the availability of only a **single**  parent. This is primarily accomplished by marginalizing over the full set of possible genotypes for the unobserved parent (with a prior based on allele frequency).

## CLI

The installation of `karyohmm` also includes  `karyohmm-infer` and `karyohmm-simulate` command-line interfaces.

The `karyohmm-infer` program includes all of implementation for inferring aneuploidy status that can be run directly from the command line. You can even specify the whether you are in `MetaHMM` or `DuoHMM` mode to indicate parental availability.

```
karyohmm-infer -i data/test_disomy_embryo.tsv -o data/out_disomy
karyohmm-infer -i data/test_mat_trisomy_embryo.tsv -o data/out_mat_trisomy
karyohmm-infer -i data/test_combined_embryo.tsv -o data/out_combined
```

For a full command-line set of options, run `karyohmm-infer --help`.

To simulate different aneuploidy types, you can use the `karyohmm-simulate` program. Currently the modes that are supported are to simulate whole-chromosome or segmental aneuploidies. You can also use a pre-existing VCF file to sample parental haplotypes from for more realistic haplotype structure (and variant density). Currently, the simulation model only supports simulation of log-R ratio and B-allele frequency data to mimic Illumina arrays (we anticipate supporting read-based approaches like exome-capture quite soon).

For a full set of command-line options for simulation, run `karyohmm-simulate --help`.

## Contact

Please submit an issue or contact @aabiddanda
