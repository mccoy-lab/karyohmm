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

The `MetaHMM` mode implements a characterization of embryo copy-number using primarily B-allele Frequency (BAF) data. However, we highly suggest orienting this to the `REF/ALT` allelic state of the parental haplotypes. We suggest applying the following steps on each chromosome for maximum performance:

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

## CLI

The installation of `karyohmm` also includes a `karyohmm-cli` implementation that can be run directly from the command-line. This is work in progress, but currently implements the majority of the `MetaHMM` inference steps for a single data file. To see this on a simulated example of three embryos:

```
karyohmm-cli -i data/test_disomy_embryo.tsv -o data/out_disomy
karyohmm-cli -i data/test_mat_trisomy_embryo.tsv -o data/out_mat_trisomy
karyohmm-cli -i data/test_combined_embryo.tsv -o data/out_combined
```

This will run the `MetaHMM` model for two simulated datasets and output a simple table of posterior probabilities of each potential karyotypic outcome. To see the full display of options when running the model, you can run `karyohmm-cli --help`.

## Contact

Please submit an issue or contact @aabiddanda
