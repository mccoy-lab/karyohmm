[![Tests](https://github.com/aabiddanda/karyohmm/actions/workflows/tests.yml/badge.svg)](https://github.com/aabiddanda/karyohmm/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/aabiddanda/karyohmm/graph/badge.svg)](https://codecov.io/gh/aabiddanda/karyohmm)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://aabiddanda.github.io/karyohmm/)
[![Nature](https://img.shields.io/badge/Nature_(2026)-10.1038%2Fs41586--025--09964--2-red)](https://doi.org/10.1038/s41586-025-09964-2)

# `karyohmm`

Karyohmm is a method to estimate copy number of chromosomes from large-scale genotyping intensity data (specifically for PGT data) when conditioning on parental haplotypes. Specifically, we estimate the posterior probability of a specific karyotypic state (e.g. disomy vs maternal trisomy) in the `MetaHMM` framework.

## Installation

`karyohmm` is installable via a local `pip` install from the repository (this is the current, most up-to-date version):

```
git clone git@github.com:aabiddanda/karyohmm.git
cd karyohmm/
pip install .
```

To install the package without cloning the entire repository, you can run (though this will exclude some of the test data):

```
pip install git+https://github.com/aabiddanda/karyohmm
```

which should handle all of the key dependencies. Installation should be completed within two minutes. Note that `aabiddanda/karyohmm` is the development version of the package and is more frequently updated than `mccoy-lab/karyohmm`.

While the majority of the interface uses `python`, many of the internal helper functions are built using `Cython` (see the `karyohmm_utils.pyx` file).

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

In specific cases we can also add in four uniparental disomy.

## `PocHMM`

The `PocHMM` (Products-of-Conception HMM) model quantifies the aneuploidy status of an embryo when only a **single** parent (mother or father) has been genotyped, as is common for products-of-conception samples following pregnancy loss. `PocHMM` extends `MetaHMM` and shares its karyotype state space (including the optional UPD states), but marginalizes over the unobserved parent's genotype at each site rather than conditioning on a second observed haplotype.

The unobserved parent's genotype is marginalized against a per-site allele frequency prior: if population allele frequencies are supplied (e.g. via an `af` column, or estimated from a haplotype reference panel using `infer_missing_af`), those are used; otherwise every site is treated as a 50/50 heterozygote for the unobserved parent.

We suggest applying the following steps on each chromosome:

1. (Optional) Estimate unobserved-parent allele frequencies from a haplotype reference panel via `infer_missing_af`, if population allele frequencies aren't otherwise available.
2. Estimate parameters ($\sigma, \pi_0$) via MLE using numerical optimization of the forward algorithm.
3. Estimate the posterior probabilities of karyotype states using the Forward-Backward algorithm, using the inferred parameters from step 2.

## CLI

The installation of `karyohmm` includes four command-line interfaces: `metahmm-infer`, `pochmm-infer`, `karyohmm-simulate`, and `karyohmm-mosaic`.

The `metahmm-infer` program runs the `MetaHMM` model for aneuploidy inference when both parental haplotypes are available.

The `pochmm-infer` program handles products-of-conception data. Use `--mode Duo` (default) when only one parent is genotyped, or `--mode Meta` when both parents are available.

To test the CLI (assuming the full repository was cloned), you can run inference on the following simulated datasets:
```
metahmm-infer -i data/test_disomy_embryo.tsv -o data/out_disomy
metahmm-infer -i data/test_mat_trisomy_embryo.tsv -o data/out_mat_trisomy
metahmm-infer -i data/test_combined_embryo.tsv -o data/out_combined
```

To simulate different aneuploidy types, you can use the `karyohmm-simulate` program. Currently the modes that are supported are to simulate whole-chromosome or segmental aneuploidies. You can also use a pre-existing VCF file to sample parental haplotypes from for more realistic haplotype structure (and variant density).

The `karyohmm-mosaic` program estimates the mosaic cell fraction for segmental or whole-chromosome aneuploidies using the `MosaicEst` model.

For full command-line options, use `--help` on any CLI entry point (e.g. `metahmm-infer --help`).

### Reference

For use in large-scale datasets, please refer to the following medRxiv preprint:

```
Common variation in meiosis genes shapes human recombination phenotypes and aneuploidy risk
Sara A. Carioscia, Arjun Biddanda, Margaret R. Starostik, Xiaona Tang, Eva R. Hoffmann, Zachary P. Demko, Rajiv C. McCoy
medRxiv 2025.04.02.25325097; doi: https://doi.org/10.1101/2025.04.02.25325097
```

There is also a set of two accompanying repositories that use `karyohmm` as an importable package in larger pipelines for aneuploidy discovery and recombination inference for the above preprint:

```
https://github.com/mccoy-lab/natera_aneuploidy
https://github.com/mccoy-lab/natera_recomb
```

## Documentation

Full API documentation is available at [https://aabiddanda.github.io/karyohmm/](https://aabiddanda.github.io/karyohmm/).

To build the documentation locally, install the docs dependencies and run Sphinx:

```
pip install ".[docs]"
cd docs/
make html
```

The rendered HTML will be written to `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to preview.

## Contact

Please submit an issue or contact @aabiddanda
