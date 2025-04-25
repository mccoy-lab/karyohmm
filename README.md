# `karyohmm`

Karyohmm is a method to estimate copy number of chromosomes from large-scale genotyping intensity data (specifically for PGT data) when conditioning on parental haplotypes. Specifically, we estimate the posterior probability of a specific karyotypic state (e.g. disomy vs maternal trisomy) in the `MetaHMM` framework.

## Installation

`karyohmm` is installable via a local `pip` install. Simply execute the following:

```
git clone git@github.com:mccoy-lab/karyohmm.git
cd karyohmm/
pip install .
```

To install the package without cloning the entire repository, you can run (though this will exclude some of the test data): 

```
pip install git+https://github.com/mccoy-lab/karyohmm
```

which should handle all of the key dependencies. Installation should be completed within two minutes.

While the majority of the interface uses `python`, many of the internal helper functions built using `Cython` (see the `karyohmm_utils.pyx` file)

The software was tested on Mac OSX and Linux with the following dependencies and versions (using `python 3.10`): 
```
cyvcf2                               0.31.1
numpy                                2.1.3
pandas                               2.2.2
scipy                                1.15.0
```

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

To test the CLI (assuming the full repository was cloned), you can run inference on the following simulated datasets:
```
karyohmm-infer -i data/test_disomy_embryo.tsv -o data/out_disomy
karyohmm-infer -i data/test_mat_trisomy_embryo.tsv -o data/out_mat_trisomy
karyohmm-infer -i data/test_combined_embryo.tsv -o data/out_combined
```

For a full command-line set of options, run `karyohmm-infer --help`:

```
Usage: karyohmm-infer [OPTIONS]

  Karyohmm-Inference CLI.

Options:
  -i, --input PATH                Input data file for PGT-A array intensity
                                  data.  [required]
  --viterbi                       Viterbi algorithm for tracing ploidy states.
  --mode [Meta|Duo]               [default: Meta; required]
  --algo [Nelder-Mead|L-BFGS-B|Powell]
                                  Optimization method for parameter inference.
                                  [default: Powell]
  --thin INTEGER                  SNP thinning to improve optimization speed
                                  for parameter inference.  [default: 1]
  -r, --recomb_rate FLOAT         Recombination rate between SNPs.  [default:
                                  1e-08]
  -a, --aneuploidy_rate FLOAT     Probability of shifting between aneuploidy
                                  states between SNPs.  [default: 0.01]
  -dm, --duo_maternal BOOLEAN     Indicator of duo being mother-child duo.
  -g, --gzip                      Gzip output files.
  -o, --out TEXT                  Output file prefix.  [required]
  --help                          Show this message and exit.
```


To simulate different aneuploidy types, you can use the `karyohmm-simulate` program. Currently the modes that are supported are to simulate whole-chromosome or segmental aneuploidies. You can also use a pre-existing VCF file to sample parental haplotypes from for more realistic haplotype structure (and variant density). Currently, the simulation model only supports simulation of log-R ratio and B-allele frequency data to mimic Illumina arrays (we anticipate supporting read-based approaches like exome-capture quite soon).

For a full set of command-line options for simulation, run `karyohmm-simulate --help`:

```
  Karyohmm-Simulator CLI.

Options:
  --mode [Whole-Chromosome|Segmental|Mosaic]
                                  [default: Whole-Chromosome; required]
  -c, --chrom TEXT                Chromosome indicator.  [default: chr1]
  -a, --afs PATH                  Allele frequency file for variants (to mimic
                                  ascertainment-bias).
  -r, --recomb_rate FLOAT         Recombination rate between SNPs.  [default:
                                  1e-08]
  -v, --vcf PATH                  VCF as input for parental haplotype data.
  --maternal_id TEXT              IID of maternal individual in VCF
  --paternal_id TEXT              IID of paternal individual in VCF.
  -l, --length FLOAT              Length of segment to simulate.  [default:
                                  50000000.0]
  -p, --ploidy [0|1|2|3]          Degree of aneuploidy to be simulated.
                                  [default: 2; required]
  -m, --m INTEGER                 Number of variants to simulate on
                                  chromosome.  [default: 5000]
  --std_dev FLOAT                 Standard deviation of BAF-distribution.
                                  [default: 0.2; required]
  --pi0 FLOAT                     Point-mass for emission distribution of BAF.
                                  [default: 0.5; required]
  --mat_skew FLOAT                Probability of being a maternal-origin
                                  aneuploidy.  [default: 0.5; required]
  --mean_size INTEGER             Mean size of a segmental aneuploidy on the
                                  chromosome.  [default: 100]
  -se, --switch_err_rate FLOAT    Switch error rate in parental haplotypes.
                                  [default: 0.01]
  --seed INTEGER                  Random seed for simulation.  [default: 42;
                                  required]
  --threads INTEGER               VCF reading threads.  [default: 1]
  -g, --gzip                      Gzip output files.
  -o, --out TEXT                  Output file prefix.  [required]
  -fmt, --format [tsv|npz]        Output file format.  [required]
  --help                          Show this message and exit.
```

## Contact

Please submit an issue or contact @aabiddanda
