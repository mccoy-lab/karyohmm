"""Karyohmm is an HMM-based model for aneuploidy detection in PGT-A datasets.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Models for sequencing-based PGT-A are planned for future versions.

Modules exported are:

* MetaHMM: module for whole chromosome aneuploidy determination via HMMs.
* QuadHMM: module leveraging multi-sibling HMM for evaluating crossover recombination estimation.
* DuoHMM: module for aneuploidy determination in duo family array data.
* RecombEst: module for heuristic estimation of crossover recombination based on Coop et al 2007.
* PhaseCorrect: module implementing Mendelian phase correction for parental haplotypes.
* MosaicEst: module for estimating mosaic cell fraction from shifts in BAF.
* PGTSim: module to generate synthetic PGT data for full aneuploidies.
* PGTSimSegmental: module to generate synthetic PGT data for a segmental aneuploidy.
* PGTSimMosaic: module to generate synthetic PGT data for a mosaic biopsy.
* PGTSimVCF: module to generate synthetic PGT data based on parental data in a VCF.
"""

__version__ = "0.7.0a"

from .io import DataReader
from .karyohmm import (
    DuoHMM,
    DuoHMMRef,
    MetaHMM,
    MosaicEst,
    PhaseCorrect,
    QuadHMM,
    RecombEst,
)
from .simulator import PGTSim, PGTSimMosaic, PGTSimSegmental, PGTSimVCF
