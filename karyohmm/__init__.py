"""Karyohmm is an HMM-based model for aneuploidy detection in PGT-A datasets.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Models for sequencing-based PGT-A are planned for future versions.

Modules exported are:

* MetaHMM: module for whole chromosome aneuploidy determination via HMMs.
* QuadHMM: module leveraging multi-sibling HMM for evaluating crossover recombination estimation.
* RecombEsst: module for heuristic estimation of crossover recombination based on Coop et al 2007.
* PhaseCorrect: module implementing Mendelian phase correction for parental haplotypes.
* MosaicEst: module for estimating mosaic cell fraction from shifts in BAF
* PGTSim: module to generate synthetic PGT data for full aneuploidies
* PGTSimMosaic: module to generate synthetic PGT data for a mosaic biopsy
"""

__version__ = "0.3.2a"

from .karyohmm import MetaHMM, MosaicEst, PhaseCorrect, QuadHMM, RecombEst
from .simulator import PGTSim, PGTSimMosaic
