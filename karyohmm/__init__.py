"""Karyohmm is an HMM-based model for aneuploidy detection in PGT-A datasets.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Models for sequencing-based PGT-A are planned for future versions.

Modules exported are:

* MetaHMM: module for whole chromosome aneuploidy determination via HMMs.
* QuadHMM: module leveraging multi-sibling design for evaluating crossover recombination estimation.
* PhaseCorrect: module implementing Mendelian phase correction for parental haplotypes.
* PGTSim: module to generate synthetic PGT data

"""

__version__ = "0.1.8b"

from .karyohmm import MetaHMM, PhaseCorrect, QuadHMM
from .simulator import PGTSim
