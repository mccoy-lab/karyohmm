"""Karyohmm is an HMM-based model for aneuploidy detection.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Modules exported are:

* MetaHMM: module for whole chromosome aneuploidy determination via HMMs
* QuadHMM: module leveraging multi-sibling design for evaluating crossover recombination estimation
* PhaseCorrect: module which implements Mendelian phase correction for parental haplotypes.

"""

__version__ = "0.1.7"

from .karyohmm import MetaHMM, PhaseCorrect, QuadHMM
