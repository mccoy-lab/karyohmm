import numpy as np
import pytest

from karyohmm import PGTSim, MosaicEst



# --- Generating test data for applications --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=2000, length=10e6, seed=42)



def test_mosaic_est_init(data=data_disomy):
	"""Test the intialization of the mosaic estimation."""
	m_est = MosaicEst(mat_haps=data['mat_haps'], pat_haps=data["pat_haps"], bafs=data["baf_embryo"])
	assert m_est.het_bafs is None

def test_baf_hets(data=data_disomy):
	m_est = MosaicEst(mat_haps=data['mat_haps'], pat_haps=data["pat_haps"], bafs=data["baf_embryo"])
	m_est.baf_hets()
	assert m_est.het_bafs is not None