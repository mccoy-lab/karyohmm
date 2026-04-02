"""Test suite for karyoHMM DuoHMM."""

import numpy as np
import pytest
from karyohmm import MccEst, PGTSim

# --- Generating test data for applications in the DuoHMM setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.01, std_dev=0.1, seed=42)



def test_init():
    """Test the initialization of the estimator."""
    x = MccEst()