"""Test suite for karyoHMM DuoHMM."""
import numpy as np
import pytest
from karyohmm_utils import create_index_arrays, transition_kernel
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import mode

from karyohmm import DuoHMMRef, PGTSim

# --- Generating test data for applications in the DuoHMM Reference Panel setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.7, std_dev=0.1, seed=42)
data_disomy["ref_panel"] = pgt_sim.sim_haplotype_ref_panel(
    haps=data_disomy["mat_haps"], pos=data_disomy["pos"], panel_size=10, seed=1
)


def test_data_integrity(data=data_disomy):
    """Test for some basic data sanity checks ..."""
    for x in ["baf", "mat_haps", "pat_haps"]:
        assert x in data
    baf = data["baf"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    assert np.all((baf <= 1.0) & (baf >= 0.0))
    assert baf.size == mat_haps.shape[1]
    assert baf.size == pos.size
    assert mat_haps.shape == pat_haps.shape
    assert np.all((mat_haps == 0) | (mat_haps == 1))
    assert np.all((pat_haps == 0) | (pat_haps == 1))


# --- Testing the metadata implementations --- #
@pytest.mark.parametrize("data", [data_disomy])
def test_forward_algorithm(data):
    """Test the implementation of the forward algorithm."""
    hmm = DuoHMMRef()
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf"],
        pos=data["pos"],
        haps=data["mat_haps"],
        ref_panel=data["ref_panel"],
        pi0=0.7,
        std_dev=0.1,
    )
