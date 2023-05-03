"""Test suite for karyoHMM."""
import numpy as np
import pytest
from utils import full_ploidy_sim

from karyohmm import EuploidyHMM

# --- Generating test data for applications --- #
data_disomy = full_ploidy_sim(m=2000, seed=42)

# --- Testing the metadata implementations --- #
@pytest.mark.parametrize("data,logr", [(data_disomy, False)])
def test_forward_algorithm(data, logr):
    """Test the implementation of the forward algorithm."""
    hmm = EuploidyHMM()
    _, _, _, _, loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert loglik < 0


@pytest.mark.parametrize("data,logr", [(data_disomy, False)])
def test_backward_algorithm(data, logr):
    """Test the implementation of the backward algorithm."""
    hmm = EuploidyHMM()
    _, _, _, _, loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert loglik < 0


def test_fwd_bwd_algorithm():
    """Test the properties of the output from the fwd-bwd algorithm."""
    pass
