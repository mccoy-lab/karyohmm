import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm import PGTSim, PGTSimMosaic

pgt_sim = PGTSim()
pgt_sim_mosaic = PGTSimMosaic()


@given(
    l=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=20000),
)
def test_pgt_sim(l, m):
		"""Test for PGT simulations."""
		data = pgt_sim.full_ploidy_sim(m=m, length=l)
		assert data['m'] == m
		assert data['length'] == l
		assert np.max(data['pos']) <= l		
