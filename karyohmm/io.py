import pandas as pd


class DataReader:
	"""Input / output class for reading in data for karyohmm."""	
	def __init__(self, mode="Meta", duo_maternal=None):
		assert mode in ['Meta', 'Duo', 'Recomb']
		karyo_dtypes = {
		    "chrom": str,
		    "pos": float,
		    "ref": str,
		    "alt": str,
		    "baf": float,
		    "mat_hap0": int,
		    "mat_hap1": int,
		    "pat_hap0": int,
		    "pat_hap1": int,
		}
		self.dtypes = karyo_dtypes
		self.mode = mode
		if (mode == 'Duo') and (duo_maternal is None):
			raise ValueError('Need to specify whether a mother-child or father-child duo!')
		if (duo_maternal is not None) and (mode == 'Duo'):
			self.duo_maternal = duo_maternal

	def read_data_np(self, input_fp):
	  """Read data from an .npy or npz file and reformat for karyohmm."""
	  data = np.load(input_fp, allow_pickle=True)
	  for x in ["chrom", "pos", "ref", "alt", "baf"]:
	      assert x in data
	  if self.mode != 'Duo':
	  	for x in ["mat_haps", "pat_haps"]
	  		assert x in data
		  df = pd.DataFrame(
		      {
		          "chrom": data["chrom"],
		          "pos": data["pos"],
		          "ref": data["ref"],
		          "alt": data["alt"],
		          "baf": data["baf"],
		          "mat_hap0": data["mat_haps"][0, :],
		          "mat_hap1": data["mat_haps"][1, :],
		          "pat_hap0": data["pat_haps"][0, :],
		          "pat_hap1": data["pat_haps"][1, :],
		      },
		      dtype=karyo_dtypes,
		  )
		  return df
		if self.mode == 'Duo':
			if self.duo_maternal:
				assert 'mat_haps' in data
				df = pd.DataFrame(
		      {
		          "chrom": data["chrom"],
		          "pos": data["pos"],
		          "ref": data["ref"],
		          "alt": data["alt"],
		          "baf": data["baf"],
		          "mat_hap0": data["mat_haps"][0, :],
		          "mat_hap1": data["mat_haps"][1, :],
		      },
		      dtype=karyo_dtypes,
		  	)
		  	return df
			else:
				assert 'pat_haps' in data
				df = pd.DataFrame(
		      {
		          "chrom": data["chrom"],
		          "pos": data["pos"],
		          "ref": data["ref"],
		          "alt": data["alt"],
		          "baf": data["baf"],
		          "pat_hap0": data["pat_haps"][0, :],
		          "pat_hap1": data["pat_haps"][1, :],
		      },
		      dtype=karyo_dtypes,
		  	)
				return df


	def read_data_df(self, input_fp):
	    """Read in data from a pre-existing text-based dataset."""
	    sep = ","
	    if ".tsv" in input_fp:
	        sep = "\t"
	    elif ".txt" in input_fp:
	        sep = " "
	    df = pd.read_csv(input_fp, dtype=karyo_dtypes, sep=sep)
	    for x in [
	        "chrom",
	        "pos",
	        "ref",
	        "alt",
	        "baf",
	    ]:
	        assert x in df.columns
	    if self.mode != 'Duo':
	    	for x in ['mat_hap0', 'mat_hap1', 'pat_hap0', 'pat_hap1']:
	    		assert x in df.columns
	    if self.mode == 'Duo':
	    	if self.duo_maternal:
	    		assert 'mat_hap0' in df.columns
	    		assert 'mat_hap1' in df.columns
	    	else:
	    		assert 'pat_hap0' in df.columns
	    		assert 'pat_hap1' in df.columns
	    return df


	def read_data(self, input_fp):
	    """Read in data in either pandas/numpy format."""
	    if (".npz" in input_fp) or (".npy" in input_fp):
	        df = self.read_data_np(input_fp)
	    else:
	        df = self.read_data_df(input_fp)
	    return df
