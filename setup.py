"""Setup module for building karyohmm."""


from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension("karyohmm_utils", ["karyohmm/karyohmm_utils.pyx"])]

setup_args = dict(ext_modules=cythonize(extensions))
setup(**setup_args)
