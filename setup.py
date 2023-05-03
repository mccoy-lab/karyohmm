from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension("karyohmm_utils", ["karyohmm/karyohmm_utils.pyx"])
]

setup_args = dict(
    ext_modules = cythonize(extensions)
)
setup(**setup_args)