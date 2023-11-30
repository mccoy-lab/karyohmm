"""Setup module for building karyohmm."""


from distutils.core import Extension

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import setup

extensions = [Extension("karyohmm_utils", ["karyohmm/karyohmm_utils.pyx"])]

setup_args = dict(
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": 3, "profile": False}
    )
)
setup(**setup_args)
