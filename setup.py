import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools import Extension

rednose_dir = os.path.dirname(os.path.abspath(__file__))

cpp_args = ["-std=c++17", "-fPIC", "-O2"]

extensions = [
    Extension(
        "rednose.helpers.ekf_sym_pyx",
        sources=[
            "rednose/helpers/ekf_sym_pyx.pyx",
            "rednose/helpers/ekf_load.cc",
            "rednose/helpers/ekf_sym.cc",
        ],
        language="c++",
        extra_compile_args=cpp_args,
        include_dirs=[rednose_dir, np.get_include()],
        libraries=["dl"],
    ),
]

setup(ext_modules=cythonize(extensions, language_level="3"))
