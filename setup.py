import os
import subprocess
import platform
import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools import Extension

rednose_dir = os.path.dirname(os.path.abspath(__file__))

cpp_args = ["-std=c++17", "-fPIC", "-O2"]

# Find eigen include path
eigen_include = []
if platform.system() == "Darwin":
    try:
        brew_prefix = subprocess.check_output(['brew', '--prefix'], encoding='utf8').strip()
        eigen_include = [os.path.join(brew_prefix, 'include')]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
else:
    # Standard Linux locations
    for p in ['/usr/include', '/usr/local/include']:
        if os.path.isdir(os.path.join(p, 'eigen3')):
            eigen_include = [p]
            break

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
        include_dirs=[
            rednose_dir,                                    # for "rednose/helpers/..." includes
            os.path.join(rednose_dir, "rednose"),           # for "logger/logger.h" includes
            os.path.join(rednose_dir, "rednose", "helpers"), # for local includes like "ekf_sym.h"
            np.get_include(),
        ] + eigen_include,
        libraries=["dl"],
    ),
]

setup(ext_modules=cythonize(extensions, language_level="3"))
