import os
import platform
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy


extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []
include_dirs = [".", "rednose", numpy.get_include()]

if platform.system() == "Darwin":
    extra_compile_args += ["-stdlib=libc++"]
    extra_link_args += ["-stdlib=libc++"]
    include_dirs += ["/opt/homebrew/include"]

# debug
if os.environ.get("DEBUG"):
    extra_compile_args += ["-O0", "-g3"]
    extra_link_args += ["-g"]


extensions = [
    Extension(
        name="rednose.helpers.ekf_sym_pyx",
        sources=[
            "rednose/helpers/ekf_sym_pyx.pyx",  # Cython
            "rednose/helpers/ekf_sym.cc",  # your C++
            "rednose/helpers/ekf_load.cc",  # your C++
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


setup(
    name="rednose",
    version="0.0.1",
    url="https://github.com/commaai/rednose",
    author="comma.ai",
    author_email="harald@comma.ai",
    packages=find_packages(),
    platforms="any",
    license="MIT",
    include_package_data=True,
    package_data={
        "rednose.helpers": ["chi2_lookup_table.npy"],
        "rednose.templates": ["*.c"],
        "rednose": ["cmake/*.cmake"]
    },
    install_requires=["numpy", "cffi", "sympy"],
    extras_require={"dev": ["scipy"]},
    description="Kalman filter library",
    long_description="See https://github.com/commaai/rednose",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=False,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "binding": True,
            "language_level": 3,
        },
    ),
)
