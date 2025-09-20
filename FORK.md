# Why we forked it

## Motivation

1. Don't use scons as the build tool
2. Support python3.8
3. Better distribute package

| change                            | description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `rednose/helpers/kalmanfilter.py` | Update typehint to use python3.8 compatible syntax   |
| `setup.py`, `pyproject.toml`      | Use setuptools to build and package instead of scons |

## Packaging the Library

```bash
# to build a wheel
python -m build --wheel
```

### What is packaged

```
.
├── __init__.py
├── helpers
│   ├── __init__.py
│   ├── chi2_lookup_table.npy
│   ├── chi2_lookup.py
│   ├── common_ekf.cc
│   ├── common_ekf.h
│   ├── ekf_load.cc
│   ├── ekf_load.h
│   ├── ekf_sym_pyx.cpp
│   ├── ekf_sym_pyx.cpython-38-darwin.so
│   ├── ekf_sym_pyx.pyx
│   ├── ekf_sym.cc
│   ├── ekf_sym.h
│   ├── ekf_sym.py
│   ├── ekf.h
│   ├── feature_handler.py
│   ├── kalmanfilter.py
│   ├── lst_sq_computer.py
│   └── sympy_helpers.py
├── logger
│   └── logger.h
└── templates
    ├── __init__.py
    ├── compute_pos.c
    ├── ekf_c.c
    └── feature_handler.c
```

- instead making users recompile the library `ekf_sym.cc` and `ekf_load.cc`, I think we can just ship a shared lib.
  this way we only need to ship the headers, the pyx ext, and the python source

## Using the library

The wheel packages c++ source, the compiled cython extension, and python library to generate code for new filters.
To use the library, you must:

1. define the filter in python
2. run the code generation step
3. compile the generated code

```bash
# example on mac
export REDNOSE_ROOT=.venv/lib/python3.8/site-packages
clang++ -std=c++17 \
  -I $REDNOSE_ROOT/rednose/helpers \
  -I $REDNOSE_ROOT/rednose \
  -I $REDNOSE_ROOT \
  -I /opt/homebrew/include \
  generated/kinematic.cpp \
  $REDNOSE_ROOT/rednose/helpers/ekf_load.cc \
  $REDNOSE_ROOT/rednose/helpers/ekf_sym.cc \
  -dynamiclib -o  generated/libkinematic.dylib

# linux
```

export REDNOSE_ROOT=.venv/lib/python3.8/site-packages
g++ -std=c++17 -fPIC \
 -I $REDNOSE_ROOT/rednose/helpers \
 -I $REDNOSE_ROOT/rednose \
 -I $REDNOSE_ROOT \
 generated/kinematic.cpp \
 $REDNOSE_ROOT/rednose/helpers/ekf_load.cc \
 $REDNOSE_ROOT/rednose/helpers/ekf_sym.cc \
 -shared -o generated/libkinematic.so

```

This library ships a compiled cythone extension that will dynamically load an arbitrary filter at runtime. For a new
filter, you are building the shared library that will be loaded.

### Depedencies

1. eigen
```
