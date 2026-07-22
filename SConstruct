import os
import numpy as np
import eigen

common = ''

cpppath = [
  '#',
  '#rednose',
  '#rednose/examples/generated',
  '/usr/lib/include',
  np.get_include(),
  eigen.INCLUDE_DIR,
]

env = Environment(
  ENV=os.environ,
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Werror=implicit-function-declaration",
    "-Werror=incompatible-pointer-types",
    "-Werror=int-conversion",
    "-Werror=return-type",
    "-Werror=format-extra-args",
    "-Wshadow",
  ],
  LIBPATH=["#rednose/examples/generated"],
  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++1z",
  CPPPATH=cpppath,
  REDNOSE_ROOT=Dir("#").abspath,
  tools=["default", "rednose_filter"],
)

Export('env', 'common')

SConscript(['#rednose/SConscript'])
SConscript(['#examples/SConscript'])
