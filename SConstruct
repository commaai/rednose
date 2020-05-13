import os
import subprocess

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

env = Environment(
  ENV=os.environ,
  CC='clang',
  CXX='clang++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Werror=implicit-function-declaration",
    "-Werror=incompatible-pointer-types",
    "-Werror=int-conversion",
    "-Werror=return-type",
    "-Werror=format-extra-args",
  ],
  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++14",
)


Export('env', 'arch')
SConscript(['SConscript'])
