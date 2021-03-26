import os
import subprocess
import sysconfig
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '#rednose',
  '/home/batman/openpilot/selfdrive/locationd/models/generated',
  '/usr/lib/include',
  python_path,
  np.get_include(),
]

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
  CXXFLAGS="-std=c++1z",
  CPPPATH=cpppath,
  tools=["default", "cython"],
)

# Cython build enviroment
envCython = env.Clone()
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-deprecated-declarations"]

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(python_path)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('envCython')
SConscript(['rednose/helpers/SConscript'])

Export('env', 'arch')
SConscript(['SConscript'])