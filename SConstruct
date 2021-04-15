import os
import subprocess
import sysconfig
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '#rednose',
  '#rednose/examples/generated',
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
  LIBPATH=["#rednose/examples/generated"],
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

rednose_config = {
  'generated_folder': '#examples/generated',
  'to_build': {
    'live': ('#examples/live_kf.py', True, []),
    'kinematic': ('#examples/kinematic_kf.py', True, []),
    'compare': ('#examples/test_compare.py', True, []),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', False, []),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', False, []),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', False, []),
  },
}

Export('env', 'envCython', 'arch', 'rednose_config')
SConscript(['#rednose/SConscript'])
