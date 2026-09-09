import os
import platform
import subprocess
import sys
import sysconfig
import numpy as np
import eigen

WINDOWS = platform.system() == "Windows"
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

common = ''

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '#rednose',
  '#rednose/examples/generated',
  '/usr/lib/include',
  python_path,
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
  tools=["mingw" if WINDOWS else "default", "cython", "rednose_filter"],  # the default tool picks MSVC on Windows
)
if WINDOWS:
  env["CC"], env["CXX"] = "clang", "clang++"  # the mingw tool assumes gcc
  env["SHLIBPREFIX"] = "lib"  # the mingw tool drops the prefix ekf_load expects
  env.Append(LINKFLAGS=["-static"])  # libc++ into the DLLs so they load outside the MSYS2 shell

# Cython build enviroment
envCython = env.Clone()
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-cpp", "-Wno-shadow", "-Wno-deprecated-declarations"]

envCython["LIBS"] = []
if platform.system() == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif WINDOWS:
  envCython["LINKFLAGS"] = ["-shared", "-static"]
  envCython["LIBPATH"] += [os.path.join(sys.base_prefix, "libs")]
  envCython["LIBS"] = [f"python{sys.version_info.major}{sys.version_info.minor}"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(python_path)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('env', 'envCython', 'common')

SConscript(['#rednose/SConscript'])
SConscript(['#examples/SConscript'])
