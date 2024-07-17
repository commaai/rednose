from setuptools import Command, setup, Distribution
from setuptools.command.build import build
import subprocess

class BinaryDistribution(Distribution):
  def has_ext_modules(self):
    return True


class SconsBuild(Command):
  def initialize_options(self) -> None:
    pass

  def finalize_options(self) -> None:
    pass

  def run(self) -> None:
    subprocess.run(["scons -j$(nproc)"], shell=True).check_returncode()


class CustomBuild(build):
  sub_commands = [('scons_build', None)] + build.sub_commands


setup(
    packages = ["rednose", "rednose.examples", "rednose.helpers"],
    package_data={'': ['**/*.cc', '**/*.c', '**/*.h', '**/*.pxd', '**/*.pyx', '**/*.py', '**/*.so', '**/*.npy']},
    include_package_data=True,
    cmdclass={'build': CustomBuild, 'scons_build': SconsBuild},
    distclass=BinaryDistribution,
    )
