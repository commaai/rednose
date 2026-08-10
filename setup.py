import os
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
  def has_ext_modules(self):
    return True


class BuildPyWithScons(build_py):
  def run(self):
    root = os.path.dirname(os.path.abspath(__file__))
    subprocess.check_call(["scons", f"-j{os.cpu_count() or 1}", "rednose"], cwd=root)
    super().run()


setup(cmdclass={"build_py": BuildPyWithScons}, distclass=BinaryDistribution)
