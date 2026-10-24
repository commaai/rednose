import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import casadi as ca
import numpy as np

from rednose.helpers.casadi_helpers import casadi_into_c


class TestCasadi(unittest.TestCase):
  def test_generated_abi(self):
    x = ca.SX.sym('x', 2, 2)
    dt, gain = ca.SX.sym('dt'), ca.SX.sym('gain')
    singleton = ca.SX.sym('singleton')
    expr = ca.blockcat([[x[0, 0] * gain, x[0, 1] + dt, 0], [x[1, 0], ca.sin(x[1, 1]), 1]])
    jac = ca.jacobian(ca.vertcat(x[0, 0]**2, x[1, 0] * dt), x[:, 0])
    routines = [('evaluate', expr, [x, dt, None]), ('derivative', jac, [x, dt]), ('singleton', singleton * gain, [singleton])]
    header, code = casadi_into_c(routines, [gain], scalar_args=[dt])
    self.assertIn('void evaluate(double *arg0, double arg1, double *arg2, double *out);', header)
    with tempfile.TemporaryDirectory() as folder:
      source = Path(folder) / 'test.cpp'
      source.write_text('#include <math.h>\nnamespace {\ndouble gain = 3;\n' + code +
                        '\n}\nextern "C" void run(double *x, double *y, double *j) { evaluate(x, 0.5, nullptr, y); derivative(x, 0.5, j); }\n' +
                        'extern "C" void one(double *x, double *y) { singleton(x, y); }\n')
      library = Path(folder) / 'test.so'
      subprocess.run(['g++', '-O2', '-shared', '-fPIC', str(source), '-o', str(library)], check=True, capture_output=True)
      lib = ctypes.CDLL(str(library))
      pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
      lib.run.argtypes = [pointer] * 3
      lib.run.restype = None
      values = np.array([[1., 2.], [3., 4.]])
      output, derivative = np.full((2, 3), np.nan), np.full((2, 2), np.nan)
      lib.run(values, output, derivative)
      np.testing.assert_allclose(output, [[3., 2.5, 0.], [3., np.sin(4.), 1.]])
      np.testing.assert_allclose(derivative, [[2., 0.], [0., 0.5]])
      lib.one.argtypes = [pointer] * 2
      lib.one.restype = None
      scalar_output = np.full(1, np.nan)
      lib.one(np.array([2.]), scalar_output)
      np.testing.assert_allclose(scalar_output, [6.])
