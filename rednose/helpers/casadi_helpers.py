import casadi as ca
import numpy as np

# TODO: remove code duplication between openpilot.common.orientation
def quat2rot(quats):
  quats = np.array(quats)
  input_shape = quats.shape
  quats = np.atleast_2d(quats)
  Rs = np.zeros((quats.shape[0], 3, 3))
  q0 = quats[:, 0]
  q1 = quats[:, 1]
  q2 = quats[:, 2]
  q3 = quats[:, 3]
  Rs[:, 0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
  Rs[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
  Rs[:, 0, 2] = 2 * (q0 * q2 + q1 * q3)
  Rs[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
  Rs[:, 1, 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
  Rs[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
  Rs[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
  Rs[:, 2, 1] = 2 * (q0 * q1 + q2 * q3)
  Rs[:, 2, 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

  if len(input_shape) < 2:
    return Rs[0]
  else:
    return Rs


def euler2quat(eulers):
  eulers = np.array(eulers)
  if len(eulers.shape) > 1:
    output_shape = (-1,4)
  else:
    output_shape = (4,)
  eulers = np.atleast_2d(eulers)
  gamma, theta, psi = eulers[:,0],  eulers[:,1],  eulers[:,2]

  q0 = np.cos(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
  q1 = np.sin(gamma / 2) * np.cos(theta / 2) * np.cos(psi / 2) - \
       np.cos(gamma / 2) * np.sin(theta / 2) * np.sin(psi / 2)
  q2 = np.cos(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2) + \
       np.sin(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2)
  q3 = np.cos(gamma / 2) * np.cos(theta / 2) * np.sin(psi / 2) - \
       np.sin(gamma / 2) * np.sin(theta / 2) * np.cos(psi / 2)

  quats = np.array([q0, q1, q2, q3]).T
  for i in range(len(quats)):
    if quats[i,0] < 0:  # pylint: disable=unsubscriptable-object
      quats[i] = -quats[i]  # pylint: disable=unsupported-assignment-operation,unsubscriptable-object
  return quats.reshape(output_shape)


def euler2rot(eulers):
  return quat2rot(euler2quat(eulers))


rotations_from_quats = quat2rot


def cross(x):
  ret = ca.SX.zeros(3, 3)
  ret[0, 1], ret[0, 2] = -x[2], x[1]
  ret[1, 0], ret[1, 2] = x[2], -x[0]
  ret[2, 0], ret[2, 1] = -x[1], x[0]
  return ret


def rot_to_euler(R):
  gamma = ca.atan2(R[2, 1], R[2, 2])
  theta = ca.asin(-R[2, 0])
  psi = ca.atan2(R[1, 0], R[0, 0])
  return ca.vertcat(gamma, theta, psi)


def rot_matrix(roll, pitch, yaw):
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rr = np.array([[1,0,0],[0, cr,-sr],[0, sr, cr]])
  rp = np.array([[cp,0,sp],[0, 1,0],[-sp, 0, cp]])
  ry = np.array([[cy,-sy,0],[sy, cy,0],[0, 0, 1]])
  return ry.dot(rp.dot(rr))


def euler_rotate(roll, pitch, yaw):
  # make symbolic rotation matrix from eulers
  matrix_roll = ca.blockcat([[1, 0, 0],
                           [0, ca.cos(roll), -ca.sin(roll)],
                           [0, ca.sin(roll), ca.cos(roll)]])
  matrix_pitch = ca.blockcat([[ca.cos(pitch), 0, ca.sin(pitch)],
                            [0, 1, 0],
                            [-ca.sin(pitch), 0, ca.cos(pitch)]])
  matrix_yaw = ca.blockcat([[ca.cos(yaw), -ca.sin(yaw), 0],
                          [ca.sin(yaw), ca.cos(yaw), 0],
                          [0, 0, 1]])
  return matrix_yaw @ matrix_pitch @ matrix_roll


def quat_rotate(q0, q1, q2, q3):
  # make symbolic rotation matrix from quat
  return ca.blockcat([[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
                    [2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)],
                    [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2]]).T


def quat_matrix_l(p):
  return ca.blockcat([[p[0], -p[1], -p[2], -p[3]],
                    [p[1],  p[0], -p[3],  p[2]],
                    [p[2],  p[3],  p[0], -p[1]],
                    [p[3], -p[2],  p[1],  p[0]]])


def quat_matrix_r(p):
  return ca.blockcat([[p[0], -p[1], -p[2], -p[3]],
                    [p[1],  p[0],  p[3], -p[2]],
                    [p[2], -p[3],  p[0],  p[1]],
                    [p[3],  p[2], -p[1],  p[0]]])


def casadi_into_c(routines, global_vars=None, scalar_args=()):
  globals_ = list(global_vars or [])
  generator = ca.CodeGenerator('rednose_casadi.c', {'with_header': False})
  headers, wrappers = [], []
  for name, expr, args in routines:
    inputs = [arg if arg is not None else ca.SX.sym(f'unused_{i}') for i, arg in enumerate(args)]
    # CasADi stores matrices column-major; the existing Eigen ABI is dense row-major.
    output = ca.densify(ca.vec(expr.T))
    function = ca.Function(f'casadi_{name}', inputs + globals_, [output], {'cse': True})
    generator.add(function)
    declarations, pointers, packing = [], [], []
    for i, arg in enumerate(args):
      scalar = arg is not None and arg.numel() == 1 and any(ca.is_equal(arg, candidate) for candidate in scalar_args)
      declarations.append(f"double {'' if scalar else '*'}arg{i}")
      if not scalar and arg is not None and arg.shape[1] > 1:
        rows, cols = map(int, arg.shape)
        elements = ', '.join(f'arg{i}[{row * cols + col}]' for col in range(cols) for row in range(rows))
        packing.append(f'  double packed{i}[] = {{{elements}}};\n')
        pointers.append(f'packed{i}')
      else:
        pointers.append(f'&arg{i}' if scalar else f'arg{i}')
    pointers += [f'&{var.name()}' for var in globals_]
    declaration = f"void {name}({', '.join(declarations + ['double *out'])})"
    headers.append(declaration + ';')
    niw, nw = function.sz_iw(), function.sz_w()
    wrappers.append(declaration + ' {\n' + ''.join(packing) +
                    f"  const double *arg[{max(1, function.sz_arg())}] = {{{', '.join(pointers)}}};\n" +
                    f"  double *res[{max(1, function.sz_res())}] = {{out}};\n" +
                    f"  casadi_int iw[{max(1, niw)}];\n  double w[{max(1, nw)}];\n" +
                    f"  casadi_{name}(arg, res, iw, w, 0);\n}}")
  return '\n'.join(headers), generator.dump() + '\n' + '\n'.join(wrappers)
