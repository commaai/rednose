import os
import logging

import numpy as np
import sympy as sp

from rednose.helpers import TEMPLATE_DIR
from rednose.helpers.sympy_helpers import sympy_into_c
from rednose.helpers.chi2_lookup import chi2_ppf


def gen_code(folder, name, f_sym, dt_sym, x_sym, obs_eqs, dim_x, dim_err, eskf_params=None, msckf_params=None,  # pylint: disable=dangerous-default-value
             maha_test_kinds=[], global_vars=None):
  # optional state transition matrix, H modifier
  # and err_function if an error-state kalman filter (ESKF)
  # is desired. Best described in "Quaternion kinematics
  # for the error-state Kalman filter" by Joan Sola

  if eskf_params:
    err_eqs = eskf_params[0]
    inv_err_eqs = eskf_params[1]
    H_mod_sym = eskf_params[2]
    f_err_sym = eskf_params[3]
    x_err_sym = eskf_params[4]
  else:
    nom_x = sp.MatrixSymbol('nom_x', dim_x, 1)
    true_x = sp.MatrixSymbol('true_x', dim_x, 1)
    delta_x = sp.MatrixSymbol('delta_x', dim_x, 1)
    err_function_sym = sp.Matrix(nom_x + delta_x)
    inv_err_function_sym = sp.Matrix(true_x - nom_x)
    err_eqs = [err_function_sym, nom_x, delta_x]
    inv_err_eqs = [inv_err_function_sym, nom_x, true_x]

    H_mod_sym = sp.Matrix(np.eye(dim_x))
    f_err_sym = f_sym
    x_err_sym = x_sym

  # This configures the multi-state augmentation
  # needed for EKF-SLAM with MSCKF (Mourikis et al 2007)
  if msckf_params:
    msckf = True
    dim_main = msckf_params[0]      # size of the main state
    dim_augment = msckf_params[1]   # size of one augment state chunk
    dim_main_err = msckf_params[2]
    dim_augment_err = msckf_params[3]
    N = msckf_params[4]
    feature_track_kinds = msckf_params[5]
    assert dim_main + dim_augment * N == dim_x
    assert dim_main_err + dim_augment_err * N == dim_err
  else:
    msckf = False
    dim_main = dim_x
    dim_augment = 0
    dim_main_err = dim_err
    dim_augment_err = 0
    N = 0

  # linearize with jacobians
  F_sym = f_err_sym.jacobian(x_err_sym)

  if eskf_params:
    for sym in x_err_sym:
      F_sym = F_sym.subs(sym, 0)

  assert dt_sym in F_sym.free_symbols

  for i in range(len(obs_eqs)):
    obs_eqs[i].append(obs_eqs[i][0].jacobian(x_sym))
    if msckf and obs_eqs[i][1] in feature_track_kinds:
      obs_eqs[i].append(obs_eqs[i][0].jacobian(obs_eqs[i][2]))
    else:
      obs_eqs[i].append(None)

  # collect sympy functions
  sympy_functions = []

  # error functions
  sympy_functions.append(('err_fun', err_eqs[0], [err_eqs[1], err_eqs[2]]))
  sympy_functions.append(('inv_err_fun', inv_err_eqs[0], [inv_err_eqs[1], inv_err_eqs[2]]))

  # H modifier for ESKF updates
  sympy_functions.append(('H_mod_fun', H_mod_sym, [x_sym]))

  # state propagation function
  sympy_functions.append(('f_fun', f_sym, [x_sym, dt_sym]))
  sympy_functions.append(('F_fun', F_sym, [x_sym, dt_sym]))

  # observation functions
  for h_sym, kind, ea_sym, H_sym, He_sym in obs_eqs:
    sympy_functions.append(('h_%d' % kind, h_sym, [x_sym, ea_sym]))
    sympy_functions.append(('H_%d' % kind, H_sym, [x_sym, ea_sym]))
    if msckf and kind in feature_track_kinds:
      sympy_functions.append(('He_%d' % kind, He_sym, [x_sym, ea_sym]))

  # Generate and wrap all th c code
  sympy_header, code = sympy_into_c(sympy_functions, global_vars)

  header = "#pragma once\n"
  header += "#include \"rednose/helpers/common_ekf.h\"\n"
  header += "extern \"C\" {\n"

  pre_code = f"#include \"{name}.h\"\n"
  pre_code += f"\nnamespace {{\n"
  pre_code += "#define DIM %d\n" % dim_x
  pre_code += "#define EDIM %d\n" % dim_err
  pre_code += "#define MEDIM %d\n" % dim_main_err
  pre_code += "typedef void (*Hfun)(double *, double *, double *);\n"

  if global_vars is not None:
    for var in global_vars:
      pre_code += f"\ndouble {var.name};\n"
      pre_code += f"\nvoid set_{var.name}(double x){{ {var.name} = x;}}\n"

  post_code = "\n}\n" # namespace
  post_code += "extern \"C\" {\n\n"

  for h_sym, kind, ea_sym, H_sym, He_sym in obs_eqs:
    if msckf and kind in feature_track_kinds:
      He_str = 'He_%d' % kind
      # ea_dim = ea_sym.shape[0]
    else:
      He_str = 'NULL'
      # ea_dim = 1 # not really dim of ea but makes c function work
    maha_thresh = chi2_ppf(0.95, int(h_sym.shape[0]))  # mahalanobis distance for outlier detection
    maha_test = kind in maha_test_kinds

    pre_code += f"const static double MAHA_THRESH_{kind} = {maha_thresh};\n"

    header += f"void {name}_update_{kind}(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);\n"
    post_code += f"void {name}_update_{kind}(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {{\n"
    post_code += f"  update<{h_sym.shape[0]}, 3, {int(maha_test)}>(in_x, in_P, h_{kind}, H_{kind}, {He_str}, in_z, in_R, in_ea, MAHA_THRESH_{kind});\n"
    post_code += f"}}\n"

  # For ffi loading of specific functions
  for line in sympy_header.split("\n"):
    if line.startswith("void "):  # sympy functions
      func_call = line[5: line.index(')') + 1]
      header += f"void {name}_{func_call};\n"
      post_code += f"void {name}_{func_call} {{\n"
      post_code += f"  {func_call.replace('double *', '').replace('double', '')};\n"
      post_code += f"}}\n"
  header += f"void {name}_predict(double *in_x, double *in_P, double *in_Q, double dt);\n"
  post_code += f"void {name}_predict(double *in_x, double *in_P, double *in_Q, double dt) {{\n"
  post_code += f"  predict(in_x, in_P, in_Q, dt);\n"
  post_code += f"}}\n"
  if global_vars is not None:
    for var in global_vars:
      header += f"void {name}_set_{var.name}(double x);\n"
      post_code += f"void {name}_set_{var.name}(double x) {{\n"
      post_code += f"  set_{var.name}(x);\n"
      post_code += f"}}\n"

  post_code += f"}}\n\n" # extern c

  funcs = ['f_fun', 'F_fun', 'err_fun', 'inv_err_fun', 'H_mod_fun', 'predict']
  func_lists = {
    'h': [kind for _, kind, _, _, _ in obs_eqs],
    'H': [kind for _, kind, _, _, _ in obs_eqs],
    'update': [kind for _, kind, _, _, _ in obs_eqs],
    'He': [kind for _, kind, _, _, _ in obs_eqs if msckf and kind in feature_track_kinds],
    'set': [var.name for var in global_vars] if global_vars is not None else [],
  }

  # For dynamic loading of specific functions
  post_code += f"const EKF {name} = {{\n"
  post_code += f"  .name = \"{name}\",\n"
  post_code += f"  .kinds = {{ {', '.join([str(kind) for _, kind, _, _, _ in obs_eqs])} }},\n"
  post_code += f"  .feature_kinds = {{ {', '.join([str(kind) for _, kind, _, _, _ in obs_eqs if msckf and kind in feature_track_kinds])} }},\n"
  for func in funcs:
    post_code += f"  .{func} = {name}_{func},\n"
  for group, kinds in func_lists.items():
    post_code += f"  .{group}s = {{\n"
    for kind in kinds:
      str_kind = f"\"{kind}\"" if type(kind) == str else kind
      post_code += f"    {{ {str_kind}, {name}_{group}_{kind} }},\n"
    post_code += f"  }},\n"
  post_code += f"}};\n\n"
  post_code += f"ekf_init({name});\n"

  # merge code blocks
  header += "}"
  code = "\n".join([pre_code, code, open(os.path.join(TEMPLATE_DIR, "ekf_c.c")).read(), post_code])

  # write to file
  if not os.path.exists(folder):
    os.mkdir(folder)

  open(os.path.join(folder, f"{name}.h"), 'w').write(header)  # header is used for ffi import
  open(os.path.join(folder, f"{name}.cpp"), 'w').write(code)
