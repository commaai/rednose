import ctypes
import math
from pathlib import Path
import platform

import numpy as np


def _ptr(array):
  return array.ctypes.data


class EKF_sym_ctypes:
  """ctypes binding for the native EKFSym implementation."""

  def __init__(self, gen_dir, name, Q, x_initial, P_initial, dim_main, dim_main_err,
               N=0, dim_augment=0, dim_augment_err=0, maha_test_kinds=None,
               quaternion_idxs=None, global_vars=None, max_rewind_age=1.0, logger=None):
    del logger  # Native logging is used by the C++ implementation.
    self.dim_x = np.asarray(x_initial).size
    self.dim_err = np.asarray(P_initial).shape[0]
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    self._lib = ctypes.CDLL(str(Path(__file__).with_name(f"libekf_sym_ctypes{extension}")))
    self._configure_library()

    Q = np.ascontiguousarray(Q, dtype=np.float64)
    x_initial = np.ascontiguousarray(x_initial, dtype=np.float64)
    P_initial = np.ascontiguousarray(P_initial, dtype=np.float64)
    maha = np.ascontiguousarray(maha_test_kinds or [], dtype=np.int32)
    quaternions = np.ascontiguousarray(quaternion_idxs or [], dtype=np.int32)
    globals_encoded = [item.encode() for item in (global_vars or [])]
    globals_array = (ctypes.c_char_p * len(globals_encoded))(*globals_encoded)
    self._handle = self._lib.ekf_sym_create(
      str(gen_dir).encode(), name.encode(), _ptr(Q), _ptr(x_initial),
      _ptr(P_initial), self.dim_x, self.dim_err, dim_main, dim_main_err, N,
      dim_augment, dim_augment_err, _ptr(maha), len(maha),
      _ptr(quaternions), len(quaternions), globals_array,
      len(globals_encoded), max_rewind_age)
    if not self._handle:
      raise RuntimeError(f"failed to create native EKF {name!r}")

  def _configure_library(self):
    lib = self._lib
    lib.ekf_sym_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
      ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
      ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_double]
    lib.ekf_sym_create.restype = ctypes.c_void_p
    lib.ekf_sym_destroy.argtypes = [ctypes.c_void_p]
    lib.ekf_sym_init_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_double]
    lib.ekf_sym_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ekf_sym_covs.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ekf_sym_set_filter_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.ekf_sym_get_filter_time.argtypes = [ctypes.c_void_p]
    lib.ekf_sym_get_filter_time.restype = ctypes.c_double
    lib.ekf_sym_set_global.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.ekf_sym_reset_rewind.argtypes = [ctypes.c_void_p]
    lib.ekf_sym_predict.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.ekf_sym_predict_and_update_batch.argtypes = [ctypes.c_void_p, ctypes.c_double,
      ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
      ctypes.c_void_p, ctypes.c_void_p]
    lib.ekf_sym_predict_and_update_batch.restype = ctypes.c_int

  def __del__(self):
    handle = getattr(self, "_handle", None)
    if handle:
      self._lib.ekf_sym_destroy(handle)
      self._handle = None

  def init_state(self, state, covs, filter_time):
    state = np.ascontiguousarray(state, dtype=np.float64)
    covs = np.ascontiguousarray(covs, dtype=np.float64)
    self._lib.ekf_sym_init_state(self._handle, _ptr(state), _ptr(covs),
                                 self.dim_x, self.dim_err,
                                 math.nan if filter_time is None else filter_time)

  def state(self):
    result = np.empty(self.dim_x, dtype=np.float64)
    self._lib.ekf_sym_state(self._handle, _ptr(result))
    return result

  def covs(self):
    result = np.empty((self.dim_err, self.dim_err), dtype=np.float64)
    self._lib.ekf_sym_covs(self._handle, _ptr(result))
    return result

  def set_filter_time(self, t):
    self._lib.ekf_sym_set_filter_time(self._handle, t)

  def get_filter_time(self):
    value = self._lib.ekf_sym_get_filter_time(self._handle)
    return None if math.isnan(value) else value

  def set_global(self, global_var, val):
    self._lib.ekf_sym_set_global(self._handle, global_var.encode(), val)

  def reset_rewind(self):
    self._lib.ekf_sym_reset_rewind(self._handle)

  def predict(self, t):
    self._lib.ekf_sym_predict(self._handle, t)

  def predict_and_update_batch(self, t, kind, z, R, extra_args=None, augment=False):
    z = np.ascontiguousarray(z, dtype=np.float64)
    R = np.ascontiguousarray(R, dtype=np.float64)
    if z.ndim != 2 or R.shape != (z.shape[0], z.shape[1], z.shape[1]):
      raise ValueError("z and R must have shapes (n, z_dim) and (n, z_dim, z_dim)")
    if extra_args is None:
      extra_args = np.empty((z.shape[0], 0), dtype=np.float64)
    extra_args = np.ascontiguousarray(extra_args, dtype=np.float64)
    if extra_args.ndim == 1:
      extra_args = extra_args.reshape(z.shape[0], -1)
    if extra_args.shape[0] != z.shape[0]:
      raise ValueError("extra_args must contain one row per observation")

    xk1 = np.empty(self.dim_x, dtype=np.float64)
    xk = np.empty(self.dim_x, dtype=np.float64)
    Pk1 = np.empty((self.dim_err, self.dim_err), dtype=np.float64)
    Pk = np.empty_like(Pk1)
    y = np.empty_like(z)
    y_dims = np.empty(z.shape[0], dtype=np.int32)
    ok = self._lib.ekf_sym_predict_and_update_batch(
      self._handle, t, kind, _ptr(z), _ptr(R), _ptr(extra_args),
      z.shape[0], z.shape[1], extra_args.shape[1], augment, _ptr(xk1),
      _ptr(xk), _ptr(Pk1), _ptr(Pk), _ptr(y), _ptr(y_dims))
    if not ok:
      return None
    return xk1, xk, Pk1, Pk, t, kind, [y[i, :size].copy() for i, size in enumerate(y_dims)], z, extra_args

  def augment(self):
    raise NotImplementedError()

  def get_augment_times(self):
    raise NotImplementedError()

  def rts_smooth(self, estimates, norm_quats=False):
    raise NotImplementedError()

  def maha_test(self, x, P, kind, z, R, extra_args=None, maha_thresh=0.95):
    raise NotImplementedError()
