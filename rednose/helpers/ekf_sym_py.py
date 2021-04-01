import os
import logging
from bisect import bisect_right

import numpy as np
import sympy as sp
from numpy import dot

from rednose.helpers.sympy_helpers import sympy_into_c
from rednose.helpers import (TEMPLATE_DIR, load_code, write_code)
from rednose.helpers.chi2_lookup import chi2_ppf


def solve(a, b):
  if a.shape[0] == 1 and a.shape[1] == 1:
    return b / a[0][0]
  else:
    return np.linalg.solve(a, b)


def null(H, eps=1e-12):
  _, s, vh = np.linalg.svd(H)
  padding = max(0, np.shape(H)[1] - np.shape(s)[0])
  null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
  null_space = np.compress(null_mask, vh, axis=0)
  return np.transpose(null_space)


class EKF_sym():
  def __init__(self, folder, name, Q, x_initial, P_initial, dim_main, dim_main_err,  # pylint: disable=dangerous-default-value
               N=0, dim_augment=0, dim_augment_err=0, maha_test_kinds=[], global_vars=None, max_rewind_age=1.0, logger=logging):
    """Generates process function and all observation functions for the kalman filter."""
    self.msckf = N > 0
    self.N = N
    self.dim_augment = dim_augment
    self.dim_augment_err = dim_augment_err
    self.dim_main = dim_main
    self.dim_main_err = dim_main_err

    self.logger = logger

    # state
    x_initial = x_initial.reshape((-1, 1))
    self.dim_x = x_initial.shape[0]
    self.dim_err = P_initial.shape[0]
    assert dim_main + dim_augment * N == self.dim_x
    assert dim_main_err + dim_augment_err * N == self.dim_err
    assert Q.shape == P_initial.shape

    # kinds that should get mahalanobis distance
    # tested for outlier rejection
    self.maha_test_kinds = maha_test_kinds

    self.global_vars = global_vars

    # process noise
    self.Q = Q

    # rewind stuff
    self.max_rewind_age = max_rewind_age
    self.rewind_t = []
    self.rewind_states = []
    self.rewind_obscache = []
    self.init_state(x_initial, P_initial, None)

    ffi, lib = load_code(folder, name, "kf")
    kinds, self.feature_track_kinds = [], []
    for func in dir(lib):
      if func[:len(name) + 3] == f'{name}_h_':
        kinds.append(int(func[len(name) + 3:]))
      if func[:len(name) + 4] == f'{name}_He_':
        self.feature_track_kinds.append(int(func[len(name) + 4:]))

    # wrap all the sympy functions
    def wrap_1lists(func_name):
      func = eval(f"lib.{name}_{func_name}", {"lib": lib})  # pylint: disable=eval-used

      def ret(lst1, out):
        func(ffi.cast("double *", lst1.ctypes.data),
             ffi.cast("double *", out.ctypes.data))
      return ret

    def wrap_2lists(func_name):
      func = eval(f"lib.{name}_{func_name}", {"lib": lib})  # pylint: disable=eval-used

      def ret(lst1, lst2, out):
        func(ffi.cast("double *", lst1.ctypes.data),
             ffi.cast("double *", lst2.ctypes.data),
             ffi.cast("double *", out.ctypes.data))
      return ret

    def wrap_1list_1float(func_name):
      func = eval(f"lib.{name}_{func_name}", {"lib": lib})  # pylint: disable=eval-used

      def ret(lst1, fl, out):
        func(ffi.cast("double *", lst1.ctypes.data),
             ffi.cast("double", fl),
             ffi.cast("double *", out.ctypes.data))
      return ret

    self.f = wrap_1list_1float("f_fun")
    self.F = wrap_1list_1float("F_fun")

    self.err_function = wrap_2lists("err_fun")
    self.inv_err_function = wrap_2lists("inv_err_fun")
    self.H_mod = wrap_1lists("H_mod_fun")

    self.hs, self.Hs, self.Hes = {}, {}, {}
    for kind in kinds:
      self.hs[kind] = wrap_2lists(f"h_{kind}")
      self.Hs[kind] = wrap_2lists(f"H_{kind}")
      if self.msckf and kind in self.feature_track_kinds:
        self.Hes[kind] = wrap_2lists(f"He_{kind}")

    if self.global_vars is not None:
      for var in self.global_vars:
        fun_name = f"{name}_set_{var.name}"
        setattr(self, fun_name, getattr(lib, fun_name))

  def init_state(self, state, covs, filter_time):
    self.x = np.array(state.reshape((-1, 1))).astype(np.float64)
    self.P = np.array(covs).astype(np.float64)
    self.filter_time = filter_time
    self.augment_times = [0] * self.N
    self.rewind_obscache = []
    self.rewind_t = []
    self.rewind_states = []

  def reset_rewind(self):
    self.rewind_obscache = []
    self.rewind_t = []
    self.rewind_states = []

  def augment(self):
    # TODO this is not a generalized way of doing this and implies that the augmented states
    # are simply the first (dim_augment_state) elements of the main state.
    assert self.msckf
    d1 = self.dim_main
    d2 = self.dim_main_err
    d3 = self.dim_augment
    d4 = self.dim_augment_err

    # push through augmented states
    self.x[d1:-d3] = self.x[d1 + d3:]
    self.x[-d3:] = self.x[:d3]
    assert self.x.shape == (self.dim_x, 1)

    # push through augmented covs
    assert self.P.shape == (self.dim_err, self.dim_err)
    P_reduced = self.P
    P_reduced = np.delete(P_reduced, np.s_[d2:d2 + d4], axis=1)
    P_reduced = np.delete(P_reduced, np.s_[d2:d2 + d4], axis=0)
    assert P_reduced.shape == (self.dim_err - d4, self.dim_err - d4)
    to_mult = np.zeros((self.dim_err, self.dim_err - d4))
    to_mult[:-d4, :] = np.eye(self.dim_err - d4)
    to_mult[-d4:, :d4] = np.eye(d4)
    self.P = to_mult.dot(P_reduced.dot(to_mult.T))
    self.augment_times = self.augment_times[1:]
    self.augment_times.append(self.filter_time)
    assert self.P.shape == (self.dim_err, self.dim_err)

  def get_state(self):
    return np.array(self.x).flatten()

  def get_covs(self):
    return self.P

  def get_filter_time(self):
    return self.filter_time

  def normalize_state(self, slice_start, slice_end_ex):
    self.x[slice_start:slice_end_ex] /= np.linalg.norm(self.x[slice_start:slice_end_ex])

  def rewind(self, t):
    # find where we are rewinding to
    idx = bisect_right(self.rewind_t, t)
    assert self.rewind_t[idx - 1] <= t
    assert self.rewind_t[idx] > t    # must be true, or rewind wouldn't be called

    # set the state to the time right before that
    self.filter_time = self.rewind_t[idx - 1]
    self.x[:] = self.rewind_states[idx - 1][0]
    self.P[:] = self.rewind_states[idx - 1][1]

    # return the observations we rewound over for fast forwarding
    ret = self.rewind_obscache[idx:]

    # throw away the old future
    # TODO: is this making a copy?
    self.rewind_t = self.rewind_t[:idx]
    self.rewind_states = self.rewind_states[:idx]
    self.rewind_obscache = self.rewind_obscache[:idx]

    return ret

  def checkpoint(self, obs):
    # push to rewinder
    self.rewind_t.append(self.filter_time)
    self.rewind_states.append((np.copy(self.x), np.copy(self.P)))
    self.rewind_obscache.append(obs)

    # only keep a certain number around
    REWIND_TO_KEEP = 512
    self.rewind_t = self.rewind_t[-REWIND_TO_KEEP:]
    self.rewind_states = self.rewind_states[-REWIND_TO_KEEP:]
    self.rewind_obscache = self.rewind_obscache[-REWIND_TO_KEEP:]

  def predict(self, t):
    # initialize time
    if self.filter_time is None:
      self.filter_time = t

    # predict
    dt = t - self.filter_time
    assert dt >= 0
    self.x, self.P = self._predict(self.x, self.P, dt)
    self.filter_time = t

  def predict_and_update_batch(self, t, kind, z, R, extra_args=[[]], augment=False):  # pylint: disable=dangerous-default-value
    # TODO handle rewinding at this level"

    # rewind
    if self.filter_time is not None and t < self.filter_time:
      if len(self.rewind_t) == 0 or t < self.rewind_t[0] or t < self.rewind_t[-1] - self.max_rewind_age:
        self.logger.error("observation too old at %.3f with filter at %.3f, ignoring" % (t, self.filter_time))
        return None
      rewound = self.rewind(t)
    else:
      rewound = []

    ret = self._predict_and_update_batch(t, kind, z, R, extra_args, augment)

    # optional fast forward
    for r in rewound:
      self._predict_and_update_batch(*r)

    return ret

  def _predict_and_update_batch(self, t, kind, z, R, extra_args, augment=False):
    """The main kalman filter function
    Predicts the state and then updates a batch of observations
    dim_x: dimensionality of the state space
    dim_z: dimensionality of the observation and depends on kind
    n: number of observations
    Args:
      t                 (float): Time of observation
      kind                (int): Type of observation
      z         (vec [n,dim_z]): Measurements
      R  (mat [n,dim_z, dim_z]): Measurement Noise
      extra_args    (list, [n]): Values used in H computations
    """
    assert z.shape[0] == R.shape[0]
    assert z.shape[1] == R.shape[1]
    assert z.shape[1] == R.shape[2]

    # initialize time
    if self.filter_time is None:
      self.filter_time = t

    # predict
    dt = t - self.filter_time
    assert dt >= 0
    self.x, self.P = self._predict(self.x, self.P, dt)
    self.filter_time = t
    xk_km1, Pk_km1 = np.copy(self.x).flatten(), np.copy(self.P)

    # update batch
    y = []
    for i in range(len(z)):
      # these are from the user, so we canonicalize them
      z_i = np.array(z[i], dtype=np.float64, order='F')
      R_i = np.array(R[i], dtype=np.float64, order='F')
      extra_args_i = np.array(extra_args[i], dtype=np.float64, order='F')
      # update
      self.x, self.P, y_i = self._update(self.x, self.P, kind, z_i, R_i, extra_args=extra_args_i)
      y.append(y_i)
    xk_k, Pk_k = np.copy(self.x).flatten(), np.copy(self.P)

    if augment:
      self.augment()

    # checkpoint
    self.checkpoint((t, kind, z, R, extra_args))

    return xk_km1, xk_k, Pk_km1, Pk_k, t, kind, y, z, extra_args

  def _predict(self, x, P, dt):
    x_new = np.zeros(x.shape, dtype=np.float64)
    self.f(x, dt, x_new)

    F = np.zeros(P.shape, dtype=np.float64)
    self.F(x, dt, F)

    if not self.msckf:
      P = dot(dot(F, P), F.T)
    else:
      # Update the predicted state covariance:
      #  Pk+1|k   =  |F*Pii*FT + Q*dt   F*Pij |
      #              |PijT*FT           Pjj   |
      # Where F is the jacobian of the main state
      # predict function, Pii is the main state's
      # covariance and Q its process noise. Pij
      # is the covariance between the augmented
      # states and the main state.
      #
      d2 = self.dim_main_err    # known at compile time
      F_curr = F[:d2, :d2]
      P[:d2, :d2] = (F_curr.dot(P[:d2, :d2])).dot(F_curr.T)
      P[:d2, d2:] = F_curr.dot(P[:d2, d2:])
      P[d2:, :d2] = P[d2:, :d2].dot(F_curr.T)

    P += dt * self.Q
    return x_new, P

  def _update(self, x, P, kind, z, R, extra_args=[]):  # pylint: disable=dangerous-default-value
    # init vars
    z = z.reshape((-1, 1))
    h = np.zeros(z.shape, dtype=np.float64)
    H = np.zeros((z.shape[0], self.dim_x), dtype=np.float64)

    # C functions
    self.hs[kind](x, extra_args, h)
    self.Hs[kind](x, extra_args, H)

    # y is the "loss"
    y = z - h

    # *** same above this line ***

    if self.msckf and kind in self.Hes:
      # Do some algebraic magic to decorrelate
      He = np.zeros((z.shape[0], len(extra_args)), dtype=np.float64)
      self.Hes[kind](x, extra_args, He)

      # TODO: Don't call a function here, do projection locally
      A = null(He.T)

      y = A.T.dot(y)
      H = A.T.dot(H)
      R = A.T.dot(R.dot(A))

      # TODO If nullspace isn't the dimension we want
      if A.shape[1] + He.shape[1] != A.shape[0]:
        self.logger.warning('Warning: null space projection failed, measurement ignored')
        return x, P, np.zeros(A.shape[0] - He.shape[1])

    # if using eskf
    H_mod = np.zeros((x.shape[0], P.shape[0]), dtype=np.float64)
    self.H_mod(x, H_mod)
    H = H.dot(H_mod)

    # Do mahalobis distance test
    # currently just runs on msckf observations
    # could run on anything if needed
    if self.msckf and kind in self.maha_test_kinds:
      a = np.linalg.inv(H.dot(P).dot(H.T) + R)
      maha_dist = y.T.dot(a.dot(y))
      if maha_dist > chi2_ppf(0.95, y.shape[0]):
        R = 10e16 * R

    # *** same below this line ***

    # Outlier resilient weighting as described in:
    # "A Kalman Filter for Robust Outlier Detection - Jo-Anne Ting, ..."
    weight = 1  # (1.5)/(1 + np.sum(y**2)/np.sum(R))

    S = dot(dot(H, P), H.T) + R / weight
    K = solve(S, dot(H, P.T)).T
    I_KH = np.eye(P.shape[0]) - dot(K, H)

    # update actual state
    delta_x = dot(K, y)
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

    # inject observed error into state
    x_new = np.zeros(x.shape, dtype=np.float64)
    self.err_function(x, delta_x, x_new)
    return x_new, P, y.flatten()

  def maha_test(self, x, P, kind, z, R, extra_args=[], maha_thresh=0.95):  # pylint: disable=dangerous-default-value
    # init vars
    z = z.reshape((-1, 1))
    h = np.zeros(z.shape, dtype=np.float64)
    H = np.zeros((z.shape[0], self.dim_x), dtype=np.float64)

    # C functions
    self.hs[kind](x, extra_args, h)
    self.Hs[kind](x, extra_args, H)

    # y is the "loss"
    y = z - h

    # if using eskf
    H_mod = np.zeros((x.shape[0], P.shape[0]), dtype=np.float64)
    self.H_mod(x, H_mod)
    H = H.dot(H_mod)

    a = np.linalg.inv(H.dot(P).dot(H.T) + R)
    maha_dist = y.T.dot(a.dot(y))
    if maha_dist > chi2_ppf(maha_thresh, y.shape[0]):
      return False
    else:
      return True

  def rts_smooth(self, estimates, norm_quats=False):
    '''
    Returns rts smoothed results of
    kalman filter estimates
    If the kalman state is augmented with
    old states only the main state is smoothed
    '''
    xk_n = estimates[-1][0]
    Pk_n = estimates[-1][2]
    Fk_1 = np.zeros(Pk_n.shape, dtype=np.float64)

    states_smoothed = [xk_n]
    covs_smoothed = [Pk_n]
    for k in range(len(estimates) - 2, -1, -1):
      xk1_n = xk_n
      if norm_quats:
        xk1_n[3:7] /= np.linalg.norm(xk1_n[3:7])
      Pk1_n = Pk_n

      xk1_k, _, Pk1_k, _, t2, _, _, _, _ = estimates[k + 1]
      _, xk_k, _, Pk_k, t1, _, _, _, _ = estimates[k]
      dt = t2 - t1
      self.F(xk_k, dt, Fk_1)

      d1 = self.dim_main
      d2 = self.dim_main_err
      Ck = np.linalg.solve(Pk1_k[:d2, :d2], Fk_1[:d2, :d2].dot(Pk_k[:d2, :d2].T)).T
      xk_n = xk_k
      delta_x = np.zeros((Pk_n.shape[0], 1), dtype=np.float64)
      self.inv_err_function(xk1_k, xk1_n, delta_x)
      delta_x[:d2] = Ck.dot(delta_x[:d2])
      x_new = np.zeros((xk_n.shape[0], 1), dtype=np.float64)
      self.err_function(xk_k, delta_x, x_new)
      xk_n[:d1] = x_new[:d1, 0]
      Pk_n = Pk_k
      Pk_n[:d2, :d2] = Pk_k[:d2, :d2] + Ck.dot(Pk1_n[:d2, :d2] - Pk1_k[:d2, :d2]).dot(Ck.T)
      states_smoothed.append(xk_n)
      covs_smoothed.append(Pk_n)

    return np.flipud(np.vstack(states_smoothed)), np.stack(covs_smoothed, 0)[::-1]
