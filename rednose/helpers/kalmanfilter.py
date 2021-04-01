from typing import Any, Dict

import numpy as np

from rednose.helpers.ekf_sym_py import EKF_sym


class KalmanFilter:
  name = "<name>"
  initial_x = np.zeros((0, 0))
  initial_P_diag = np.zeros((0, 0))
  Q = np.zeros((0, 0))
  obs_noise: Dict[int, Any] = {}

  def __init__(self, generated_dir):
    dim_state = self.initial_x.shape[0]
    dim_state_err = self.initial_P_diag.shape[0]

    # init filter
    self.filter = EKF_sym(generated_dir, self.name, self.Q, self.initial_x, np.diag(self.initial_P_diag), dim_state, dim_state_err)

  @property
  def x(self):
    return self.filter.state()

  @property
  def t(self):
    return self.filter.filter_time

  @property
  def P(self):
    return self.filter.covs()

  def init_state(self, state, covs_diag=None, covs=None, filter_time=None):
    if covs_diag is not None:
      P = np.diag(covs_diag)
    elif covs is not None:
      P = covs
    else:
      P = self.filter.covs()
    self.filter.init_state(state, P, filter_time)

  def get_R(self, kind, n):
    obs_noise = self.obs_noise[kind]
    dim = obs_noise.shape[0]
    R = np.zeros((n, dim, dim))
    for i in range(n):
      R[i, :, :] = obs_noise
    return R

  def predict_and_observe(self, t, kind, data, R=None):
    if len(data) > 0:
      data = np.atleast_2d(data)

    if R is None:
      R = self.get_R(kind, len(data))

    self.filter.predict_and_update_batch(t, kind, data, R)
