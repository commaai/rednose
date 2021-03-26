# cython: language_level=3
# distutils: language = c++

cimport cython
from libcpp.string cimport string
cimport numpy as np

import numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from rednose.helpers.eigency.core_pyx cimport *

cdef extern from "rednose/helpers/ekf_sym.h" namespace "EKFS":
  cdef cppclass EKFSym:
    EKFSym(
      string name,
      Map[VectorXd] Q,
      Map[VectorXd] x_initial,
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] P_initial,
      int dim_main,
      int dim_main_err,
      int N,
      int dim_augment,
      int dim_augment_err,
      vector[int] maha_test_kinds,
      vector[string] global_vars,
      double max_rewind_age)

    Map[VectorXd] state()
    double filter_time
    FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] covs()
    void init_state(
      Map[VectorXd] state,
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] covs,
      double filter_time)
    predict_and_update_batch(
      double t,
      int kind,
      Map[VectorXd] z,
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] R,
      vector[vector[double]] extra_args,
      bool augment)
    # rts_smooth
    # normalize_quats

cdef class EKF_sym:
  cdef EKFSym* ekf
  def __cinit__(self, str name, np.ndarray Q, np.ndarray x_initial, np.ndarray P_initial, int dim_main,
      int dim_main_err, int N=0, int dim_augment=0, int dim_augment_err=0, list maha_test_kinds=[],
      list global_vars=[], float max_rewind_age=1.0):
    self.ekf = new EKFSym(
      name.encode('utf8'),
      Map[VectorXd](Q),
      Map[VectorXd](x_initial),
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor](P_initial),
      dim_main,
      dim_main_err,
      N,
      dim_augment,
      dim_augment_err,
      maha_test_kinds,
      global_vars,
      max_rewind_age
    )

  def __dealloc__(self):
    del self.ekf
