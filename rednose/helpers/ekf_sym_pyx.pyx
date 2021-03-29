# cython: language_level=3
# distutils: language = c++

cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np

import numpy as np

from rednose.helpers.eigency.core_pyx cimport *

ctypedef vector[double] vector_double

# TODO replace PlainObjectBase with Matrix[double, Dynamic, Dynamic, RowMajor]
cdef extern from "rednose/helpers/ekf_sym.h" namespace "EKFS":
  ctypedef struct Estimate:
    VectorXd xk1
    VectorXd xk
    PlainObjectBase Pk1
    PlainObjectBase Pk
    double t
    int kind
    vector[VectorXd] y
    vector[VectorXd] z
    vector[vector_double] extra_args

  cdef cppclass EKFSym:
    EKFSym(
      string name,
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] Q,
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

    void init_state(
      Map[VectorXd] state,
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor] covs,
      double filter_time)

    VectorXd get_state()
    PlainObjectBase get_covs()
    double get_filter_time()

    Estimate predict_and_update_batch(
      double t,
      int kind,
      vector[Map[VectorXd]] z,
      vector[FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor]] z,
      vector[vector[double]] extra_args,
      bool augment)

    # rts_smooth
    # normalize_quats

cdef class EKF_sym:
  cdef EKFSym* ekf
  def __cinit__(
    self,
    str gen_dir,
    str name,
    np.ndarray[np.float64_t, ndim=2] Q,
    np.ndarray[np.float64_t] x_initial,
    np.ndarray[np.float64_t, ndim=2] P_initial,
    int dim_main,
    int dim_main_err,
    int N=0,
    int dim_augment=0,
    int dim_augment_err=0,
    list maha_test_kinds=[],
    list global_vars=[],
    float max_rewind_age=1.0,
    logger=None):
    assert x_initial.ndim <= 2
    self.ekf = new EKFSym(
      name.encode('utf8'),
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor](Q),
      Map[VectorXd](x_initial if x_initial.ndim == 1 else x_initial[0]),
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

  def init_state(
    self,
    np.ndarray[np.float64_t] state,
    np.ndarray[np.float64_t, ndim=2] covs,
    filter_time):
    assert state.ndim <= 2
    self.ekf.init_state(
      Map[VectorXd](state if state.ndim == 1 else state[0]),
      FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor](covs),
      np.nan if filter_time is None else filter_time
    )

  def get_state(self):
    return ndarray_copy(self.ekf.get_state()).flatten()

  def get_covs(self):
    return ndarray_copy(self.ekf.get_covs())

  def get_filter_time(self):
    return self.ekf.get_filter_time()

  def predict_and_update_batch(
    self,
    double t,
    int kind,
    z,
    R,
    extra_args=[[]],
    bool augment=False):
    cdef vector[Map[VectorXd]] z_map
    for zi in z:
      z_map.push_back(Map[VectorXd](zi))

    cdef vector[FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor]] R_map
    for Ri in R:
      R_map.push_back(FlattenedMapWithOrder[Matrix, double, Dynamic, Dynamic, RowMajor](Ri))

    cdef vector[vector[double]] extra_args_map
    cdef vector[double] args_map
    for args in extra_args:
      args_map.clear()  # TODO new memory addr?
      for a in args:
        args_map.push_back(a)
      extra_args_map.push_back(args_map)

    cdef Estimate res = self.ekf.predict_and_update_batch(t, kind, z_map, R_map, extra_args_map, augment)
    cdef VectorXd tmpvec
    return (
      ndarray_copy(res.xk1).flatten(),
      ndarray_copy(res.xk).flatten(),
      ndarray_copy(res.Pk1),
      ndarray_copy(res.Pk),
      res.t,
      res.kind,
      [ndarray_copy(tmpvec).flatten() for tmpvec in res.y],
      [ndarray_copy(tmpvec).flatten() for tmpvec in res.z],
      extra_args,
    )

  def __dealloc__(self):
    del self.ekf
