# cython: language_level=3
# distutils: language = c++

cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np

import numpy as np

cdef extern from "rednose/helpers/ekf_sym.h" namespace "EKFS":
  cdef cppclass MapVectorXd "Eigen::Map<Eigen::VectorXd>":
    MapVectorXd(double*, int)

  cdef cppclass MapMatrixXdr "Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >":
    MapMatrixXdr(double*, int, int)

  cdef cppclass VectorXd "Eigen::VectorXd":
    VectorXd()
    double* data()
    int rows()

  cdef cppclass MatrixXdr "Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>":
    MatrixXdr()
    double* data()
    int rows()
    int cols()

  ctypedef struct Estimate:
    VectorXd xk1
    VectorXd xk
    MatrixXdr Pk1
    MatrixXdr Pk
    double t
    int kind
    vector[VectorXd] y
    vector[VectorXd] z
    vector[vector[double]] extra_args

  cdef cppclass EKFSym:
    EKFSym(
      string name,
      MapMatrixXdr Q,
      MapVectorXd x_initial,
      MapMatrixXdr P_initial,
      int dim_main,
      int dim_main_err,
      int N,
      int dim_augment,
      int dim_augment_err,
      vector[int] maha_test_kinds,
      vector[string] global_vars,
      double max_rewind_age)

    void init_state(
      MapVectorXd state,
      MapMatrixXdr covs,
      double filter_time)

    VectorXd get_state()
    MatrixXdr get_covs()
    double get_filter_time()

    void predict_and_update_batch(
      double t,
      int kind,
      vector[MapVectorXd] z,
      vector[MapMatrixXdr] z,
      vector[vector[double]] extra_args,
      bool augment)

    # rts_smooth
    # normalize_quats

@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim=2, mode="c"] matrix_to_numpy(MatrixXdr arr):
  cdef double[:,:] mem_view = <double[:arr.rows(),:arr.cols()]>arr.data()
  cdef int itemsize = np.dtype(np.double).itemsize
  return np.asarray(mem_view, dtype=np.double, order="C")

  # cdef int size[2]
  # size[0] = arr.rows()
  # size[1] = arr.cols()
  # return np.PyArray_SimpleNewFromData(2, size, np.double, <void*>arr.data())

@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim=1, mode="c"] vector_to_numpy(VectorXd arr):
  cdef double[:] mem_view = <double[:arr.rows()]>arr.data()
  cdef int itemsize = np.dtype(np.double).itemsize
  return np.asarray(mem_view, dtype=np.double, order="C")
  # cdef int size[1]
  # size[0] = arr.rows()
  # return np.PyArray_SimpleNewFromData(1, size, NPY_FLOAT64, <void*>arr.data())

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
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Q_b = np.ascontiguousarray(Q, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] x_initial_b = np.ascontiguousarray(x_initial, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] P_initial_b = np.ascontiguousarray(P_initial, dtype=np.double)
    self.ekf = new EKFSym(
      name.encode('utf8'),
      MapMatrixXdr(<double*> Q_b.data, Q.shape[0], Q.shape[1]),
      MapVectorXd(<double*> x_initial_b.data, x_initial.shape[0]),
      MapMatrixXdr(<double*> x_initial_b.data, P_initial.shape[0], P_initial.shape[1]),
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
    np.ndarray[np.float64_t, ndim=1] state,
    np.ndarray[np.float64_t, ndim=2] covs,
    filter_time):
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] state_b = np.ascontiguousarray(state, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] covs_b = np.ascontiguousarray(covs, dtype=np.double)
    self.ekf.init_state(
      MapVectorXd(<double*> state_b.data, state.shape[0]),
      MapMatrixXdr(<double*> covs_b.data, covs.shape[0], covs.shape[1]),
      np.nan if filter_time is None else filter_time
    )

  def get_state(self):
    return vector_to_numpy(self.ekf.get_state())

  def get_covs(self):
    return matrix_to_numpy(self.ekf.get_covs())

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
    cdef vector[MapVectorXd] z_map
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] zi_b
    for zi in z:
      zi_b = np.ascontiguousarray(zi, dtype=np.double)
      z_map.push_back(MapVectorXd(<double*> zi_b.data, zi.shape[0]))

    cdef vector[MapMatrixXdr] R_map
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Ri_b
    for Ri in R:
      Ri_b = np.ascontiguousarray(Ri, dtype=np.double)
      R_map.push_back(MapMatrixXdr(<double*> Ri_b.data, Ri.shape[0], Ri.shape[1]))

    cdef vector[vector[double]] extra_args_map
    cdef vector[double] args_map
    for args in extra_args:
      args_map.clear()  # TODO new memory addr?
      for a in args:
        args_map.push_back(a)
      extra_args_map.push_back(args_map)

    self.ekf.predict_and_update_batch(t, kind, z_map, R_map, extra_args_map, augment)
    #cdef VectorXd tmpvec
    #return (
    #  ndarray_copy(res.xk1).flatten(),
    #  ndarray_copy(res.xk).flatten(),
    #  ndarray_copy(res.Pk1),
    #  ndarray_copy(res.Pk),
    #  res.t,
    #  res.kind,
    #  [ndarray_copy(tmpvec).flatten() for tmpvec in res.y],
    #  [ndarray_copy(tmpvec).flatten() for tmpvec in res.z],
    #  extra_args, # TODO take return
    #)

  def __dealloc__(self):
    del self.ekf
