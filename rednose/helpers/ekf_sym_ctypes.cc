#include "ekf_sym.h"

#include <algorithm>
#include <cstring>

using EKFS::EKFSym;
using EKFS::Estimate;
using EKFS::MatrixXdr;
using Eigen::Map;
using Eigen::VectorXd;

extern "C" {

void *ekf_sym_create(const char *directory, const char *name, double *Q, double *x_initial,
                     double *P_initial, int dim_x, int dim_err, int dim_main, int dim_main_err,
                     int N, int dim_augment, int dim_augment_err, const int *maha_test_kinds,
                     int maha_count, const int *quaternion_idxs, int quaternion_count,
                     const char *const *global_vars, int global_count, double max_rewind_age) {
  ekf_load_and_register(directory, name);
  std::vector<int> maha;
  std::vector<int> quaternions;
  if (maha_count) maha.assign(maha_test_kinds, maha_test_kinds + maha_count);
  if (quaternion_count) quaternions.assign(quaternion_idxs, quaternion_idxs + quaternion_count);
  std::vector<std::string> globals;
  for (int i = 0; i < global_count; ++i) globals.emplace_back(global_vars[i]);
  return new EKFSym(name, Map<MatrixXdr>(Q, dim_err, dim_err), Map<VectorXd>(x_initial, dim_x),
                    Map<MatrixXdr>(P_initial, dim_err, dim_err), dim_main, dim_main_err, N,
                    dim_augment, dim_augment_err, maha, quaternions, globals, max_rewind_age);
}

void ekf_sym_destroy(void *handle) { delete static_cast<EKFSym *>(handle); }

void ekf_sym_init_state(void *handle, double *state, double *covs, int dim_x, int dim_err,
                        double filter_time) {
  static_cast<EKFSym *>(handle)->init_state(Map<VectorXd>(state, dim_x),
                                            Map<MatrixXdr>(covs, dim_err, dim_err), filter_time);
}

void ekf_sym_state(void *handle, double *out) {
  const VectorXd state = static_cast<EKFSym *>(handle)->state();
  std::memcpy(out, state.data(), state.size() * sizeof(double));
}

void ekf_sym_covs(void *handle, double *out) {
  const MatrixXdr covs = static_cast<EKFSym *>(handle)->covs();
  std::memcpy(out, covs.data(), covs.size() * sizeof(double));
}

void ekf_sym_set_filter_time(void *handle, double t) { static_cast<EKFSym *>(handle)->set_filter_time(t); }
double ekf_sym_get_filter_time(void *handle) { return static_cast<EKFSym *>(handle)->get_filter_time(); }
void ekf_sym_set_global(void *handle, const char *name, double value) {
  static_cast<EKFSym *>(handle)->set_global(name, value);
}
void ekf_sym_reset_rewind(void *handle) { static_cast<EKFSym *>(handle)->reset_rewind(); }
void ekf_sym_predict(void *handle, double t) { static_cast<EKFSym *>(handle)->predict(t); }

int ekf_sym_predict_and_update_batch(void *handle, double t, int kind, double *z, double *R,
                                     double *extra_args, int count, int z_dim, int extra_dim,
                                     bool augment, double *xk1, double *xk, double *Pk1, double *Pk,
                                     double *y, int *y_dims) {
  std::vector<Map<VectorXd>> z_maps;
  std::vector<Map<MatrixXdr>> R_maps;
  std::vector<std::vector<double>> extras;
  z_maps.reserve(count);
  R_maps.reserve(count);
  extras.reserve(count);
  for (int i = 0; i < count; ++i) {
    z_maps.emplace_back(z + i * z_dim, z_dim);
    R_maps.emplace_back(R + i * z_dim * z_dim, z_dim, z_dim);
    extras.emplace_back();
    if (extra_dim) extras.back().assign(extra_args + i * extra_dim, extra_args + (i + 1) * extra_dim);
  }

  const auto result = static_cast<EKFSym *>(handle)->predict_and_update_batch(
      t, kind, z_maps, R_maps, extras, augment);
  if (!result.has_value()) return 0;

  const Estimate &estimate = result.value();
  std::memcpy(xk1, estimate.xk1.data(), estimate.xk1.size() * sizeof(double));
  std::memcpy(xk, estimate.xk.data(), estimate.xk.size() * sizeof(double));
  std::memcpy(Pk1, estimate.Pk1.data(), estimate.Pk1.size() * sizeof(double));
  std::memcpy(Pk, estimate.Pk.data(), estimate.Pk.size() * sizeof(double));
  for (int i = 0; i < count; ++i) {
    y_dims[i] = estimate.y[i].size();
    std::memcpy(y + i * z_dim, estimate.y[i].data(), estimate.y[i].size() * sizeof(double));
  }
  return 1;
}

}
