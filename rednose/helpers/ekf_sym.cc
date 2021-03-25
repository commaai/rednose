#include "ekf_sym.h"

using namespace EKFS;
using namespace Eigen;

EKFSym::EKFSym(
  std::string name,
  VectorXd Q,
  VectorXd x_initial,
  MatrixXd P_initial,
  int dim_main,
  int dim_main_err,
  int N,
  int dim_augment,
  int dim_augment_err,
  std::vector<int> maha_test_kinds,
  std::vector<std::string> global_vars,
  double max_rewind_age
) :
  N(N),
  dim_augment(dim_augment),
  dim_augment_err(dim_augment_err),
  dim_main(dim_main),
  dim_main_err(dim_main_err),
  maha_test_kinds(maha_test_kinds),
  global_vars(global_vars),
  Q(Q),
  max_rewind_age(max_rewind_age)
{
  this->msckf = N > 0;

  this->dim_x = x_initial.rows();
  this->dim_err = P_initial.rows();

  assert(dim_main + dim_augment * N == dim_x);
  assert(dim_main_err + dim_augment_err * N == this->dim_err);
  assert(Q.rows() == P_initial.rows() && Q.cols() == P_initial.cols());

  // rewind stuff
  this->rewind_t = std::vector<double>();
  this->rewind_states = std::vector<int>();
  this->rewind_obscache = std::vector<int>();
  this->init_state(x_initial, P_initial, NAN);

  // Load shared library for kalman specific kinds
  this->feature_track_kinds = {}; // 'He_'
  std::vector<int> kinds = { 3, 4, 9, 10, 12, 31, 32, 13, 14, 19 }; // 'h_'

  this->f_dfun = f_fun;
  this->F_dfun = F_fun;

  this->err_dfun = err_fun;
  this->inv_err_dfun = inv_err_fun;
  this->H_mod_dfun = H_mod_fun;

  this->predict_dfun = predict;

  for (int kind : kinds) {
    // TODO fill in for h_, H_ and update_
  }
  this->h_dfuns[3] = h_3;
  this->h_dfuns[4] = h_4;
  this->h_dfuns[9] = h_9;
  this->h_dfuns[10] = h_10;
  this->h_dfuns[12] = h_12;
  this->h_dfuns[13] = h_13;
  this->h_dfuns[14] = h_14;
  this->h_dfuns[19] = h_19;
  this->h_dfuns[31] = h_31;
  this->h_dfuns[32] = h_32;

  this->H_dfuns[3] = H_3;
  this->H_dfuns[4] = H_4;
  this->H_dfuns[9] = H_9;
  this->H_dfuns[10] = H_10;
  this->H_dfuns[12] = H_12;
  this->H_dfuns[13] = H_13;
  this->H_dfuns[14] = H_14;
  this->H_dfuns[19] = H_19;
  this->H_dfuns[31] = H_31;
  this->H_dfuns[32] = H_32;

  this->update_dfuns[3] = update_3;
  this->update_dfuns[4] = update_4;
  this->update_dfuns[9] = update_9;
  this->update_dfuns[10] = update_10;
  this->update_dfuns[12] = update_12;
  this->update_dfuns[13] = update_13;
  this->update_dfuns[14] = update_14;
  this->update_dfuns[19] = update_19;
  this->update_dfuns[31] = update_31;
  this->update_dfuns[32] = update_32;

  /* if (this->msckf) {
    for (int kind : this->feature_track_kinds) {
      this->He_dfuns[kind] = He_<kind>; // TODO fix
    }
  } */

  /* for (std::string glob : this->global_vars) {
    this->set_global_dfuns[glob] = set_[glob];
  } */
}

std::tuple<VectorXd, MatrixXd, VectorXd> EKFSym::_update(
  VectorXd x,
  MatrixXd P,
  int kind,
  VectorXd z,
  MatrixXd R,
  std::vector<double> extra_args)
{
  this->update_dfuns[kind](x.data(), P.data(), z.data(), R.data(), extra_args.data());
  VectorXd y;
  if (this->msckf && std::find(this->feature_track_kinds.begin(), this->feature_track_kinds.end(), kind) != this->feature_track_kinds.end()) {
    y = z.head(z.rows() - extra_args.size());
  } else {
    y = z;
  }
  return std::tuple(x, P, y);
}

std::pair<VectorXd, MatrixXd> EKFSym::_predict(VectorXd x, MatrixXd P, double dt) {
  this->predict_dfun(x.data(), P.data(), this->Q.data(), dt);
  return std::pair(x, P);
}

void EKFSym::init_state(VectorXd state, MatrixXd covs, double filter_time) {
  this->x = state;
  this->P = covs;
  this->filter_time = filter_time;
  this->augment_times = VectorXd::Zero(this->N);
  this->reset_rewind();
}

void EKFSym::reset_rewind() {
  this->rewind_obscache.clear();
  this->rewind_t.clear();
  this->rewind_states.clear();
}

void EKFSym::augment() {
  // TODO this is not a generalized way of doing this and implies that the augmented states
  // are simply the first (dim_augment_state) elements of the main state.
  assert(this->msckf);

  int d1 = this->dim_main;
  int d2 = this->dim_main_err;
  int d3 = this->dim_augment;
  int d4 = this->dim_augment_err;

  // push through augmented states
  this->x.segment(d1, this->x.rows() - d1 - d3) = this->x.tail(this->x.rows() - d1 - d3);
  assert(this->x.rows() == this->dim_x);

  // push through augmented covs
  assert(this->P.rows() == this->dim_err && this->P.cols() == this->dim_err);
  MatrixXd P_reduced = this->P;
  P_reduced.block(d2, 0, d4, P_reduced.cols()) = P_reduced.block(d2 + d4, 0, d4, P_reduced.cols());
  P_reduced.block(0, d2, P_reduced.rows(), d4) = P_reduced.block(0, d2 + d4, P_reduced.rows(), d4);
  P_reduced.conservativeResize(this->dim_err - d4, this->dim_err - d4);
  assert(P_reduced.rows() == this->dim_err - d4 && P_reduced.cols() == this->dim_err - d4);
  MatrixXd to_mult = MatrixXd::Zero(this->dim_err, this->dim_err - d4);
  /* TODO to_mult.block(0, 0, to_mult.rows() - d4, to_mult.cols()) = MatrixXd::Identity(this->dim_err - d4, this->dim_err - d4);
  to_mult(lastN(d4), seq(0, d4 - 1)) = MatrixXd::Identity(d4, d4);
  this->P = to_mult.dot(P_reduced.dot(to_mult.transpose()));
  this->augment_times(seq(0, last - 1)) = this->augment_times(seq(1, last));
  this->augment_times.tail(1) = this->filter_time;
  assert(this->P.rows() == this->dim_err && this->P.cols() == this->dim_err);*/
}

VectorXd EKFSym::state() {
  return this->x;
}

MatrixXd EKFSym::covs() {
  return this->P;
}

void EKFSym::_predict(double t) {
  // initialize time
  if (this->filter_time == NAN) {
    this->filter_time = t;
  }

  // predict
  double dt = t - this->filter_time;
  assert(dt >= 0.0);
  auto res = this->_predict(this->x, this->P, dt);
  this->x = res.first;
  this->P = res.second;
  this->filter_time = t;
}

Estimate EKFSym::predict_and_update_batch(
  double t,
  int kind,
  std::vector<VectorXd> z,
  std::vector<MatrixXd> R,
  std::vector<std::vector<double>> extra_args,
  bool augment)
{
  // TODO handle rewinding at this level

  /*// rewind
  if (this->filter_time != NAN && t < this->filter_time) {
    if (this->rewind_t.size() == 0 || t < this->rewind_t[0] || t < this->rewind_t[this->rewind_t.size()-1] - this->max_rewind_age) {
      self.logger.error("observation too old at %.3f with filter at %.3f, ignoring" % (t, self.filter_time))
      return;
    }
    rewound = self.rewind(t)
  } else {
    rewound = []
  }*/

  auto ret = this->_predict_and_update_batch(t, kind, z, R, extra_args, augment);

  // optional fast forward
  /*for r in rewound:
    self._predict_and_update_batch(*r)*/

  return ret;
}

Estimate EKFSym::_predict_and_update_batch(
  double t,
  int kind,
  std::vector<VectorXd> z,
  std::vector<MatrixXd> R,
  std::vector<std::vector<double>> extra_args,
  bool augment)
{
  assert(z.size() == R.size());

  this->_predict(t);

  VectorXd xk_km1 = this->x;
  MatrixXd Pk_km1 = this->P;

  // update batch
  std::vector<VectorXd> y;
  for (int i = 0; i < z.size(); i++) {
    assert(z[i].rows() == R[i].rows());
    assert(z[i].rows() == R[i].cols());
    // update
    auto res = this->_update(this->x, this->P, kind, z[i], R[i], extra_args[i]);
    this->x = std::get<0>(res);
    this->P = std::get<1>(res);
    y.push_back(std::get<2>(res));
  }

  VectorXd xk_k = this->x;
  MatrixXd Pk_k = this->P;

  if (augment) {
    this->augment();
  }

  // checkpoint TODO rewind
  //this->checkpoint((t, kind, z, R, extra_args))

  return { xk_km1, xk_k, Pk_km1, Pk_k, t, kind, y, z, extra_args };
}

bool EKFSym::maha_test(VectorXd x, MatrixXd P, int kind, VectorXd z, MatrixXd R, std::vector<double> extra_args, double maha_thresh) {
  // init vars
  VectorXd h = VectorXd::Zero(z.rows());
  MatrixXd H = MatrixXd::Zero(z.rows(), this->dim_x);

  // C functions
  this->h_dfuns[kind](x.data(), extra_args.data(), h.data());
  this->H_dfuns[kind](x.data(), extra_args.data(), H.data());

  // y is the "loss"
  VectorXd y = z - h;

  // if using eskf
  MatrixXd H_mod = MatrixXd::Zero(x.rows(), P.rows());
  this->H_mod_dfun(x.data(), H_mod.data());
  H = H * H_mod;

  MatrixXd a = ((H * P) * H.transpose() + R).inverse();
  double maha_dist = y.transpose() * (a * y);
  return (maha_dist <= chi2_ppf(maha_thresh, y.rows()));
}

double EKFSym::chi2_ppf(double thres, int dim) {
  return 1.0; // TODO
}

Eigen::MatrixXd EKFSym::rts_smooth(std::vector<Estimate> estimates, bool norm_quats) {
  // Returns rts smoothed results of kalman filter estimates
  // If the kalman state is augmented with old states only the main state is smoothed
  VectorXd xk_n = estimates[estimates.size() - 1].xk1;
  MatrixXd Pk_n = estimates[estimates.size() - 1].Pk1;
  MatrixXd Fk_1 = MatrixXd::Zero(Pk_n.rows(), Pk_n.cols());

  std::vector<VectorXd> states_smoothed = { xk_n };
  std::vector<MatrixXd> covs_smoothed = { Pk_n };
  for (int k = estimates.size() - 2; k >= 0; k--) {
    VectorXd xk1_n = xk_n;
    if (norm_quats) {
      xk1_n.segment<4>(3) /= xk1_n.segment<4>(3).norm();
    }
    MatrixXd Pk1_n = Pk_n;

    VectorXd xk1_k = estimates[k + 1].xk1;
    MatrixXd Pk1_k = estimates[k + 1].Pk1;
    double t2 = estimates[k + 1].t;
    VectorXd xk_k = estimates[k + 1].xk;
    MatrixXd Pk_k = estimates[k + 1].Pk;
    double t1 = estimates[k + 1].t;
    double dt = t2 - t1;
    this->F_dfun(xk_k.data(), dt, Fk_1.data());

    int d1 = this->dim_main;
    int d2 = this->dim_main_err;

    // TODO:
    /*Ck = np.linalg.solve(Pk1_k[:d2, :d2], Fk_1[:d2, :d2].dot(Pk_k[:d2, :d2].T)).T
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
    covs_smoothed.append(Pk_n)*/
  }

  //return np.flipud(np.vstack(states_smoothed)), np.stack(covs_smoothed, 0)[::-1]
  return Fk_1;
}