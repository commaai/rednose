#include "ekf_sym.h"

using namespace EKFS;
using namespace Eigen;

EKFSym::EKFSym(
  std::string name,
  MatrixXdr Q,
  VectorXd x_initial,
  MatrixXdr P_initial,
  int dim_main,
  int dim_main_err,
  int N,
  int dim_augment,
  int dim_augment_err,
  std::vector<int> maha_test_kinds,
  std::vector<std::string> global_vars,
  double max_rewind_age
) {
  // TODO: add logger

  this->msckf = N > 0;
  this->N = N;
  this->dim_augment = dim_augment;
  this->dim_augment_err = dim_augment_err;
  this->dim_main = dim_main;
  this->dim_main_err = dim_main_err;

  this->dim_x = x_initial.rows();
  this->dim_err = P_initial.rows();

  assert(dim_main + dim_augment * N == dim_x);
  assert(dim_main_err + dim_augment_err * N == this->dim_err);
  assert(Q.rows() == P_initial.rows() && Q.cols() == P_initial.cols());

  // kinds that should get mahalanobis distance
  // tested for outlier rejection
  this->maha_test_kinds = maha_test_kinds;

  this->global_vars = global_vars;

  // Process nosie
  this->Q = Q;

  this->max_rewind_age = max_rewind_age;
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

std::tuple<VectorXd, MatrixXdr, VectorXd> EKFSym::_update(
  VectorXd x,
  MatrixXdr P,
  int kind,
  VectorXd z,
  MatrixXdr R,
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

std::pair<VectorXd, MatrixXdr> EKFSym::_predict(VectorXd x, MatrixXdr P, double dt) {
  this->predict_dfun(x.data(), P.data(), this->Q.data(), dt);
  return std::pair(x, P);
}

void EKFSym::init_state(VectorXd state, MatrixXdr covs, double filter_time) {
  this->x = state;
  this->P = covs;
  this->filter_time = filter_time;
  this->augment_times = VectorXd::Zero(this->N);
  this->reset_rewind();
}

VectorXd EKFSym::get_state() {
  return this->x;
}

MatrixXdr EKFSym::get_covs() {
  return this->P;
}

double EKFSym::get_filter_time() {
  return this->filter_time;
}

void EKFSym::normalize_state(int slice_start, int slice_end_ex) {
  this->x.block(slice_start, 0, slice_end_ex - slice_start, this->x.cols()).normalize();
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
  MatrixXdr P_reduced = this->P;
  P_reduced.block(d2, 0, d4, P_reduced.cols()) = P_reduced.block(d2 + d4, 0, d4, P_reduced.cols());
  P_reduced.block(0, d2, P_reduced.rows(), d4) = P_reduced.block(0, d2 + d4, P_reduced.rows(), d4);
  P_reduced.conservativeResize(this->dim_err - d4, this->dim_err - d4);
  assert(P_reduced.rows() == this->dim_err - d4 && P_reduced.cols() == this->dim_err - d4);
  MatrixXdr to_mult = MatrixXdr::Zero(this->dim_err, this->dim_err - d4);
  /* TODO to_mult.block(0, 0, to_mult.rows() - d4, to_mult.cols()) = MatrixXdr::Identity(this->dim_err - d4, this->dim_err - d4);
  to_mult(lastN(d4), seq(0, d4 - 1)) = MatrixXdr::Identity(d4, d4);
  this->P = to_mult.dot(P_reduced.dot(to_mult.transpose()));
  this->augment_times(seq(0, last - 1)) = this->augment_times(seq(1, last));
  this->augment_times.tail(1) = this->filter_time;
  assert(this->P.rows() == this->dim_err && this->P.cols() == this->dim_err);*/
}

void EKFSym::rewind(double t, std::deque<Observation>& rewound) {
  // rewind observations until t is after previous observation
  while (this->rewind_t.back() > t) {
    rewound.push_front(this->rewind_obscache.back());
    this->rewind_t.pop_back();
    this->rewind_states.pop_back();
    this->rewind_obscache.pop_back();
  }

  // set the state to the time right before that
  this->filter_time = this->rewind_t.back();
  this->x = this->rewind_states.back().first;
  this->P = this->rewind_states.back().second;

  Eigen::IOFormat fmt(4, 0, ", ", ",", "", "");
}

void EKFSym::checkpoint(Observation obs) {
  // push to rewinder
  this->rewind_t.push_back(this->filter_time);
  this->rewind_states.push_back(std::make_pair(this->x, this->P));
  this->rewind_obscache.push_back(obs);

  // only keep a certain number around
  if (this->rewind_t.size() > REWIND_TO_KEEP) {
    this->rewind_t.pop_front();
    this->rewind_states.pop_front();
    this->rewind_obscache.pop_front();
  }
}

void EKFSym::_predict(double t) {
  // initialize time
  if (std::isnan(this->filter_time)) {
    this->filter_time = t;
  }

  // predict
  double dt = t - this->filter_time;
  assert(dt >= 0.0);
  std::pair<VectorXd, MatrixXdr> res = this->_predict(this->x, this->P, dt);
  this->x = res.first;
  this->P = res.second;
  this->filter_time = t;
}

bool EKFSym::predict_and_update_batch(
  Estimate* res,
  double t,
  int kind,
  std::vector<Map<VectorXd> > z_map,
  std::vector<Map<MatrixXdr> > R_map,
  std::vector<std::vector<double>> extra_args,
  bool augment)
{
  std::vector<VectorXd> z;
  for (Map<VectorXd> zi : z_map) {
    z.push_back(zi);
  }
  std::vector<MatrixXdr> R;
  for (Map<MatrixXdr> Ri : R_map) {
    R.push_back(Ri);
  }

  // TODO handle rewinding at this level

  std::deque<Observation> rewound;
  if (!std::isnan(this->filter_time) && t < this->filter_time) {
    if (this->rewind_t.empty() || t < this->rewind_t.front() || t < this->rewind_t.back() - this->max_rewind_age) {
      std::cout << "C observation too old at " << t << " with filter at " << this->filter_time << ", ignoring" << std::endl;
      // TODO self.logger.error("observation too old at %.3f with filter at %.3f, ignoring" % (t, self.filter_time))
      return false;
    }
    this->rewind(t, rewound);
  }

  this->_predict_and_update_batch(res, { t, kind, z, R, extra_args }, augment);

  // optional fast forward
  while (!rewound.empty()) {
    this->_predict_and_update_batch(NULL, rewound.front(), false);
    rewound.pop_front();
  }

  return true;
}

void EKFSym::_predict_and_update_batch(Estimate* res, Observation obs, bool augment) {
  assert(obs.z.size() == obs.R.size());

  this->_predict(obs.t);

  VectorXd xk_km1 = this->x;
  MatrixXdr Pk_km1 = this->P;

  // update batch
  std::vector<VectorXd> y;
  for (int i = 0; i < obs.z.size(); i++) {
    assert(obs.z[i].rows() == obs.R[i].rows());
    assert(obs.z[i].rows() == obs.R[i].cols());
    // update
    std::tuple<VectorXd, MatrixXdr, VectorXd> res =
      this->_update(this->x, this->P, obs.kind, obs.z[i], obs.R[i], obs.extra_args[i]);
    this->x = std::get<0>(res);
    this->P = std::get<1>(res);
    y.push_back(std::get<2>(res));
  }

  VectorXd xk_k = this->x;
  MatrixXdr Pk_k = this->P;

  assert(!augment); // TODO
  if (augment) {
    this->augment();
  }

  this->checkpoint(obs);

  if (res != NULL) {
    res->xk1 = xk_km1;
    res->xk = xk_k;
    res->Pk1 = Pk_km1;
    res->Pk = Pk_k;
    res->t = obs.t;
    res->kind = obs.kind;
    res->y = y;
    res->z = obs.z;
    res->extra_args = obs.extra_args;
  }
}

bool EKFSym::maha_test(VectorXd x, MatrixXdr P, int kind, VectorXd z, MatrixXdr R, std::vector<double> extra_args, double maha_thresh) {
  // init vars
  VectorXd h = VectorXd::Zero(z.rows());
  MatrixXdr H = MatrixXdr::Zero(z.rows(), this->dim_x);

  // C functions
  this->h_dfuns[kind](x.data(), extra_args.data(), h.data());
  this->H_dfuns[kind](x.data(), extra_args.data(), H.data());

  // y is the "loss"
  VectorXd y = z - h;

  // if using eskf
  MatrixXdr H_mod = MatrixXdr::Zero(x.rows(), P.rows());
  this->H_mod_dfun(x.data(), H_mod.data());
  H = H * H_mod;

  MatrixXdr a = ((H * P) * H.transpose() + R).inverse();
  double maha_dist = y.transpose() * (a * y);
  return (maha_dist <= chi2_ppf(maha_thresh, y.rows()));
}

double EKFSym::chi2_ppf(double thres, int dim) {
  return 1.0; // TODO
}

MatrixXdr EKFSym::rts_smooth(std::vector<Estimate> estimates, bool norm_quats) {
  // Returns rts smoothed results of kalman filter estimates
  // If the kalman state is augmented with old states only the main state is smoothed
  VectorXd xk_n = estimates[estimates.size() - 1].xk1;
  MatrixXdr Pk_n = estimates[estimates.size() - 1].Pk1;
  MatrixXdr Fk_1 = MatrixXdr::Zero(Pk_n.rows(), Pk_n.cols());

  std::vector<VectorXd> states_smoothed = { xk_n };
  std::vector<MatrixXdr> covs_smoothed = { Pk_n };
  for (int k = estimates.size() - 2; k >= 0; k--) {
    VectorXd xk1_n = xk_n;
    if (norm_quats) {
      xk1_n.segment<4>(3) /= xk1_n.segment<4>(3).norm();
    }
    MatrixXdr Pk1_n = Pk_n;

    VectorXd xk1_k = estimates[k + 1].xk1;
    MatrixXdr Pk1_k = estimates[k + 1].Pk1;
    double t2 = estimates[k + 1].t;
    VectorXd xk_k = estimates[k + 1].xk;
    MatrixXdr Pk_k = estimates[k + 1].Pk;
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