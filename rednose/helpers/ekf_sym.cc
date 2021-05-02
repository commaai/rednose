#include "ekf_sym.h"

using namespace EKFS;
using namespace Eigen;

EKFSym::EKFSym(std::string name, Map<MatrixXdr> Q, Map<VectorXd> x_initial, Map<MatrixXdr> P_initial, int dim_main,
    int dim_main_err, int N, int dim_augment, int dim_augment_err, std::vector<int> maha_test_kinds,
    std::vector<std::string> global_vars, double max_rewind_age)
{
  // TODO: add logger

  this->ekf = ekf_lookup(name);
  assert(this->ekf);

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
}

void EKFSym::init_state(Map<VectorXd> state, Map<MatrixXdr> covs, double filter_time) {
  this->x = state;
  this->P = covs;
  this->filter_time = filter_time;
  this->augment_times = VectorXd::Zero(this->N);
  this->reset_rewind();
}

VectorXd EKFSym::state() {
  return this->x;
}

MatrixXdr EKFSym::covs() {
  return this->P;
}

void EKFSym::set_filter_time(double t) {
  this->filter_time = t;
}

double EKFSym::get_filter_time() {
  return this->filter_time;
}

void EKFSym::normalize_state(int slice_start, int slice_end_ex) {
  this->x.block(slice_start, 0, slice_end_ex - slice_start, this->x.cols()).normalize();
}

void EKFSym::set_global(std::string global_var, double val) {
  this->ekf->sets.at(global_var)(val);
}

std::optional<Estimate> EKFSym::predict_and_update_batch(double t, int kind, std::vector<Map<VectorXd>> z_map,
    std::vector<Map<MatrixXdr>> R_map, std::vector<std::vector<double>> extra_args, bool augment)
{
  // TODO handle rewinding at this level

  std::deque<Observation> rewound;
  if (!std::isnan(this->filter_time) && t < this->filter_time) {
    if (this->rewind_t.empty() || t < this->rewind_t.front() || t < this->rewind_t.back() - this->max_rewind_age) {
      std::cout << "observation too old at " << t << " with filter at " << this->filter_time << ", ignoring" << std::endl;
      return std::nullopt;
    }
    rewound = this->rewind(t);
  }

  Observation obs;
  obs.t = t;
  obs.kind = kind;
  obs.extra_args = extra_args;
  for (Map<VectorXd> zi : z_map) {
    obs.z.push_back(zi);
  }
  for (Map<MatrixXdr> Ri : R_map) {
    obs.R.push_back(Ri);
  }

  std::optional<Estimate> res = std::make_optional(this->predict_and_update_batch(obs, augment));

  // optional fast forward
  while (!rewound.empty()) {
    this->predict_and_update_batch(rewound.front(), false);
    rewound.pop_front();
  }

  return res;
}

void EKFSym::reset_rewind() {
  this->rewind_obscache.clear();
  this->rewind_t.clear();
  this->rewind_states.clear();
}

std::deque<Observation> EKFSym::rewind(double t) {
  std::deque<Observation> rewound;

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

  return rewound;
}

void EKFSym::checkpoint(Observation& obs) {
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

Estimate EKFSym::predict_and_update_batch(Observation& obs, bool augment) {
  assert(obs.z.size() == obs.R.size());

  this->predict(obs.t);

  Estimate res;
  res.t = obs.t;
  res.kind = obs.kind;
  res.z = obs.z;
  res.extra_args = obs.extra_args;
  res.xk1 = this->x;
  res.Pk1 = this->P;

  // update batch
  std::vector<VectorXd> y;
  for (int i = 0; i < obs.z.size(); i++) {
    assert(obs.z[i].rows() == obs.R[i].rows());
    assert(obs.z[i].rows() == obs.R[i].cols());

    // update state
    y.push_back(this->update(obs.kind, obs.z[i], obs.R[i], obs.extra_args[i]));
  }

  res.xk = this->x;
  res.Pk = this->P;
  res.y = y;

  assert(!augment); // TODO
  // if (augment) {
  //   this->augment();
  // }

  this->checkpoint(obs);

  return res;
}

void EKFSym::predict(double t) {
  // initialize time
  if (std::isnan(this->filter_time)) {
    this->filter_time = t;
  }

  // predict
  double dt = t - this->filter_time;
  assert(dt >= 0.0);

  this->ekf->predict(this->x.data(), this->P.data(), this->Q.data(), dt);
  this->filter_time = t;
}

VectorXd EKFSym::update(int kind, VectorXd z, MatrixXdr R, std::vector<double> extra_args) {
  this->ekf->updates.at(kind)(this->x.data(), this->P.data(), z.data(), R.data(), extra_args.data());

  if (this->msckf && std::find(this->feature_track_kinds.begin(), this->feature_track_kinds.end(), kind) != this->feature_track_kinds.end()) {
    return z.head(z.rows() - extra_args.size());
  }
  return z;
}

void EKFSym::augment() {
    // TODO this is not a generalized way of doing this and implies that the
    // augmented states are simply the first (dim_augment_state) elements of
    // the main state.

    // TODO remove commented python code after testing pass
    assert(this->msckf);

    int d1 = this->dim_main;
    int d2 = this->dim_main_err;
    int d3 = this->dim_augment;
    int d4 = this->dim_augment_err;

    // push through augmented states
    // this->x[d1:-d3] = this->x[d1 + d3:];
    int s = this->dim_x - d3 - d1;
    this->x.segment<s>(d1) = this->x.segment<s>(d1+d3);

    // this->x[-d3:] = this->x[:d3];
    this->x.tail(d3) = this->x.head(d3);

    assert(this->x.rows() == this->dim_x);
    assert(this->x.cols() == 1);

    // push through augmented covs
    assert(this->P.rows() == this->dim_err);
    assert(this->P.cols() == this->dim_err);

    // Build P_reduced by selectively copying P's contents
    // Skip rows from d2 to before (d2+d4), do same with columns
    int red_side = this->P.rows() - d4;
    MatrixXd P_reduced = MatrixXd::Zero(red_side, red_side);
    MatrixXd to_mult   = MatrixXd::Zero(this->dim_err, red_side);
    int irow,        icol;
    int irow_offset, icol_offset;
    // TODO evaluate whether to convert this to a series of block operations,
    // although this may be faster since it involves just looping red_side
    // times once. The sacrifice is maintainability though.
    for (int i = 0; i < red_side; i++) {
        // Build P_reduced
        icol = i % this->dim_err;
        irow = i / this->dim_err;

        icol_offset = (icol >= d2) ? d4 : 0;
        irow_offset = (irow >= d2) ? d4 : 0;

        P_reduced(irow,icol) = P(irow+irow_offset,icol+icol_offset);

        // Build to_mult
        // to_mult[:-d4, :] = np.eye(self.dim_err - d4)
        // The submatrix comprised of all rows except last d4 is set to an identity
        // matrix
        to_mult(i, i) = 1;
        // to_mult[-d4:, :d4] = np.eye(d4)
        // The submatrix comprised by the last d4 rows and therein all columns up
        // to the d4-th column is set to an identity matrix
        if (i < d4) to_mult(red_side+i, i) = 1;
    }

    // TODO aren't we certain enough that the side of P_reduced is red_side?
    // Are these asserts really required?
    assert(P_reduced.rows() == red_side);
    assert(P_reduced.cols() == red_side);

    this->P = to_mult.dot(P_reduced.dot(to_mult.transpose()));

    // P = to_mult dot (P_reduced dot transpose(to_mult))
    // to_mult             is dim_err  x red_side
    // P_reduced           is red_side x red_side
    // to_mult.transpose() is red_side x dim_err
    // P_reduced . (above) is red_side x dim_err
    // to_mult . (above)   is dim_err  x dim_err

    // TODO aren't we certain enough that the side of P is dim_err?
    // Are these asserts really required?
    assert(this->P.rows() == this->dim_err);
    assert(this->P.cols() == this->dim_err);

    this->augment_times.head(this->N-1) = this->augment_times.tail(this->N-1);
    this->augment_times.tail(1) = this->filter_time;
}

VectorXd EKFSym::get_augment_times() {
    return this->augment_times;
}

RTSSmoothResult EKFSym::rts_smooth(std::vector<Estimate> *estimates, bool norm_quats) {
    /*
     * Returns rts smoothed results of kalman filter estimates
     * If the kalman state is augmented with old states only the main state is
     * smoothed
     */

    RTSSmoothResult retobj;

    VectorXd xk_n  = estimates->back().xk1;
    MatrixXdr Pk_n = estimates->back().Pk1;

    MatrixXdr Fk_1 = MatrixXd::Zero(Pk_n.rows(), Pk_n.cols());

    retobj.states_smoothed.push_back(xk_n);
    retobj.covs_smoothed.push_back(Pk_n);

    VectorXd xk1_n, xk1_k, xk_k;
    MatrixXdr Pk1_n, Pk1_k, Pk_k;

    double dt, t2, t1;

    int d1 = this->dim_main;
    int d2 = this->dim_main_err;

    VectorXd delta_x, x_new;
    MatrixXd Ck;
    for (int k = (int) estimates->size() - 2; k >= -1; i--) {
        xk1_n = xk_n;
        Pk1_n = Pk_n;

        if (norm_quats) {
            xk1_n.segment<7-3>(3) /= xk1_n.segment<7-3>(3).norm();
        }

        xk1_k = estimates[k+1].xk1;
        Pk1_k = estimates[k+1].Pk1;
        t2    = estimates[k+1].t;

        xk_k  = estimates[k].xk;
        Pk_k  = estimates[k].Pk;
        t1    = estimates[k].t;

        dt = t2 - t1;

        this->ekf->F(xk_k, dt, Fk_1);
        // TODO find out: is this how this works?

        Ck = Pk1_k.block<d2,d2>(0,0).ldlt().solve(
                Fk_1.block<d2,d2>.block(0,0).dot(
                    Pk_k.block<d2,d2>(0,0).transpose()
                ).transpose());
        xk_n = xk_k;
        delta_x = VectorXd::Zero(Pk_n.rows());
        this->ekf->inv_err_function(xk1_k, xk1_n, delta_x); // TODO: see prev TODO
        delta_x.head(d2) = Ck.dot(delta_x.head(d2));
        x_new = VectorXd::Zero(xk_n.rows());
        this->ekf->err_function(xk_k, delta_x, x_new); // TODO see prev TODO
        xk_n.head(d1) = x_new.head(d1);
        Pk_n = Pk_k;
        Pk_n.block<d2,d2>(0,0) =   Pk_k.block<d2,d2>(0,0)
                                 + Ck.dot(  Pk1_n.block<d2,d2>(0,0)
                                          - Pk1_k.block<d2,d2>(0,0).dot(Ck.transpose()));
        retobj.states_smoothed.push_back(xk_n);
        retobj.covs_smoothed.push_back(Pk_n);
    }

    std::reverse(retobj.states_smoothed.begin(), retobj.states_smoothed.end());
    std::reverse(retobj.covs_smoothed.begin(),   retobj.covs_smoothed.end());

    return retobj;
}

// TODO figure out what to do with the following python comment on the python
// maha_test function implementation header:
// # pylint: disable=dangerous-default-value
bool maha_test(VectorXd x, MatrixXdr P, int kind, VectorXd z, MatrixXdr R, std::vector<double> extra_args, double maha_thresh = 0.95)
{
    // init vars
    VectorXd z1(z.data(), z.size());
    VectorXd h = VectorXd::Zero(z1.size());
    MatrixXd H = MatrixXd::Zero(z1.size(), this->dim_x);

    // C functions
    this->hs[kind](x, extra_args, &h);
    this->Hs[kind](x, extra_args, &H);

    // y is the "loss"
    y = z - h;

    // if using eskf
    MatrixXd H_mod = MatrixXd::Zero(x.rows(), P.rows());
    this->H_mod(x, H_mod);
    H = H.dot(H_mod);

    MatrixXd a = H.dot(P).dot(H.transpose()) + R;
    a = a.inverse();
    maha_dist = y.transpose().dot(a.dot(y));

    return (maha_dist <= chi2_ppf(maha_thresh, y.rows()));
}
