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
    // ^ make elements from d1 to (element d3 elements far from the end)
    // Traversal must be in reverse to prevent overwriting data
    // Start at end-d3, work way back to d1
    int x_size = this->x.size();
    for (int ix = (x_size-1)-d3; ix >= d1; ix--) {
        this->x[ix] = this->x[ix+d3];
    }

    // this->x[-d3:] = this->x[:d3];
    // ^ last d3 elements = first d3 elements
    for (int ix = 0; ix < d3; ix++) {
        this->x[x_size-d3+ix] = this->x[ix];
    }

    assert(this->x.rows() == this->dim_x);
    assert(this->x.cols() == 1);

    // push through augmented covs
    assert(this->P.rows() == this->dim_err);
    assert(this->P.cols() == this->dim_err);

    // Build reduced P
    MatrixXdr P_reduced;
    // Delete rows from d2 to before (d2+d4), do same with columns
    int red_side = this->P.rows() - this->dim_err;
    int irow,        icol;
    int irow_offset, icol_offset;
    // Skip elements
    for (int i = 0; i < red_side; i++) {
        icol = i % this->dim_err;
        irow = i / this->dim_err;

        icol_offset = (icol >= d2) ? d4 : 0;
        irow_offset = (irow >= d2) ? d4 : 0;

        P_reduced(irow,icol) = P(irow+irow_offset,icol+icol_offset);
    }

    assert(P_reduced.rows() == red_side);
    assert(P_reduced.cols() == red_side);

    MatrixXd to_mult = MatrixXd::Zero(this->dim_err, this->dim_err - d4);
    // --- C++ code above | Python code below ---

    // to_mult = np.zeros((self.dim_err, self.dim_err - d4))
    // to_mult[:-d4, :] = np.eye(self.dim_err - d4)
    // to_mult[-d4:, :d4] = np.eye(d4)
    // self.P = to_mult.dot(P_reduced.dot(to_mult.T))
    // self.augment_times = self.augment_times[1:]
    // self.augment_times.append(self.filter_time)

    // --- Python code above | C++ code below ---

    assert(this->P.rows() == this->dim_err);
    assert(this->P.cols() == this->dim_err);
}
