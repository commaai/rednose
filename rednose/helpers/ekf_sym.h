#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <map>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "/home/batman/openpilot/selfdrive/locationd/models/generated/live.h"

namespace EKFS {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;

typedef struct Estimate {
  Eigen::VectorXd xk1;
  Eigen::VectorXd xk;
  MatrixXdr Pk1;
  MatrixXdr Pk;
  double t;
  int kind;
  std::vector<Eigen::VectorXd> y;
  std::vector<Eigen::VectorXd> z;
  std::vector<std::vector<double>> extra_args;
} Estimate;

class EKFSym {
public:
  EKFSym(
    std::string name,
    Eigen::VectorXd Q,
    Eigen::VectorXd x_initial,
    MatrixXdr P_initial,
    int dim_main,
    int dim_main_err,
    int N = 0,
    int dim_augment = 0,
    int dim_augment_err = 0,
    std::vector<int> maha_test_kinds = std::vector<int>(),
    std::vector<std::string> global_vars = std::vector<std::string>(),
    double max_rewind_age = 1.0
  );

  void init_state(
    Eigen::VectorXd state,
    MatrixXdr covs,
    double filter_time
  );

  void reset_rewind();
  void augment();
  Eigen::VectorXd state();
  MatrixXdr covs();

  std::vector<int> rewind(double t);
  void checkpoint(int obs);
  void _predict(double t);
  Estimate predict_and_update_batch(
    double t,
    int kind,
    std::vector<Eigen::VectorXd> z,
    std::vector<MatrixXdr> R,
    std::vector<std::vector<double>> extra_args = std::vector<std::vector<double>>(),
    bool augment = false
  );
  Estimate _predict_and_update_batch(
    double t,
    int kind,
    std::vector<Eigen::VectorXd> z,
    std::vector<MatrixXdr> R,
    std::vector<std::vector<double>> extra_args,
    bool augment
  );

  bool maha_test(
    Eigen::VectorXd x,
    MatrixXdr P,
    int kind,
    Eigen::VectorXd z,
    MatrixXdr R,
    std::vector<double> extra_args = std::vector<double>(),
    double maha_thresh = 0.95
  );

  MatrixXdr rts_smooth(std::vector<Estimate> estimates, bool norm_quats = false);

  std::pair<Eigen::VectorXd, MatrixXdr> _predict(
    Eigen::VectorXd x,
    MatrixXdr P,
    double dt
  );
  std::tuple<Eigen::VectorXd, MatrixXdr, Eigen::VectorXd> _update(
    Eigen::VectorXd x,
    MatrixXdr P,
    int kind,
    Eigen::VectorXd z,
    MatrixXdr R,
    std::vector<double> extra_args = std::vector<double>()
  );

private:
  static double chi2_ppf(double thres, int dim);

  Eigen::VectorXd x;  // state
  MatrixXdr P;  // covs

  bool msckf;
  int N;
  int dim_augment;
  int dim_augment_err;
  int dim_main;
  int dim_main_err;

  // state
  int dim_x;
  int dim_err;

  double filter_time;

  std::vector<int> maha_test_kinds;

  std::vector<std::string> global_vars;

  // process noise
  Eigen::VectorXd Q;

  // rewind stuff
  int max_rewind_age;
  std::vector<double> rewind_t;
  std::vector<int> rewind_states;
  std::vector<int> rewind_obscache;

  Eigen::VectorXd augment_times;

  std::vector<int> feature_track_kinds;

  // dynamic functions
  void (*f_dfun)(double *, double, double *);
  void (*F_dfun)(double *, double, double *);
  void (*err_dfun)(double *, double *, double *);
  void (*inv_err_dfun)(double *, double *, double *);
  void (*H_mod_dfun)(double *, double *);
  void (*predict_dfun)(double *, double *, double *, double);
  std::unordered_map<int, void (*)(double *, double *, double *)> h_dfuns = {};
  std::unordered_map<int, void (*)(double *, double *, double *)> H_dfuns = {};
  std::unordered_map<int, void (*)(double *, double *, double *)> He_dfuns = {};
  std::unordered_map<int, void (*)(double *, double *, double *, double *, double *)> update_dfuns = {};
  std::unordered_map<std::string, void (*)(double)> set_global_dfuns = {};
};

}