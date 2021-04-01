#pragma once

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <map>
#include <cmath>

#include <eigen3/Eigen/Dense>

struct EKF {
  std::string name;
  std::vector<int> kinds;
  std::vector<int> feature_kinds;

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

std::vector<const EKF*>& get_ekfs();
const EKF* ekf_lookup(const std::string& ekf_name);

void ekf_register(const EKF* ekf);

#define ekf_init(ekf) \
static void __attribute__((constructor)) do_ekf_init_ ## ekf(void) { \
  ekf_register(&ekf); \
}
