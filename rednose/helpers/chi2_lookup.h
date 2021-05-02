#pragma once

#include <fstream>
#include <stdint.h>

#include <eigen3/Eigen/Dense>
#include "npy_parser.h"

double chi2_ppf(double p, uint8_t dim);
