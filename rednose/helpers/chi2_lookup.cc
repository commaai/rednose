#include "chi2_lookup.h"

#define CLIP(bottom,top,x) ((x < bottom) ? bottom : ((x > top) ? top : x))

#define P_MIN 0.01
#define P_MAX 0.99
#define P_IND_MAX 97
#define P_IND_MIN  0
#define P_CMP_TH  0.01

double chi2_ppf(double p, uint8_t dim)
{
    Eigen::VectorXd chi2_table_row;
    std::ifstream fobj;
    // TODO determine this path correctly
    fobj.open("/tmp/chi2_lookup_table.npy",
              std::ifstream::in | std::ifstream::binary);
    populate_lookup_row(&fobj, &chi2_table_row, dim);
    fobj.close();
    // Assert p is between 0.01 and 0.99, inclusive
    // ((p >= 0.01) && (p <= 0.99))
    // Find closest upper and lower boundaries of p in the range from 0.01 to
    // 0.99 and stride 0.01
    double p_fact = (double) 100*p; // same as div. by 0.01
    double p_low  = std::floor (p_fact)/100;
    double p_high = std::ceil  (p_fact)/100;

    if (std::fabs(p_low - p_high) < P_CMP_TH) {
        return chi2_table_row[CLIP(P_IND_MIN, P_IND_MAX, ((uint8_t) p_fact) - 1)];
    } else {
        if      (std::fabs(p_high - P_MAX) < P_CMP_TH)
            p_low = p_high - 0.01;
        else if (std::fabs(p_low - P_MIN) < P_CMP_TH)
            p_high = p_low + 0.01;

        double f_low  = chi2_table_row[CLIP(P_IND_MIN, P_IND_MAX, 100*((uint8_t) p_low ) - 1)];
        double f_high = chi2_table_row[CLIP(P_IND_MIN, P_IND_MAX, 100*((uint8_t) p_high) - 1)];

        return f_low + (p - p_low)/(p_high - p_low) * (f_high - f_low);
    }
}
