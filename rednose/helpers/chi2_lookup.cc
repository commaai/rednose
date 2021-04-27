#include "chi2_lookup.h"

void gen_chi2_ppf_lookup(int max_dim) {// TODO max_dim default to 200
    MatrixXd table = MatrixXd::Zero(max_dim, 98);
    range = range from 0.01 to 0.99 in steps of 0.01
    for (int idim = 0; idim < max_dim; idim++) {
        table[idim] = chi2.ppf(range, idim);
    }
    // save to file
}

double chi2_ppd(type p, type dim) {
    MatrixXd table = // load from file
    double result;
    return result;
}

int main(int argc, char *argv[]) {
    gen_chi2_ppf_lookup();
}

// Python underneath
def gen_chi2_ppf_lookup(max_dim=200):
  from scipy.stats import chi2
  table = np.zeros((max_dim, 98))
  for dim in range(1, max_dim):
    table[dim] = chi2.ppf(np.arange(.01, .99, .01), dim)

  np.save('chi2_lookup_table', table)


def chi2_ppf(p, dim):
  table = np.load(os.path.dirname(os.path.realpath(__file__)) + '/chi2_lookup_table.npy')
  result = np.interp(p, np.arange(.01, .99, .01), table[dim])
  return result


if __name__ == "__main__":
  gen_chi2_ppf_lookup()
