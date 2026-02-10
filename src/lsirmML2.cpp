#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]


// helper: maximum over an IntegerMatrix ignoring NAs
int max_data(const IntegerMatrix& data) {
  int maxval = INT_MIN;
  for (int n = 0; n < data.nrow(); n++) {
    for (int i = 0; i < data.ncol(); i++) {
      int v = data(n, i);
      if (v != NA_INTEGER && v > maxval) {
        maxval = v;
      }
    }
  }
  return maxval;
}
////////////////////////////////////////////////////////////////////////////////



// [[Rcpp::export]]
NumericVector P_lsirm_cpp(NumericMatrix theta,
                          double slope,
                          NumericVector threshold,
                          NumericVector coord) {
  int n = theta.nrow();
  int m = theta.ncol();
  int c = threshold.size();

  NumericMatrix result(n, c + 1);

  for (int i = 0; i < n; i++) {
    NumericVector c_prob(c + 1, 1.0);
    double sumsq = 0.0;

    for (int j = 1; j < m; j++) {
      double diff = theta(i, j) - coord[j-1];
      sumsq += diff * diff;
    }
    double dist = std::sqrt(sumsq);

    for (int k = 0; k < c; k++) {
      c_prob[k + 1] = 1.0 / (1.0 + std::exp( - (slope * theta(i, 0) - threshold[k] - dist)));
      result(i, k) = c_prob[k] - c_prob[k + 1];
    }
    result(i, c) = c_prob[c];
  }

  return result;
}



// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//

/*** R
P_lsirm_cpp(matrix(c(1,0,0,
                     0,1,1,
                     0,.5,.5), nrow=3, byrow = T), 1, c(-.2,0,.2), c(1,1))
*/


// [[Rcpp::export]]
Rcpp::NumericMatrix llik_cpp(const Rcpp::IntegerVector& data,
                             const Rcpp::NumericMatrix& grid,
                             double slope,
                             const Rcpp::NumericVector& threshold,
                             const Rcpp::NumericVector& coord,
                             int c) {

  int N = data.size();     // persons
  int M = grid.nrow();     // grid points
  int d = grid.ncol();     // dimension

  Rcpp::NumericMatrix out(N, M);

  // ---- loop over grid points ----
  for (int m = 0; m < M; m++) {

    // distance term
    double sumsq = 0.0;
    for (int j = 1; j < d; j++) {
      double diff = grid(m, j) - coord[j - 1];
      sumsq += diff * diff;
    }
    double dist = std::sqrt(sumsq);

    // cumulative probs
    Rcpp::NumericVector c_prob(c + 1, 1.0);
    for (int k = 0; k < c - 1; k++) {
      c_prob[k + 1] =
        1.0 / (1.0 + std::exp(-(slope * grid(m, 0) - threshold[k] - dist)));
    }

    // category probs
    Rcpp::NumericVector prob(c + 1);
    for (int k = 0; k < c; k++) {
      prob[k] = c_prob[k] - c_prob[k + 1];
    }
    prob[c] = c_prob[c];

    // likelihood
    for (int n = 0; n < N; n++) {
      int y = data[n];
      if (y == NA_INTEGER) {
        out(n, m) = 0.0;
      } else {
        out(n, m) = std::log(prob[y]);
      }
    }
  }

  return out;
}



/*** R
llik_cpp(c(0,1,NA),
         matrix(c(1,0,0,
                  0,1,1,
                  0,0,0), nrow=3, byrow = T),
         1, c(-.2,0,.2), c(1,1),
         3)
*/


#include <omp.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List Estep_cpp(const Rcpp::IntegerMatrix& data,
                     const Rcpp::NumericMatrix& item,
                     const Rcpp::NumericMatrix& coord,
                     const Rcpp::NumericMatrix& grid,
                     const Rcpp::NumericVector& prior,
                     const Rcpp::IntegerVector& categories) {
  int N = data.nrow();
  int I = data.ncol();
  int M = grid.nrow();

  // log-likelihood accumulator
  std::vector<double> posterior_loglik_vec(N * M, 0.0);

  // accumulate over items
  for (int i = 0; i < I; i++) {
    IntegerVector data_col = data(_, i);
    NumericVector item_row = item(i, _);
    NumericVector coord_row = coord(i, _);
    int c = categories[i];
    double slope = item_row[0];  // C++ is 0-based
    Rcpp::NumericVector threshold =
      item_row[Rcpp::Range(1, item_row.size() - 1)];

    NumericMatrix ll = llik_cpp(data_col, grid, slope, threshold, coord_row, c); // N x M (R object)

#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      for (int m = 0; m < M; m++) {
        posterior_loglik_vec[n * M + m] += ll(n, m);
      }
    }
  }

  // precompute log(prior)
  std::vector<double> posterior_vec(N * M, 0.0);
  std::vector<double> freq_vec(M, 0.0);

  std::vector<double> log_prior_vec(M);
  for (int m = 0; m < M; ++m) log_prior_vec[m] = std::log(prior[m]);

#pragma omp parallel for
  for (int n = 0; n < N; n++) {
    double rowsum = 0.0;
    for (int m = 0; m < M; m++) {
      double val = std::exp(posterior_loglik_vec[n * M + m] + log_prior_vec[m]);
      posterior_vec[n * M + m] = val;
      rowsum += val;
    }
    if (rowsum > 0) {
      double inv = 1.0 / rowsum;
      for (int m = 0; m < M; m++) posterior_vec[n * M + m] *= inv;
    }
  }

  // expected response frequencies
  int categ = max_data(data) + 1;
  std::vector<double> e_response_vec(I * M * categ, 0.0);

#pragma omp parallel
{
  std::vector<double> e_private(I * M * categ, 0.0);

#pragma omp for nowait
  for (int n = 0; n < N; n++) {
    for (int i = 0; i < I; i++) {
      int val = data(n, i);
      if (val == NA_INTEGER) continue;
      for (int m = 0; m < M; m++) {
        int idx = i + I * m + I * M * val;
        e_private[idx] += posterior_vec[n * M + m];
      }
    }
  }

#pragma omp critical
{
  for (size_t idx = 0; idx < e_response_vec.size(); ++idx) {
    e_response_vec[idx] += e_private[idx];
  }
}
}

// freq = column sums
#pragma omp parallel for
for (int m = 0; m < M; m++) {
  double s = 0.0;
  for (int n = 0; n < N; n++) s += posterior_vec[n * M + m];
  freq_vec[m] = s;
}

Rcpp::NumericVector e_response_r(e_response_vec.begin(), e_response_vec.end());
Rcpp::NumericVector freq_r(freq_vec.begin(), freq_vec.end());
Rcpp::NumericMatrix posterior_r(N, M);
for (int n = 0; n < N; n++) {
  for (int m = 0; m < M; m++) {
    posterior_r(n, m) = posterior_vec[n * M + m];
  }
}

// logL
double total_logL = 0.0;
#pragma omp parallel for reduction(+:total_logL)
for (int n = 0; n < N; n++) {
  for (int m = 0; m < M; m++) {
    double val = posterior_vec[n * M + m];
    if (val > 0) {
      total_logL += posterior_loglik_vec[n * M + m] * val;
      total_logL += val * log_prior_vec[m];
      total_logL -= val * std::log(val);
    }
  }
}


// Ak
double total_freq = std::accumulate(freq_r.begin(), freq_r.end(), 0.0);
Rcpp::NumericVector Ak(M);
if (total_freq > 0) {
  for (int m = 0; m < M; m++) Ak[m] = freq_r[m] / total_freq;
}



return Rcpp::List::create(
  Rcpp::_["posterior"]  = posterior_r,
  Rcpp::_["freq"]       = freq_r,
  Rcpp::_["e.response"] = e_response_r,
  Rcpp::_["grid"]       = grid,
  Rcpp::_["prior"]      = prior,
  Rcpp::_["logL"]       = total_logL,
  Rcpp::_["Ak"]         = Ak
);
}
