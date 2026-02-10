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

// [[Rcpp::export]]
Rcpp::IntegerVector max_data_cols(const Rcpp::IntegerMatrix& data) {
  int nrow = data.nrow();
  int ncol = data.ncol();

  Rcpp::IntegerVector maxval(ncol, NA_INTEGER);

  for (int j = 0; j < ncol; j++) {
    int col_max = INT_MIN;
    bool any_non_na = false;

    for (int i = 0; i < nrow; i++) {
      int v = data(i, j);
      if (v != NA_INTEGER) {
        if (!any_non_na || v > col_max) {
          col_max = v;
          any_non_na = true;
        }
      }
    }

    if (any_non_na) {
      maxval[j] = col_max;
    }
  }

  return maxval;
}

int max_vec(const Rcpp::IntegerVector& x) {
  int maxval = INT_MIN;
  bool any = false;

  for (int i = 0; i < x.size(); i++) {
    int v = x[i];
    if (v != NA_INTEGER) {
      if (!any || v > maxval) {
        maxval = v;
        any = true;
      }
    }
  }

  return any ? maxval : NA_INTEGER;
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix make_band_matrix(int K) {
  int nrow = K + 1;
  int ncol = K + 2;

  Rcpp::IntegerMatrix mat(nrow, ncol);

  for (int i = 0; i < nrow; i++) {
    mat(i, i)     = -1;
    mat(i, i + 1) = -1;
  }

  return mat;
}

////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
Rcpp::NumericMatrix llik_cpp0(const Rcpp::IntegerVector& data,
                             const Rcpp::NumericMatrix& grid,
                             const Rcpp::NumericVector& item) {
  int N = data.size();   // persons
  int M = grid.nrow();   // grid points
  int d = grid.ncol();   // dimension

  Rcpp::NumericMatrix out(N, M);

  // loop over grid points
  for (int m = 0; m < M; m++) {
    // compute p(m | item) once
    double first = item[0] * grid(m, 0) - item[1];
    double sumsq = 0.0;
    for (int j = 1; j < d; j++) {
      double diff = grid(m, j) - item[j+1];
      sumsq += diff * diff;
    }
    double eta = first - std::sqrt(sumsq);
    double p = 1.0 / (1.0 + std::exp(-eta));

    // precompute logs
    double logp   = std::log(p);
    double log1mp = std::log1p(-p);  // safer near p≈0

    // fill this column for all persons
    for (int n = 0; n < N; n++) {
      int y = data[n];
      if (y == NA_INTEGER) {
        out(n, m) = 0.0;            // missing → log(1)
      } else if (y == 0) {
        out(n, m) = log1mp;         // response 0
      } else {
        out(n, m) = logp;           // response 1
      }
    }
  }

  return out;
}

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
    Rcpp::NumericVector c_prob(c + 1);
    c_prob[0] = 1.0;
    c_prob[c] = 0.0;

    for (int k = 0; k < c - 1; k++) {
      double eta = slope * grid(m, 0) - threshold[k] - dist;
      c_prob[k + 1] = 1.0 / (1.0 + std::exp(-eta));
    }


    // category probs
    Rcpp::NumericVector prob(c);
    for (int k = 0; k < c; k++) {
      prob[k] = std::max(c_prob[k] - c_prob[k + 1], 1e-12);
    }

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
llik_cpp0(c(0,1,NA),
          matrix(c(1,0,0,
                   0,1,1,
                   0,0,0), nrow=3, byrow = T),
          c(1,0,1,1))

llik_cpp(c(0,1,NA),
         matrix(c(1,0,0,
                  0,1,1,
                  0,0,0), nrow=3, byrow = T),
         1, c(0), c(1,1), 2)

llik_cpp(c(3,1,NA),
         matrix(c(1,0,0,
                  0,1,1,
                  0,0,0), nrow=3, byrow = T),
         1, c(-.2,0,.2), c(1,1), 4)
*/



#include <omp.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List Estep_cpp0(const Rcpp::IntegerMatrix& data,
                     const Rcpp::NumericMatrix& item,
                     const Rcpp::NumericMatrix& grid,
                     const Rcpp::NumericVector& prior) {
  int N = data.nrow();
  int I = data.ncol();
  int M = grid.nrow();

  // log-likelihood accumulator
  std::vector<double> posterior_loglik_vec(N * M, 0.0);

  // accumulate over items
  for (int i = 0; i < I; i++) {
    IntegerVector data_col = data(_, i);
    NumericVector item_row = item(i, _);
    NumericMatrix ll = llik_cpp0(data_col, grid, item_row); // N x M (R object)

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


#include <omp.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List Estep_cpp(const Rcpp::IntegerMatrix& data,
                     const Rcpp::NumericMatrix& item,
                     const Rcpp::NumericMatrix& coord,
                     const Rcpp::NumericMatrix& grid,
                     const Rcpp::NumericVector& prior) {
  int N = data.nrow();
  int I = data.ncol();
  int M = grid.nrow();

  Rcpp::IntegerVector categories = max_data_cols(data);
  // log-likelihood accumulator
  std::vector<double> posterior_loglik_vec(N * M, 0.0);

  // accumulate over items
  for (int i = 0; i < I; i++) {
    IntegerVector data_col = data(_, i);
    NumericVector item_row = item(i, _);
    NumericVector coord_row = coord(i, _);
    int c = categories[i] + 1;  // number of categories
    double slope = item_row[0];  // C++ is 0-based
    Rcpp::NumericVector threshold = item_row[Rcpp::Range(1, item_row.size() - 1)];

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
  int categ = max_vec(categories) + 1;
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
/*** R
Estep_cpp0(matrix(c(1,0,
                   1,1,
                   0,1,
                   0,0), ncol=2, byrow = T),
          matrix(c(1,.2,1,-.2,
                   1.2,1,0,.2), nrow=2, byrow = T),
          matrix(c(1,0,0,
                   0,1,1,
                   0,0,0), nrow=3, byrow = T),
          c(0.2,.3,.4))

Estep_cpp(matrix(c(1,0,
                   1,1,
                   0,1,
                   0,0), ncol=2, byrow = T),
          matrix(c(1,.2,
                   1.2,1), nrow=2, byrow = T),
          matrix(c(1,-.2,
                   0,.2), nrow=2, byrow = T),
          matrix(c(1,0,0,
                   0,1,1,
                   0,0,0), nrow=3, byrow = T),
          c(0.2,.3,.4))

Estep_cpp(matrix(c(1,2,
                   3,2,
                   0,1,
                   2,1), ncol=2, byrow = T),
          matrix(c(1,-.2,0,.2,
                   1.2,-1,0,1), nrow=2, byrow = T),
          matrix(c(1,-.2,0,.2), nrow=2),
          matrix(c(1,0,0,
                   0,1,1,
                   0,0,0), nrow=3, byrow = T),
          c(0.2,.3,.4))
*/





#include <omp.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List L1L2_lsirm_cpp(const Rcpp::NumericMatrix& e_response,
                    const Rcpp::NumericVector& item,
                    const Rcpp::NumericVector& coord,
                    const Rcpp::NumericMatrix& grid,
                    int c) {  // number of categories
  int n_item  = item.size();    // slope + thresholds
  int n_coord = coord.size();   // spatial coordinates
  int npar    = n_item + n_coord;
  int M = grid.nrow();            // grid points
  int d = grid.ncol();            // dimension

  // sum over rows
  Rcpp::NumericVector f(M);
  for (int i = 0; i < M; i++) {
    double s = 0.0;
    for (int j = 0; j < e_response.ncol(); j++) s += e_response(i, j);
    f[i] = s;
  }

  // item parameters
  double slope = item[0];  // C++ is 0-based
  Rcpp::NumericVector threshold = item[Rcpp::Range(1, item.size() - 1)];

  if (threshold.size() != c - 1)
    stop("threshold length must be c-1");
  Rcpp::IntegerMatrix eta_beta = make_band_matrix(threshold.size());

  // Gradient and Information Matrix
  Rcpp::NumericVector gradient(npar);
  Rcpp::NumericMatrix IM(npar, npar);

  // elements per grid
  for (int m = 0; m < M; m++) {
    // distance term
    double sumsq = 0.0;
    for (int j = 1; j < d; j++) {
      double diff = grid(m, j) - coord[j - 1];
      sumsq += diff * diff;
    }
    double dist = std::sqrt(sumsq);

    // cumulative probs
    Rcpp::NumericVector c_prob(c + 1);
    c_prob[0] = 1;
    c_prob[c] = 0;
    Rcpp::NumericVector pd_sigmoid(c + 1, 0.0);
    for (int k = 0; k < (c - 1); k++) {
      c_prob[k + 1] =
        1.0 / (1.0 + std::exp(-(slope * grid(m, 0) - threshold[k] - dist)));
      pd_sigmoid[k + 1] = c_prob[k + 1] * (1- c_prob[k + 1]);
    }

    // chain rule eta/par
    Rcpp::NumericMatrix eta_par(npar, c + 1);
    for (int p = 0; p < npar; p++) {
      for (int k = 0; k < c; k++) {
        if (p == 0) {
          // derivative w.r.t. slope
          eta_par(p,k) = grid(m, 0);
        }
        else if (p < n_item) {
          // derivative w.r.t. threshold[p-1]
          eta_par(p,k) = eta_beta(p - 1,k);
        }
        else {
          // derivative w.r.t. item location
          if (dist > 0) {
            eta_par(p,k) = (grid(m, p-n_item+1) - coord[p-n_item]) / dist;
          } else {
            eta_par(p,k) = 0.0;
          }

        }
      }
    }
    // category probs
    Rcpp::NumericVector prob(c + 1);
    for (int k = 0; k < c; k++) {
      prob[k] = c_prob[k] - c_prob[k + 1];
    }


    // gradient
    for (int p = 0; p < npar; p++) {
      for (int k = 0; k < c; k++) {
        gradient[p] += e_response(m, k)/prob[k] * (eta_par(p,k)*pd_sigmoid[k] - eta_par(p,k + 1)*pd_sigmoid[k + 1]);
      }
    }

    // information matrix
    for (int p = 0; p < npar; p++) {
      for (int q = 0; q < npar; q++) {
        for (int k = 0; k < c; k++) {
          double p1 = eta_par(p,k)*pd_sigmoid[k] - eta_par(p,k + 1)*pd_sigmoid[k + 1];
          double q1 = eta_par(q,k)*pd_sigmoid[k] - eta_par(q,k + 1)*pd_sigmoid[k + 1];
          IM(p, q) += f[m]/prob[k] * p1 *q1;
        }
      }
    }

  }


  return List::create(
    _["gradient"] = gradient,
    _["IM"]       = IM
  );
}

/*** R
L1L2_lsirm_cpp(
  matrix(c(1,0,0,0,
           1,1,1,0,
           0,1,1,1,
           0,1,1,1), ncol=4, byrow = T),
  c(1,-.2,0,.2),
  c(1,-.2),
  matrix(c(1,0,0,
           0,1,1,
           0,0,0,
           .5,1,0), ncol=3, byrow = T),
  4
)
*/
