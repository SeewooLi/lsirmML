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
NumericVector P_lsirm_cpp(NumericMatrix theta, NumericVector param) {
  int n = theta.nrow();
  int m = theta.ncol();

  NumericVector result(n);

  for (int i = 0; i < n; i++) {
    double first = theta(i, 0) - param[0];
    double sumsq = 0.0;
    for (int j = 1; j < m; j++) {
      double diff = theta(i, j) - param[j];
      sumsq += diff * diff;
    }
    double eta = first - std::sqrt(sumsq);
    result[i] = 1.0 / (1.0 + std::exp(-eta));
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
                     0,0,0), nrow=3, byrow = T), c(0,1,1))
*/


// [[Rcpp::export]]
Rcpp::NumericMatrix llik_cpp(const Rcpp::IntegerVector& data,
                             const Rcpp::NumericMatrix& grid,
                             const Rcpp::NumericVector& item) {
  int N = data.size();   // persons
  int M = grid.nrow();   // grid points
  int d = grid.ncol();   // dimension

  Rcpp::NumericMatrix out(N, M);

  // loop over grid points
  for (int m = 0; m < M; m++) {
    // compute p(m | item) once
    double first = grid(m, 0) - item[0];
    double sumsq = 0.0;
    for (int j = 1; j < d; j++) {
      double diff = grid(m, j) - item[j];
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


/*** R
llik_cpp(c(0,1,NA),
         matrix(c(1,0,0,
                  0,1,1,
                  0,0,0), nrow=3, byrow = T),
         c(0,1,1))
*/


#include <omp.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List Estep_cpp(const Rcpp::IntegerMatrix& data,
                         const Rcpp::NumericMatrix& item,
                         const Rcpp::NumericMatrix& grid,
                         const Rcpp::NumericVector& prior) {
  int N = data.nrow();
  int I = data.ncol();
  int M = grid.nrow();

  // log-likelihood accumulator
  Rcpp::NumericMatrix posterior_loglik(N, M);

  // accumulate over items
  for (int i = 0; i < I; i++) {
    Rcpp::IntegerVector data_col = data(_, i);
    Rcpp::NumericVector item_row = item(i, _);
    Rcpp::NumericMatrix ll = llik_cpp(data_col, grid, item_row); // N×M

    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
      for (int m = 0; m < M; m++) {
        posterior_loglik(n, m) += ll(n, m);
      }
    }
  }

  // precompute log(prior)
  Rcpp::NumericVector log_prior(M);
  for (int m = 0; m < M; m++) log_prior[m] = std::log(prior[m]);

  // posterior = exp(loglik + log(prior)) + normalization
  Rcpp::NumericMatrix posterior(N, M);
  #pragma omp parallel for
  for (int n = 0; n < N; n++) {
    double rowsum = 0.0;
    for (int m = 0; m < M; m++) {
      double val = std::exp(posterior_loglik(n, m) + log_prior[m]);
      posterior(n, m) = val;
      rowsum += val;
    }
    if (rowsum > 0) {
      double inv = 1.0 / rowsum;
      for (int m = 0; m < M; m++) posterior(n, m) *= inv;
    }
  }

  // expected response frequencies
  int categ = max_data(data) + 1;
  Rcpp::NumericVector e_response(I * M * categ);

  #pragma omp parallel
  {
    Rcpp::NumericVector e_private(I * M * categ); // thread-local accumulator

    #pragma omp for nowait
    for (int n = 0; n < N; n++) {
      for (int i = 0; i < I; i++) {
        int val = data(n, i);
        if (val == NA_INTEGER) continue;
        for (int m = 0; m < M; m++) {
          e_private[i + I * m + I * M * val] += posterior(n, m);
        }
      }
    }

    #pragma omp critical
    {
    for (int idx = 0; idx < e_response.size(); idx++) {
      e_response[idx] += e_private[idx];
      }
    }
  }

  // freq = column sums
  Rcpp::NumericVector freq(M);
  #pragma omp parallel for
  for (int m = 0; m < M; m++) {
    double s = 0.0;
    for (int n = 0; n < N; n++) s += posterior(n, m);
    freq[m] = s;
  }

  // logL
  double total_logL = 0.0;
  #pragma omp parallel for reduction(+:total_logL)
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      double val = posterior(n, m);
      if (val > 0) {
        total_logL += posterior_loglik(n, m) * val;
        total_logL += val * log_prior[m];
        total_logL -= val * std::log(val);
      }
    }
  }

  // Ak
  double total_freq = std::accumulate(freq.begin(), freq.end(), 0.0);
  Rcpp::NumericVector Ak(M);
  if (total_freq > 0) {
    for (int m = 0; m < M; m++) Ak[m] = freq[m] / total_freq;
  }

  return Rcpp::List::create(
    Rcpp::_["posterior"]  = posterior,
    Rcpp::_["freq"]       = freq,
    Rcpp::_["e.response"] = e_response,
    Rcpp::_["grid"]       = grid,
    Rcpp::_["prior"]      = prior,
    Rcpp::_["logL"]       = total_logL,
    Rcpp::_["Ak"]         = Ak
  );
}

// [[Rcpp::export]]
List L1L2_lsirm_cpp(NumericMatrix e_response,
                    NumericVector par,
                    NumericMatrix grid) {
  int I = e_response.nrow();      // number of rows (I*M)
  int M = grid.nrow();            // grid points
  int d = grid.ncol();            // dimension

  // sum over rows
  NumericVector f(I);
  for (int i = 0; i < I; i++) {
    double s = 0.0;
    for (int j = 0; j < e_response.ncol(); j++) s += e_response(i, j);
    f[i] = s;
  }

  // P_lsirm
  NumericVector p0 = P_lsirm_cpp(grid, par);

  // compute eta_par = (grid[,-1] - par[-1]) / sqrt(rowSums(...))
  NumericMatrix eta_par(M, d);
  for (int i = 0; i < M; i++) {
    double sumsq = 0.0;
    for (int j = 1; j < d; j++) {
      eta_par(i, j) = grid(i, j) - par[j];
      sumsq += eta_par(i, j) * eta_par(i, j);
    }
    double norm = std::sqrt(sumsq);
    for (int j = 1; j < d; j++) {
      if (norm > 0) eta_par(i, j) /= norm;
      else eta_par(i, j) = 0.0;
    }
  }
  // prepend -1 in first column
  for (int i = 0; i < M; i++) eta_par(i, 0) = -1.0;

  // gradient: colSums(eta_par * (e_response[,2] - f*p0))
  int ncol_er = e_response.ncol();
  NumericVector gradient(d);
  for (int j = 0; j < d; j++) {
    double s = 0.0;
    for (int i = 0; i < M; i++) {
      double val = 0.0;
      if (ncol_er > 1) val = e_response(i, 1) - f[i] * p0[i]; // e_response[,2] - f*p0
      s += eta_par(i, j) * val;
    }
    gradient[j] = s;
  }

  // information matrix: t(eta_par) %*% diag(f*p0*(1-p0)) %*% eta_par
  NumericMatrix IM(d, d);
  for (int i = 0; i < M; i++) {
    double w = f[i] * p0[i] * (1.0 - p0[i]);
    for (int r = 0; r < d; r++) {
      for (int c = 0; c < d; c++) {
        IM(r, c) += eta_par(i, r) * w * eta_par(i, c);
      }
    }
  }

  return List::create(
    _["gradient"] = gradient,
    _["IM"]       = IM
  );
}


// // [[Rcpp::export]]
// List Mstep_cpp(List E,
//                NumericMatrix item,
//                NumericMatrix contrast_m,
//                NumericVector sds,
//                int max_iter = 5,
//                double threshold = 1e-6) {
//   int I = item.nrow();
//   int d = item.ncol();
//
//   NumericMatrix estimated_item = clone(item);
//   NumericMatrix IM(I*d, I*d);
//
//   int iter = 0;
//   bool done = false;
//
//   while (!done) {
//     iter++;
//
//     for (int i = 0; i < I; i++) {
//       // Compute linear index for information matrix
//       IntegerVector index(d);
//       for (int j = 0; j < d; j++) index[j] = i*d + j;
//
//       // Call L1L2_lsirm_cpp
//       NumericMatrix e_resp_i = E["e.response"];
//       // Extract slice for item i (I*M rows)
//       // Assuming e.response stored as 3D array flattened: I x M x categ
//       // We take slice e.response[i,,]
//       NumericMatrix grid = E["grid"];  // cast to NumericMatrix
//       int M = grid.nrow(); // number of grid points
//       int categ = e_resp_i.ncol() / M; // assume column-major flatten
//       NumericMatrix e_slice(M, categ);
//       for (int m = 0; m < M; m++) {
//         for (int c = 0; c < categ; c++) {
//           e_slice(m, c) = e_resp_i(i*M + m, c);
//         }
//       }
//
//       List L1L2 = L1L2_lsirm_cpp(e_slice, item(i,_), E["grid"]);
//
//       NumericMatrix IM_i = L1L2["IM"];
//       NumericVector grad_i = L1L2["gradient"];
//
//       // Fill IM block
//       for (int r = 0; r < d; r++) {
//         for (int c = 0; c < d; c++) {
//           IM(index[r], index[c]) = IM_i(r, c);
//         }
//       }
//
//       // Update estimated item
//       NumericMatrix IM_reg(d, d);
//       for (int j = 0; j < d; j++) {
//         for (int k = 0; k < d; k++) {
//           IM_reg(j, k) = IM_i(j, k);
//           if (j == k) IM_reg(j, k) += 1.0 / (sds[j]*sds[j]);
//         }
//       }
//
//       NumericVector rhs(d);
//       for (int j = 0; j < d; j++) {
//         rhs[j] = grad_i[j] - (estimated_item(i, j) - 0.0)/(sds[j]*sds[j]);
//       }
//
//       // Solve linear system
//       arma::mat A = as<arma::mat>(IM_reg);
//       arma::vec b = as<arma::vec>(rhs);
//       arma::vec x = arma::solve(A, b);
//       NumericVector diff = Rcpp::wrap(x);
//
//
//       for (int j = 0; j < d; j++) {
//         estimated_item(i, j) += diff[j] * contrast_m(i, j);
//       }
//     }
//
//     // Check convergence
//     double max_abs_diff = 0.0;
//     for (int i = 0; i < I; i++) {
//       for (int j = 0; j < d; j++) {
//         double delta = std::abs(estimated_item(i, j) - item(i, j));
//         if (delta > max_abs_diff) max_abs_diff = delta;
//       }
//     }
//
//     if (max_abs_diff < threshold || iter >= max_iter) done = true;
//
//     item = clone(estimated_item);
//   }
//
//   return List::create(
//     _["estimated_item"] = estimated_item,
//     _["IM"] = IM
//   );
// }
//
