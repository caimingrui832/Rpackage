#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

double backtracking(arma::mat X, arma::vec y, arma::vec beta_k, arma::vec grad, double alpha=0.25, double backtracking_beta=0.8){
  int n = X.n_rows;
  double t(1.0);
  arma::vec beta_k1 = beta_k - t * grad;
  while((dot(y - X * beta_k1, y - X * beta_k1) - dot(y - X * beta_k, y - X * beta_k)) / 2 / n > - alpha * t * dot(grad, grad)){
    t = backtracking_beta * t;
    beta_k1 = beta_k - t * grad;
  }
  return t;
}

// [[Rcpp::export]]
Rcpp::List rcpp_mbgd_lm(arma::mat X, arma::vec y, arma::vec beta_init, double eps=1e-4,
                        bool backtrack = false, double step_size=0.01,
                        int max_iter=10000,double alpha=0.25, double backtracking_beta=0.8,
                        int batch_size = 20){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec beta = beta_init;
  double e(10000);
  double loss(dot(y - X * beta, y - X * beta) / 2 / n), loss_temp;
  int iter(0);
  arma::vec grad(p);
  
  arma::uvec indices(n);
  unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
  
  if(!backtrack){
    while(e >= eps && iter <= max_iter){
      for(int i = 0; i< n; i++){
        indices(i) = i;
      }
      shuffle(indices.begin(), indices.end(),std::default_random_engine (seed));
      arma::mat X_shuffled = X.rows(indices);
      arma::vec y_shuffled = y(indices);
      arma::mat xi;
      arma::vec yi;
      for(int i=0; i< n; i=i+batch_size){
        if(i+batch_size-1<= n-1){
          xi = X_shuffled.rows(i,i+batch_size-1);
          yi = y_shuffled.subvec(i,i+batch_size-1);
        }else{
          xi = X_shuffled.rows(i,n-1);
          yi = y_shuffled.subvec(i,n-1);
        }
        grad =  xi.t()*(xi*beta - yi)/batch_size;
        beta = beta - step_size * grad;
      }
      loss_temp = dot(y - X * beta, y - X * beta) / 2 / n;
      e = fabs((loss_temp - loss) / loss);
      iter += 1;
      loss = loss_temp;
    }
  }
  
  else{
    while(e >= eps && iter <= max_iter){
      for(int i = 0; i< n; i++){
        indices(i) = i;
      }
      shuffle(indices.begin(), indices.end(),std::default_random_engine (seed));
      arma::mat X_shuffled = X.rows(indices);
      arma::vec y_shuffled = y(indices);
      arma::mat xi;
      arma::vec yi;
      for(int i=0; i< n; i=i+batch_size){
        if(i+batch_size-1<= n-1){
          xi = X_shuffled.rows(i,i+batch_size-1);
          yi = y_shuffled.subvec(i,i+batch_size-1);
        }else{
          xi = X_shuffled.rows(i,n-1);
          yi = y_shuffled.subvec(i,n-1);
        }
        grad =  xi.t()*(xi*beta - yi)/batch_size;
        step_size = backtracking(X, y, beta, grad, alpha, backtracking_beta);
        beta = beta - step_size * grad;
      }
      loss_temp = dot(y - X * beta, y - X * beta) / 2 / n;
      e = fabs((loss_temp - loss) / loss);
      iter += 1;
      loss = loss_temp;
    }
  }
  
  
  Rcpp::CharacterVector if_coverged = "Not Converged!";
  if(e < eps){
    if_coverged = "Converged!";
    iter = iter + 1;
  }
  Rcpp::List result = List::create(beta,if_coverged,iter-1,loss);
  return result;
}


// [[Rcpp::export]]
Rcpp::List rcpp_Momentum_lm(arma::mat X, arma::vec y, arma::vec beta_init, double eps=1e-4,
                            bool backtrack = false, double step_size=0.01,
                            int max_iter=10000,double alpha=0.25, double backtracking_beta=0.8,
                            int batch_size =1,double rho = 0.9){
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec beta = beta_init;
  double e(10000);
  double loss(dot(y - X * beta, y - X * beta) / 2 / n), loss_temp;
  int iter(0);
  arma::vec grad(p);
  
  arma::vec v = zeros<vec>(p);
  arma::vec v_re = zeros<vec>(p);
  arma::uvec indices(n);
  unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();
  
  if(!backtrack){
    while(e >= eps && iter <= max_iter){
      for(int i = 0; i< n; i++){
        indices(i) = i;
      }
      shuffle(indices.begin(), indices.end(),std::default_random_engine (seed));
      arma::mat X_shuffled = X.rows(indices);
      arma::vec y_shuffled = y(indices);
      arma::mat xi;
      arma::vec yi;
      for(int i=0; i< n; i=i+batch_size){
        if(i+batch_size-1<= n-1){
          xi = X_shuffled.rows(i,i+batch_size-1);
          yi = y_shuffled.subvec(i,i+batch_size-1);
        }else{
          xi = X_shuffled.rows(i,n-1);
          yi = y_shuffled.subvec(i,n-1);
        }
        grad =  xi.t()*(xi*beta - yi)/batch_size;
        v = rho * v + (1-rho) * grad;
        v_re = (1/(1-pow(rho,iter+1)))* v;
        beta = beta - step_size * v_re;
      }
      loss_temp = dot(y - X * beta, y - X * beta) / 2 / n;
      e = fabs((loss_temp - loss) / loss);
      iter += 1;
      loss = loss_temp;
    }
  }
  
  else{
    while(e >= eps && iter <= max_iter){
      for(int i = 0; i< n; i++){
        indices(i) = i;
      }
      shuffle(indices.begin(), indices.end(),std::default_random_engine (seed));
      arma::mat X_shuffled = X.rows(indices);
      arma::vec y_shuffled = y(indices);
      arma::mat xi;
      arma::vec yi;
      for(int i=0; i< n; i=i+batch_size){
        if(i+batch_size-1<= n-1){
          xi = X_shuffled.rows(i,i+batch_size-1);
          yi = y_shuffled.subvec(i,i+batch_size-1);
        }else{
          xi = X_shuffled.rows(i,n-1);
          yi = y_shuffled.subvec(i,n-1);
        }
        grad =  xi.t()*(xi*beta - yi)/batch_size;
        v = rho * v + (1-rho) * grad;
        v_re = (1/(1-pow(rho,iter+1)))* v;
        step_size = backtracking(X, y, beta, grad = v_re, alpha, backtracking_beta);
        beta = beta - step_size * v_re;
      }
      loss_temp = dot(y - X * beta, y - X * beta) / 2 / n;
      e = fabs((loss_temp - loss) / loss);
      iter += 1;
      loss = loss_temp;
    }
  }
  
  
  Rcpp::CharacterVector if_coverged = "Not Converged!";
  if(e < eps){
    if_coverged = "Converged!";
    iter = iter + 1;
  }
  Rcpp::List result = List::create(beta,if_coverged,iter-1,loss);
  return result;
}
