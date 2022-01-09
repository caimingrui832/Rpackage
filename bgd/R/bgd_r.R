sgd_lm <- function(X,y,beta_init= rep(0,ncol(X)+1),eps=1e-4,backtrack = FALSE,step_size=0.01,max_iter=10000,alpha=0.25,backtracking_beta=0.8){
  X_ <- cbind(1, X)
  z <- rcpp_mbgd_lm(X_,y,beta_init,eps,backtrack,step_size,max_iter,alpha,backtracking_beta,batch_size =1)
  names(z) <- list('beta','if_coverged','iters','loss')
  return(z)
}

bgd_lm <- function(X,y,beta_init= rep(0,ncol(X)+1),eps=1e-4,backtrack = FALSE,step_size=0.01,max_iter=10000,alpha=0.25,backtracking_beta=0.8){
  X_ <- cbind(1, X)
  z <- rcpp_mbgd_lm(X_,y,beta_init,eps,backtrack,step_size,max_iter,alpha,backtracking_beta,batch_size =nrow(X))
  names(z) <- list('beta','if_coverged','iters','loss')
  return(z)
}

mbgd_lm <- function(X,y,beta_init= rep(0,ncol(X)+1),eps=1e-4,backtrack = FALSE,step_size=0.01,
                    max_iter=10000,alpha=0.25,backtracking_beta=0.8,batch_size = nrow(X)){
  X_ <- cbind(1, X)
  z <- rcpp_mbgd_lm(X_,y,beta_init,eps,backtrack,step_size,max_iter,alpha,backtracking_beta,batch_size)
  names(z) <- list('beta','if_coverged','iters','loss')
  return(z)
}

Momentum_lm <- function(X,y,beta_init= rep(0,ncol(X)+1),eps=1e-4,backtrack = FALSE,step_size=0.01,
                        max_iter=10000,alpha=0.25,backtracking_beta=0.8,batch_size = nrow(X),rho = 0.9){
  X_ <- cbind(1, X)
  z <- rcpp_Momentum_lm(X_,y,beta_init,eps,backtrack,step_size,max_iter,alpha,backtracking_beta,batch_size,rho)
  names(z) <- list('beta','if_coverged','iters','loss')
  return(z)
}