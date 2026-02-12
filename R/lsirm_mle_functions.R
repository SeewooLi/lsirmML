count_cat <- function(x) length(unique(x))
extract_cat <- function(x) sort(unique(x))

reorder_vec <- function(x){
  match(x, table = extract_cat(x))-1
}

reorder_mat <- function(x) apply(x, MARGIN = 2, FUN = reorder_vec)

# probability of endorsement
P_lsirm <- function(theta, param, a=1){
  if(is.matrix(theta)){
    theta[,1] <- theta[,1] * a
    eta <- sweep(theta, MARGIN = 2, STATS = param, FUN = "-")
  }else {
    theta[1] <- theta[1] * a
    eta <- sweep(-param, MARGIN = 2, STATS = theta, FUN = "+")
  }
  eta <- eta[,1] - sqrt(rowSums(eta[,-1]^2))
  return(1/(1+exp(- eta )))
}

# log-likelihood of a given item
llik <- function(data, theta, item){
  p_matrix <- P_lsirm(theta, item)
  p_matrix <- cbind(1 - p_matrix, p_matrix)
  likelihood <- sapply(1:nrow(theta), function(j) p_matrix[j, data+1])
  likelihood[is.na(likelihood)] <- 1
  return(log(likelihood))
}


#' Data Generation
#'
#' @param seed seed number for replication
#' @param N the number of respondents
#' @param nitem the number of items
#' @param gamma gamma parameter of LSIRM
#'
#' @return
#' @export
#'
#' @examples
data_generation <- function(seed = 1, N, nitem=NULL, gamma = 0.5, item=NULL, dimension=3){
  if(!is.null(item))dimension <- ncol(item)
  theta <- mvtnorm::rmvnorm(N,
                            mean = rep(0, dimension),
                            sigma = diag(c(1,rep(gamma^2, dimension-1))))
  if(is.null(item)){
    item <- cbind(rnorm(nitem),
                  c(rnorm(nitem/2, -gamma/2, gamma/10), rnorm(nitem/2, gamma/2, gamma/10)),
                  0
                  )
    item[,2:dimension][upper.tri(item[,2:dimension])] <- 0
  }else{
    nitem <- nrow(item)
  }

  data <- matrix(nrow = N, ncol = nitem)

  for(j in 1:nitem){
    ppp <- P_lsirm(theta, item[j,])
    for(i in 1:N){
      data[i,j] <- rbinom(1,1,ppp[i])
    }
  }
  return(list(
    data = data,
    item = item,
    theta = theta
  ))
}
# item <- matrix(c(rep(c(1,-1,0,0), each=5),
#                  rep(c(0,0,1,-1), each=5)), ncol=2)
# set.seed(1)
# item <- cbind(rnorm(10), item)
# dataset <- data_generation(N = 1000, item = item, gamma = 1)

# E-step
Estep <- function(data, item, grid, prior = NULL){
  d <- ncol(grid)
  if(is.null(prior)){
    mu <- rep(0, d)
    sigma <- diag(d)
    prior <- mvtnorm::dmvnorm(grid, mean = mu, sigma = sigma)
    prior <- prior/sum(prior)
  }

  # posterior distribution
  posterior <- matrix(0, nrow = nrow(data), ncol = nrow(grid))
  for(i in 1:nrow(item)){
    posterior <- posterior + llik(data[,i], grid, item[i,])
  }
  logL <- posterior
  posterior <- exp(sweep(posterior, 2, log(prior), FUN = "+"))
  posterior <- sweep(posterior, 1, 1/rowSums(posterior), FUN = "*")

  # expected frequency of response
  categ <- max(data, na.rm = TRUE) + 1
  e.response <- array(dim = c(nrow(item), nrow(grid), categ))
  for(i in 1:categ){
    d_ <- data==(i-1)
    d_[is.na(d_)] <- 0
    e.response[,,i] <- crossprod(d_,posterior)
  }

  # output
  freq <- colSums(posterior)
  logL <- sum(logL * posterior) + as.numeric(freq%*%log(prior)) - sum(posterior*log(posterior))
  return(
    list(
      posterior = posterior,
      freq = freq,
      e.response = e.response,
      grid = grid,
      prior = prior,
      logL = logL,
      Ak = freq/sum(freq)
    )
  )
}

# M-step
Mstep <- function(E, item, contrast_m, sds, max_iter = 5, threshold = 0.000001, model){
  estimated_item <- item

  IM <- list()

  iter <- 0
  repeat{
    iter <- iter + 1
    for(i in 1:nrow(item)){
      # index <- (ncol(item)*i-(ncol(item)-1)):(ncol(item)*i)

      L1L2 <- L1L2_lsirm_cpp(E$e.response[i,,], item, E$grid)
      # IM[index,index] <- L1L2$IM
      if(model== 1){
        diff <- as.vector(
          solve(L1L2$IM[-1,-1] + diag(1/(sds^2))) %*%
            (L1L2$gradient[-1] - ((estimated_item[i,-1]-0)/(sds^2)))
        )
        estimated_item[i,-1] <- estimated_item[i,-1] + diff * contrast_m[i,-1]
      } else if(model == 2){
        diff <- as.vector(
          solve(L1L2$IM + diag(c(0, 1/(sds^2)))) %*%
            (L1L2$gradient - ((estimated_item[i,]-0)/c(1,sds^2)))
        )
        estimated_item[i,] <- estimated_item[i,] + diff * contrast_m[i,]
      }
      IM[[i]] <- L1L2$IM + diag(c(0, 1/(sds^2)))
    }
    if(max(abs(estimated_item - item)) < threshold | iter > max_iter) break

    item <- estimated_item
  }

  return(list(
    estimated_item,
    IM
  ))
}

Mstep2 <- function(E, item, contrast_m, sds, max_iter = 5, threshold = 0.000001, model, max_cat){
  estimated_item <- item



  IM <- list()

  iter <- 0
  repeat{
    iter <- iter + 1
    for(i in 1:nrow(item)){
      prior_prec <- diag(
        c(rep(0,max_cat[i]+1), 1/(sds[-1]^2))
      )
      # prior_g_div <- c(rep(1,(max_cat[i]+1)), (sds[-1]^2))

      L1L2 <- L1L2_lsirm_cpp(E$e.response[i,,], item[i,1:(max_cat[i]+1)], item[i,(max_cat[i]+2):ncol(item)], E$grid, max_cat[i]+1)
      # IM[index,index] <- L1L2$IM
      if(model== 1){

        diff <- as.vector(
          solve(L1L2$IM[-1,-1] + prior_prec[-1,-1]) %*%
            (L1L2$gradient[-1] -
               c(rep(0,(max_cat[i])), (estimated_item[i,(max_cat[i]+2):ncol(item)]-0)/(sds[-1]^2))
             )
        )
        # diff <- as.vector(solve(L1L2$IM[-1,-1])%*%  (L1L2$gradient[-1]))
        estimated_item[i,-1] <- estimated_item[i,-1] + diff * contrast_m[i,-1]
      } else if(model == 2){
        diff <- as.vector(
          solve(L1L2$IM + prior_prec) %*%
            (L1L2$gradient -
               c(rep(0,(max_cat[i]+1)), (estimated_item[i,(max_cat[i]+2):ncol(item)]-0)/(sds[-1]^2))
             )
        )
        estimated_item[i,] <- estimated_item[i,] + diff * contrast_m[i,] / 2
      }
      IM[[i]] <- L1L2$IM + prior_prec
    }
    if(max(abs(estimated_item - item)) < threshold | iter > max_iter) break

    item <- estimated_item
  }

  return(list(
    estimated_item,
    IM
    ))
}

L1L2_lsirm <- function(e.response, par, grid){
  f <- rowSums(e.response)
  p0 <- P_lsirm(grid, par)

  eta_par <- sweep(grid[,-1], MARGIN = 2, STATS = par[-1], FUN = "-")
  eta_par <- sweep(eta_par, MARGIN = 1, STATS = sqrt(rowSums(eta_par^2)), FUN = "/")
  eta_par <- cbind(-1, eta_par)
  eta_par[is.na(eta_par)] <- 0

  gradient <- colSums(sweep(eta_par, 1, e.response[,2] - f * p0, "*"), na.rm = TRUE)
  IM <- t(eta_par) %*% diag(f * p0 * (1-p0)) %*% eta_par

  return(list(
    gradient = gradient,
    IM = IM
  ))
}

# Louis's (1982) correction for s.e.
se_correction <- function(e.response, quad, par_est, se_list, model){
  nitem <- nrow(par_est)
  se <- list()
  se_mat <- list()
  for(item in 1:nitem){
    f <- rowSums(e.response[item,,])
    # cnts <- colSums(e.response[item,,])
    p0 <- P_lsirm(quad, par_est[item,-1], par_est[item,1])

    eta_par <- sweep(quad[,-1], MARGIN = 2, STATS = par_est[item,-(1:2)], FUN = "-")
    eta_par <- sweep(eta_par, MARGIN = 1, STATS = sqrt(rowSums(eta_par^2)), FUN = "/")
    eta_par <- cbind(quad[,1], -1, eta_par)
    eta_par[is.na(eta_par)] <- 0


    E.sst <- t(eta_par) %*% diag((e.response[item,,2] - f * p0)^2) %*% eta_par

    grad <- colSums(sweep(eta_par, 1, e.response[item,,2] - f * p0, "*"), na.rm = TRUE)

    if(model == 1){
      se[[item]] <- se_list[[item]][-1,-1] - E.sst[-1,-1]
      se_list[[item]] <- se_list[[item]][-1,-1]
    }else if(model == 2){
      se[[item]] <- se_list[[item]] - E.sst
    }
    se_mat[[item]] <- solve(se[[item]])
  }
  return(list(
    complete = se_list,
    observed = se,
    se_mat_obs = se_mat
    ))
}

#' ML estimation of LSIRM
#'
#' @param data data matrix
#' @param dimension the number of dimension (main + interaction). The default is 3.
#' @param range range of the quadrature points
#' @param q the number of quadrature points
#' @param max_iter the maximum number of EM iteration
#' @param threshold the threshold to determine the EM convergence
#'
#' @return
#' @export
#'
#' @examples
#' \dontrun{
#' ################################################################################
#' # SIMULATION DATA
#' ################################################################################
#'
#' # data generation
#' dataset <- data_generation(seed = 1234,
#'                            N = 2000,
#'                            nitem = 10,
#'                            gamma = 1.0)
#'
#' data <- dataset$data
#' item <- dataset$item
#' theta <- dataset$theta
#'
#' # model fitting
#' fit <- lsirm(data,
#'              max_iter = 200,
#'              threshold = 0.001)
#'
#' # plotting
#' plot.lsirm(item)
#' plot.lsirm(fit$par_est)
#' plot.lsirm(fit)
#'
#' plot(item[,1], fit$par_est[,1])
#' abline(0,1)
#'
#' ################################################################################
#' # DRV DATA
#' ################################################################################
#' drv_data <- read.table("drv_data.txt")
#' fit.drv <- lsirm(drv_data,
#'                  max_iter = 200,
#'                  threshold = 0.001)
#'
#' plot.lsirm(fit.drv)
#' }
lsirm <- function(data,
                  dimension = 3,
                  model=1,
                  range = c(-4,4),
                  q = 11,
                  max_iter = 500,
                  threshold = 0.001,
                  contrast_m=NULL,
                  initial_item=NULL){
  args_list <- as.list(environment())

  ranges <- lapply(c(1,rep(0.8, dimension-1)), function(sd) range * sd)
  grid_list <- lapply(ranges, function(r) seq(r[1], r[2], length.out = q))
  grid <- as.matrix(do.call(expand.grid, grid_list))

  prior <- mvtnorm::dmvnorm(grid,
                            mean = rep(0, dimension),
                            sigma = diag(dimension))
  prior <- prior/sum(prior)
  sds <- rep(1, dimension)
  nitem <- ncol(data)

  set.seed(1)

  if(is.null(contrast_m)){
    contrast_m <- matrix(1, nrow = nitem, ncol = (dimension+1))
    contrast_m[,3:(dimension+1)][upper.tri(contrast_m[,3:(dimension+1)])] <- 0
  }
  if(is.null(initial_item)){
    initial_item <- cbind(1, 0, matrix(rnorm(nitem*(dimension-1),0,.1), nrow = nitem))
  }
  initial_item <- initial_item * contrast_m

  iter <- 0
  # EM_history <- list()
  repeat{
    iter <- iter + 1

    E <- Estep_cpp(data, initial_item, grid, prior)
    dim(E$e.response) <- c(nitem, nrow(grid), max(data, na.rm = TRUE) + 1)

    old_sds <- sds
    factor_means <- as.vector(E$Ak%*%E$grid)
    cov_mat <- t(E$grid) %*% sweep(E$grid, 1, E$Ak, FUN = "*") - factor_means %*% t(factor_means)
    sds <- sqrt(diag(cov_mat))
    sds[-1] <- sqrt(mean(sds[-1]^2))
    # sds[-1] <- sds[2]/sds[1]
    sds[1] <- 1

    M <- Mstep(E, initial_item, contrast_m, sds, model=model)

    M[[1]][,2] <- M[[1]][,2] - factor_means[1] * M[[1]][,1]
    M[[1]][,-(1:2)] <- sweep(M[[1]][,-(1:2)], 2, factor_means[-1], FUN = "-")
    M[[1]][which(contrast_m==0)] <- 0
    # M[[1]] <- sweep(M[[1]], 2, old_sds/sds, FUN = "*")
    # M[,1] <- M[,1]/sds[1]

    # EM_history[[iter]] <- M[[1]]

    ranges <- lapply(c(1,rep(0.8, dimension-1))*sds, function(sd) range * sd)
    grid_list <- lapply(ranges, function(r) seq(r[1], r[2], length.out = q))
    grid <- as.matrix(do.call(expand.grid, grid_list))

    prior <- mvtnorm::dmvnorm(grid,
                              mean = rep(0, dimension),
                              sigma = diag(sds^2))
    prior <- prior/sum(prior)

    diff <- max(abs(initial_item - M[[1]]), na.rm = TRUE)

    initial_item <- M[[1]]
    message("\r","\r",
            "EM cycle = ",iter,
            ", gamma = ",round(sds[2], 2),
            ", logL = ", sprintf("%.2f", E$logL),
            ", Max-Change = ",sprintf("%.5f",diff),sep="",appendLF=FALSE)
    flush.console()
    if(diff < threshold | iter >= max_iter) break
  }


  theta <- E$posterior%*%E$grid
  theta_se <- sqrt(E$posterior%*%(E$grid^2)-theta^2)
  return(list(
    par_est = initial_item,
    IM = se_correction(E$e.response, grid, initial_item, M[[2]], model),
    # EM_history = EM_history,
    fk=E$freq,
    iter=iter,
    quad=grid,
    diff=diff,
    prior=E$prior,
    posterior=E$posterior,
    Ak=E$Ak,
    theta = theta,
    theta_se = theta_se,
    logL= E$logL,
    f_cov = diag(sds^2),
    args_list = args_list,
    contrast_m = contrast_m
  ))
}

#' ML estimation of LSIRM
#'
#' @param data data matrix
#' @param dimension the number of dimension (main + interaction). The default is 3.
#' @param range range of the quadrature points
#' @param q the number of quadrature points
#' @param max_iter the maximum number of EM iteration
#' @param threshold the threshold to determine the EM convergence
#'
#' @return
#' @export
#'
#' @examples
#' \dontrun{
#' ################################################################################
#' # SIMULATION DATA
#' ################################################################################
#'
#' # data generation
#' dataset <- data_generation(seed = 1234,
#'                            N = 2000,
#'                            nitem = 10,
#'                            gamma = 1.0)
#'
#' data <- dataset$data
#' item <- dataset$item
#' theta <- dataset$theta
#'
#' # model fitting
#' fit <- lsirm(data,
#'              max_iter = 200,
#'              threshold = 0.001)
#'
#' # plotting
#' plot.lsirm(item)
#' plot.lsirm(fit$par_est)
#' plot.lsirm(fit)
#'
#' plot(item[,1], fit$par_est[,1])
#' abline(0,1)
#'
#' ################################################################################
#' # DRV DATA
#' ################################################################################
#' drv_data <- read.table("drv_data.txt")
#' fit.drv <- lsirm(drv_data,
#'                  max_iter = 200,
#'                  threshold = 0.001)
#'
#' plot.lsirm(fit.drv)
#' }
lsgrm <- function(data,
                  dimension = 3,
                  model=1,
                  range = c(-4,4),
                  q = 11,
                  max_iter = 500,
                  threshold = 0.001,
                  contrast_m = NULL,
                  initial_item=NULL){
  args_list <- as.list(environment())

  data <- reorder_mat(as.matrix(data))

  ranges <- lapply(c(1,rep(0.8, dimension-1)), function(sd) range * sd)
  grid_list <- lapply(ranges, function(r) seq(r[1], r[2], length.out = q))
  grid <- as.matrix(do.call(expand.grid, grid_list))

  prior <- mvtnorm::dmvnorm(grid,
                            mean = rep(0, dimension),
                            sigma = diag(dimension))
  prior <- prior/sum(prior)
  sds <- rep(1, dimension)
  nitem <- ncol(data)
  n_thres <- max(data[!is.na(data)])
  max_cat <- apply(data, 2, max, na.rm=TRUE)

  set.seed(1)

  if(is.null(initial_item)){
    threshold_mat <- matrix(nrow=nitem, ncol=n_thres)
    for(i in 1:nitem){
      threshold_mat[i, 1:max_cat[i]] <- seq(-.5, .5, length = max_cat[i])
    }
    initial_item <- cbind(1, threshold_mat, matrix(rnorm(nitem*(dimension-1),0,.1), nrow = nitem))
  }
  if(is.null(contrast_m)){
    contrast_m <- matrix(1, nrow = nitem, ncol = ncol(initial_item))
    contrast_m[,(n_thres+2):ncol(initial_item)][upper.tri(contrast_m[,(n_thres+2):ncol(initial_item)])] <- 0
  }

  initial_item <- initial_item * contrast_m


  iter <- 0
  # EM_history <- list()
  repeat{
    iter <- iter + 1

    E <- Estep_cpp(data, initial_item[,1:(1+n_thres)], initial_item[,((1:(dimension-1))+1+n_thres)], grid, prior)
    dim(E$e.response) <- c(nitem, nrow(grid), max(data, na.rm = TRUE) + 1)

    old_sds <- sds
    factor_means <- as.vector(E$Ak%*%E$grid)
    cov_mat <- t(E$grid) %*% sweep(E$grid, 1, E$Ak, FUN = "*") - factor_means %*% t(factor_means)
    sds <- sqrt(diag(cov_mat))
    sds[-1] <- sqrt(mean(sds[-1]^2))

    M <- Mstep2(E, initial_item, contrast_m, sds, model=model, max_cat=max_cat)


    if(model == 1) M[[1]][,1] <- M[[1]][,1] * sds[1]
    sds[1] <- 1
    M[[1]][,2:(1+n_thres)] <- sweep(M[[1]][,2:(1+n_thres), drop=FALSE], 1, factor_means[1] * M[[1]][,1], FUN = "-")
    M[[1]][,tail(seq_len(ncol(M[[1]])), 2)] <- sweep(M[[1]][,tail(seq_len(ncol(M[[1]])), 2)], 2, factor_means[-1], FUN = "-")
    M[[1]][which(contrast_m==0)] <- 0
    # M[[1]] <- sweep(M[[1]], 2, old_sds/sds, FUN = "*")
    # M[,1] <- M[,1]/sds[1]

    # EM_history[[iter]] <- M[[1]]

    ranges <- lapply(c(1,rep(0.8, dimension-1))*sds, function(sd) range * sd)
    grid_list <- lapply(ranges, function(r) seq(r[1], r[2], length.out = q))
    grid <- as.matrix(do.call(expand.grid, grid_list))

    prior <- mvtnorm::dmvnorm(grid,
                              mean = rep(0, dimension),
                              sigma = diag(sds^2))
    prior <- prior/sum(prior)

    diff <- max(abs(initial_item - M[[1]]), na.rm = TRUE)

    initial_item <- M[[1]]
    message("\r","\r",
            "EM cycle = ",iter,
            ", gamma = ",round(sds[2], 2),
            ", logL = ", sprintf("%.2f", E$logL),
            ", Max-Change = ",sprintf("%.5f",diff),sep="",appendLF=FALSE)
    flush.console()
    if(diff < threshold | iter >= max_iter) break
  }


  theta <- E$posterior%*%E$grid
  # theta_se <- sqrt(E$posterior%*%(E$grid^2)-theta^2)
  return(list(
    par_est = initial_item,
    # IM = se_correction(E$e.response, grid, initial_item, M[[2]], model),
    # EM_history = EM_history,
    fk=E$freq,
    iter=iter,
    quad=grid,
    diff=diff,
    prior=E$prior,
    posterior=E$posterior,
    Ak=E$Ak,
    theta = theta,
    # theta_se = theta_se,
    logL= E$logL,
    f_cov = diag(sds^2),
    args_list = args_list,
    contrast_m = contrast_m
  ))
}


#' Title
#'
#' @param data
#' @param item
#'
#' @returns
#' @export
#'
#' @examples
eap_score <- function(data, item, q=11){
  x <- seq(-4, 4, length.out=q)
  grid_list <- replicate(3, x, simplify = FALSE)
  grid <- as.matrix(do.call(expand.grid, grid_list))

  prior <- mvtnorm::dmvnorm(grid,
                            mean = rep(0, 3),
                            sigma = diag(3)
  )
  prior <- prior/sum(prior)

  E <- Estep(matrix(data, nrow=1), item, grid, prior)

  factor_means <- as.vector(E$Ak%*%E$grid)
  cov_mat <- t(E$grid) %*% sweep(E$grid, 1, E$Ak, FUN = "*") - factor_means %*% t(factor_means)

  return(list(
    theta = factor_means,
    se = cov_mat,
    grids = E$grid,
    heights = E$Ak
  ))
}

#' Title
#'
#' @param data
#' @param item
#'
#' @returns
#' @export
#'
#' @examples
map_score <- function(data, item){
  theta <- c(0,0,0)

  iter <- 0
  repeat{
    iter <- iter + 1

    eta0 <- sweep(-item, MARGIN = 2, STATS = theta, FUN = "+")
    dist <- sqrt(rowSums(eta0[,-1]^2))
    eta <- eta0[,1] - dist


    p0 <- 1/(1+exp(- eta ))

    eta1 <- cbind(1, sweep(-eta0[,-1], MARGIN = 1, STATS = 1/dist, FUN = "*"))

    l1 <- as.vector((data - p0) %*% eta1)

    H <- t(eta1) %*% diag(p0 * (1 - p0)) %*% eta1
    # diff <- l1[1] / H[1,1]
    diff <- as.vector((l1 - (theta - 0)) %*% solve(H + diag(3)))

    # theta[1] <- theta[1] + diff/2
    theta <- theta + diff/2

    if(sum(abs(diff)) < 0.000001 | iter > 50) break
  }

  return(list(
    theta = theta,
    se = solve(H + diag(3)),
    iter = iter
  ))
}

################################################################################
# PLOTTING
################################################################################
library(ggplot2)
#' Title
#'
#' @param item item parameters, or a list object from \code{lsirm}
#' @param range range of the coordinates
#'
#' @return
#' @export
#' @import ggplot2
#'
#' @examples
plot.lsirm <- function(item, range=c(-2.5, 2.5), ls_positions=NULL, gamma=NULL){
  if(is.null(gamma)) gamma <- 1
  if(is.list(item)){
    if(is.null(gamma)) gamma <- sqrt(item$f_cov[2,2])
    d <- item$args_list$dimension
    ls_positions <- as.data.frame(item$theta[,-1])
    item <- item$par_est
    item <- item[, tail(seq_len(ncol(item)), 2)]
  }

  if(is.null(rownames(item))){
    rownames(item) <- paste0("Q", 1:nrow(item))
  }
  df <- as.data.frame(item)
  colnames(df) <- c("Dim1", "Dim2")
  df$Label <- rownames(df)

  # Plot
  p <- ggplot2::ggplot() +
    # geom_point(size = 0) +
    ggplot2::geom_text(data = df, mapping = ggplot2::aes(x = Dim1, y = Dim2, label = Label), vjust = 0, hjust = 0) +
    ggplot2::theme_minimal() +
    ggplot2::coord_cartesian(xlim = range * gamma, ylim = range * gamma)+
    ggplot2::labs(title = NULL,
         x = "coordinate 1", y = "coordinate 2")
  if(!is.null(ls_positions)){
    colnames(ls_positions) <- c("coordinate1","coordinate2")

    p <- p +
      ggplot2::geom_point(data = ls_positions,
                 mapping = ggplot2::aes(x = coordinate1, y = coordinate2),
                 size = 0,
                 color="red",
                 alpha=0.7)
  }
  return(p)
}
