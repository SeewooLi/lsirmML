# probability of endorsement
P_lsirm <- function(theta, param){
  eta <- sweep(theta, MARGIN = 2, STATS = param, FUN = "-")
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
data_generation <- function(seed = 1, N, nitem=NULL, gamma = 0.5, item=NULL){
  dimension <- 3
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
Mstep <- function(E, item, contrast_m, sds, max_iter = 5, threshold = 0.000001){
  estimated_item <- item

  IM <- matrix(0, nrow = length(item), ncol = length(item))

  iter <- 0
  repeat{
    iter <- iter + 1
    for(i in 1:nrow(item)){
      index <- (ncol(item)*i-(ncol(item)-1)):(ncol(item)*i)

      L1L2 <- L1L2_lsirm(E$e.response[i,,], item[i,], E$grid)
      IM[index,index] <- L1L2$IM
      diff <- as.vector(
        solve(L1L2$IM + diag(1/(sds^2))) %*%
          (L1L2$gradient - ((estimated_item[i,]-0)/(sds^2)))
        )
      estimated_item[i,] <- estimated_item[i,] + diff * contrast_m[i,]
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
#'                            gamma = 0.5)
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
lsirm <- function(data, dimension = 3, range = c(-4,4), q = 11, max_iter = 500, threshold = 0.001){
  args_list <- as.list(environment())

  x <- seq(range[1], range[2], length.out=q)
  grid_list <- replicate(dimension, x, simplify = FALSE)
  grid <- as.matrix(do.call(expand.grid, grid_list))
  prior <- mvtnorm::dmvnorm(grid,
                            mean = rep(0, dimension),
                            sigma = diag(dimension))
  prior <- prior/sum(prior)
  sds <- rep(1, dimension)
  nitem <- ncol(data)
  # initial_item <- matrix(rep(c(1,0,0), nitem), nrow = nitem, byrow = TRUE)
  set.seed(1)
  initial_item <- cbind(0, matrix(rnorm(nitem*2,0,.1), nrow = nitem))
  contrast_m <- matrix(1, nrow = nrow(initial_item), ncol = ncol(initial_item))
  contrast_m[,2:dimension][upper.tri(contrast_m[,2:dimension])] <- 0
  initial_item <- initial_item * contrast_m

  iter <- 0
  EM_history <- list()
  repeat{
    iter <- iter + 1

    E <- Estep(data, initial_item, grid, prior)

    old_sds <- sds
    factor_means <- as.vector(E$Ak%*%E$grid)
    cov_mat <- t(E$grid) %*% sweep(E$grid, 1, E$Ak, FUN = "*") - factor_means %*% t(factor_means)
    sds <- sqrt(diag(cov_mat))
    sds[-1] <- sqrt(mean(sds[-1]^2))
    # sds[-1] <- sds[2]/sds[1]
    sds[1] <- 1

    M <- Mstep(E, initial_item, contrast_m, sds)

    M[[1]] <- sweep(M[[1]], 2, factor_means, FUN = "-")
    M[[1]][which(contrast_m==0)] <- 0
    # M[[1]] <- sweep(M[[1]], 2, old_sds/sds, FUN = "*")
    # M[,1] <- M[,1]/sds[1]

    EM_history[[iter]] <- M[[1]]

    ranges <- lapply(sds, function(sd) c(-4, 4) * sd)
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
    IM = M[[2]],
    EM_history = EM_history,
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
    args_list = args_list
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
    ls_positions <- as.data.frame(item$theta[,-1])
    item <- item$par_est
  }

  if(is.null(rownames(item))){
    rownames(item) <- paste0("Q", 1:nrow(item))
  }
  df <- as.data.frame(item[,-1])
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
