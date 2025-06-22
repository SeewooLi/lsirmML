#' Title
#'
#' @param fit
#' @param threshold
#'
#' @returns
#' @export
#'
#' @examples
se <- function(fit, threshold = NULL){
  if(is.null(threshold)) threshold <- sqrt(fit$args_list$threshold)
  item <- fit$par_est
  dim <- ncol(item)

  contrast_m <- matrix(1, nrow = nrow(item), ncol = ncol(item))
  contrast_m[,2:dim][upper.tri(contrast_m[,2:dim])] <- 0
  position_fixed <- which(contrast_m == 0)
  IM_c <- fit$IM
  EM_history <- fit$EM_history

  n_cycle <- length(EM_history)
  n_par <- length(item)
  Del <- matrix(0, nrow = n_par, ncol = n_par)
  for(i in (n_cycle %/% 2):n_cycle){
    oldDel <- Del
    for(j in 1:n_par){
      item_temp <- item
      item_temp[j] <- EM_history[[i]][j]

      E <- Estep(fit$args_list$data, item_temp, fit$quad, fit$prior)
      M <- Mstep(E, item_temp, contrast_m, sqrt(diag(fit$f_cov)))

      Del[j,] <- as.vector(M[[1]] - item)/(EM_history[[i]][j]-item[j])
    }
    diff <- max(abs(
      Del[-position_fixed,-position_fixed]-oldDel[-position_fixed,-position_fixed]
    ))

    message("\r",i, ",  Diff: ",diff,sep="",appendLF=FALSE)
    flush.console()
    if ((i > 5) & (diff < sqrt(threshold))) break
  }
  reorder <- as.vector(t(matrix(1:length(item), ncol = ncol(item))))
  position_fixed <- which(reorder == position_fixed)
  Del <- Del[reorder,reorder]
  V_o <- solve(IM_c[-position_fixed,-position_fixed]) %*% solve(diag(rep(1,length(reorder)-1))-Del[-position_fixed,-position_fixed])

  V_o <- rbind(V_o[1:(position_fixed-1), , drop = FALSE],
                NA,
                V_o[position_fixed:nrow(V_o), , drop = FALSE])
  V_o <- cbind(V_o[,1:(position_fixed-1), drop = FALSE],
                NA,
                V_o[,position_fixed:ncol(V_o), drop = FALSE])

  se_mat <- matrix(sqrt(diag(V_o)), ncol = ncol(item), byrow = TRUE)

  return(list(
    V_o = V_o,
    se_mat = se_mat
  ))
}
