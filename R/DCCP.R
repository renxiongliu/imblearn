#' Core algorithm for improving a linear classifier in terms of some metrics
#'
#' @description
#' Compute improved linear classifier to initialized linear classifier under certain metrics.
#'
#' @param X A \eqn{n \times p} feature matrix.
#' @param Y A \eqn{n \times 1} binary vector with value -1 and +1.
#' @param beta_init Slope coefficient for initialized discriminant function.
#' @param b_init Intercept coefficient for initialized discriminant function.
#' @param gamma hyperparameter value of regularizer in the object function.
#' @param psi_k the hyperparameter in psi function: psi(x, psi_k) = min(1, max(0, 1-k*x))
#' @param max_iter_num maximal number of iterations in DCCP iterations.
#' @param max_rel_gap relative tolerance of object function value improvement during iteration.
#' @param metrics Specify the metric to improve for initialized linear classifier
#'                "ROC" means optimizing type I/II error while "PR" means optimize
#'                recall/precision.
#' @return beta: Generated slope coefficient
#' @return b: Generated intercept coefficient
#' @export
dccp <- function(X, Y, beta_init, b_init, gamma = 1e-3, psi_k = 10, max_iter_num = 25, max_rel_gap = 5 * 1e-3, metrics = "ROC") {
    if (metrics == "ROC") {
        .Call(`_imblearn_DCCP_ROC`, X, Y, beta_init, b_init, gamma, psi_k, max_iter_num, max_rel_gap)
    } else if (metrics == "PR") {
        .Call(`_imblearn_DCCP_PR`, X, Y, beta_init, b_init, gamma, psi_k, max_iter_num, max_rel_gap)
    } else {
        cat("\n The current version only supports 'ROC' and 'PR' metrics options.\n")
    }
}

#' Core algorithm for initializing solution
#'
#' @description
#' Compute initialized linear classifier for later usage
#' 
#' @importFrom LiblineaR LiblineaR
#' @param X A \eqn{n \times p} feature matrix.
#' @param Y A \eqn{n \times 1} binary vector with value -1 and +1.
#' @param w_neg class weight for negative label with value in (0,1).
#' @param C inverse of regularized hyperparameter in SVM;
#' @return list res with res$beta and res$b denoting the slope and intercept seperately
#' @export
initialize.svm <- function(X, Y, w_neg = 0.5, C = 1) {
    wi = c(1-w_neg, w_neg)
    names(wi) = c(1, -1)
    model <- LiblineaR(X, Y, type = 2, wi = wi, cost = C)
    params <- model$W
    p <- length(params) - 1
    return(list(beta = params[1:p], b = params[p + 1]))
}
