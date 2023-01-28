#' Core algorithm for improving a linear classifier in terms of some metrics
#'
#' @description
#' Compute improved linear classifier to initialized linear classifier under certain metrics.
#'
#' @param X A \eqn{n \times p} feature matrix.
#' @param Y A \eqn{n \times 1} binary vector.
#' @param beta_init Slope coefficient for initialized discriminant function.
#' @param b_init Intercept coefficient for initialized discriminant function.
#' @param metrics Specify the metric to improve for initialized linear classifier.
#'
#'
#' @return beta: Generated slope coefficient
#' @return b: Generated intercept coefficient
#' @export
dccp <- function(X, Y, beta_init, b_init, gamma, psi_k, max_iter_num, max_rel_gap, metrics = "ROC") {
    if (metrics == "ROC") {
        .Call(`_imblearn_DCCP_ROC`, X, Y, beta_init, b_init, gamma, psi_k, max_iter_num, max_rel_gap)
    }
}